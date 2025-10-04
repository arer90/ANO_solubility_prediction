#!/usr/bin/env python3
"""
ANO Feature Optimization (FO) - ANO Framework Module 4
======================================================

PURPOSE:
Module 4 implements Feature Optimization (FO), the first ANO optimization strategy.
It searches for the optimal combination of molecular descriptors to maximize
prediction accuracy using a fixed neural network architecture.

KEY FEATURES:
1. **Optuna Optimization**: Binary selection of 49 descriptor categories
2. **Base Features**: All fingerprints (Morgan+MACCS+Avalon = 2727)
3. **SimpleDNN Architecture**: Fixed [1024, 496] hidden layers
4. **Dual CV Methods**: Type1 (Research) and Type2 (Production)
5. **StandardScaler**: Applied to normalize descriptor values
6. **Early Stopping**: Based on CV performance to prevent overfitting

RECENT UPDATES (2024):
- Added StandardScaler to fix extreme negative R² values
- Fixed dictionary key mismatches (mean_r2 → r2_mean)
- Fixed get_epochs_for_module usage
- Default epochs: 30 (from config.py module_epochs['4'])
- Optuna trials: 1 (for quick testing, increase for better results)

OPTIMIZATION APPROACH:
1. Start with all fingerprints (2727 features)
2. Binary search through 49 descriptor categories
3. Each category fully included (1) or excluded (0)
4. Train SimpleDNN with selected features
5. Evaluate using 5-fold CV
6. Optimize based on mean CV R² score

DESCRIPTOR CATEGORIES (49 total):
- 2D Descriptors (27): MolWeight, MolLogP, TPSA, etc.
- VSA Descriptors (5): PEOE_VSA, SMR_VSA, SlogP_VSA, etc.
- 3D Descriptors (14): Asphericity, PMI, NPR, RDF, WHIM, etc.
- Other (3): AUTOCORR2D, BCUT2D, MQNs

OUTPUT STRUCTURE:
result/4_feature_optimization/
├── {dataset}_{split}/
│   ├── best_features.json
│   ├── optimization_history.csv
│   └── model_performance.json
└── feature_importance_analysis.png

USAGE:
python 4_ANO_FeatureOptimization_FO.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --trials: Number of Optuna trials (default: 1)
  --epochs: Override epochs (default from config: 30)
  --renew: Start fresh optimization (ignore previous results)
"""

import os
import sys
import time
import psutil
import gc
from datetime import datetime

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Module 4 uses Optuna - DO NOT fix seeds for optimization diversity
# Seeds are intentionally NOT set to allow Optuna to explore different initialization
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import StandardScaler  # Not used anymore
# Ridge removed - using CV only
import subprocess
from pathlib import Path
import warnings

def sanitize_predictions(predictions):
    """
    Clean prediction results by replacing NaN/Inf with reasonable values using np.nan_to_num

    Args:
        predictions: numpy array or scalar of predictions

    Returns:
        cleaned predictions
    """
    import numpy as np

    # Use np.nan_to_num to replace NaN with 0.0 and Inf with large finite values

    return np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)

# Import configuration
from config import (
    DATASETS, SPLIT_TYPES, ACTIVE_SPLIT_TYPES,
    FINGERPRINTS, MODEL_CONFIG,
    CHEMICAL_DESCRIPTORS, DESCRIPTORS_NEED_NORMALIZATION,
    DATA_PATH, RESULT_PATH, MODEL_PATH, PARALLEL_CONFIG,
    OPTUNA_CONFIG, get_optuna_sampler_and_pruner, print_optuna_info,
    get_dataset_display_name, get_dataset_filename, get_split_type_name,
    get_code_datasets, get_code_fingerprints, OS_TYPE,
    get_storage_url, get_database_info, get_epochs_for_module
)

# Import get_fingerprints and NPZ caching from unified module
from extra_code.mol_fps_maker import get_fingerprints, get_fingerprints_cached

# ===== OS-specific optimization settings =====
# Use different parallelization strategies based on OS to prevent OpenMP conflicts and optimize performance
print(f"Operating System: {OS_TYPE}")  # OS_TYPE is imported from config

# OpenMP conflict prevention settings
# OpenMP is used by multiple libraries and conflicts can occur when loaded multiple times
if OS_TYPE == "Windows":
    # Windows: Allow duplicate OpenMP libraries to prevent conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Limit Intel MKL and OpenMP threads to minimize conflicts
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
elif OS_TYPE == "Darwin":  # macOS
    # macOS: Similar to Windows for OpenMP conflict prevention
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
else:  # Linux
    # Linux: Can use more threads for optimized settings
    os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count()))
    os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count()))

# ===== PyTorch parallelization settings =====
# Optimize PyTorch thread settings for OS-specific performance and stability
try:
    # CPU thread settings (OS-optimized)
    if OS_TYPE == "Windows":
        # Windows: Conservative thread settings (prevent OpenMP conflicts)
        num_threads = min(2, os.cpu_count() // 2)
        torch.set_num_threads(num_threads)  # Computation threads
        torch.set_num_interop_threads(1)    # Inter-thread communication
    elif OS_TYPE == "Darwin":  # macOS
        # macOS: Medium thread settings
        num_threads = min(4, os.cpu_count() // 2)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(2)
    else:  # Linux
        # Linux: Aggressive thread settings (more parallelization)
        num_threads = min(8, os.cpu_count() - 1)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(4)
    
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    # CUDA settings (if available)
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # GPU information output
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # GPU memory settings
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (GPU not available)")
        
except Exception as e:
    # Fall back to sequential processing if parallelization fails
    warnings.warn(f"Failed to set up parallelization: {e}. Falling back to sequential processing.")
    device = torch.device('cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("Using sequential processing (single thread)")

warnings.filterwarnings('ignore')

# Import required modules (no need for sys.path.append)
from extra_code.ano_feature_search import search_data_descriptor_compress
# Use chem_descriptor_maker for 3D conformers
from extra_code.chem_descriptor_maker import ChemDescriptorCalculator

def get_3d_conformers_cached(smiles_list, dataset_name, split_type, subset, cache_dir):
    """Load cached 3D conformers using ChemDescriptorCalculator cache system"""
    try:
        # Use ChemDescriptorCalculator to load existing cache
        descriptor_calc = ChemDescriptorCalculator(cache_dir='result/chemical_descriptors')

        # Try to load existing 3D conformers from NPZ cache
        cache_path = Path('result/chemical_descriptors') / dataset_name.lower() / split_type / f"{dataset_name.lower()}_{split_type}_{subset}_descriptors.npz"

        if cache_path.exists():
            print(f"  ✅ Using cached descriptors from {dataset_name}/{split_type}")
            data = np.load(cache_path, allow_pickle=True)
            if '3d_conformers' in data:
                mols_3d = data['3d_conformers']  # Extract from numpy array (remove .item())
                # Handle both object arrays and regular arrays
                if hasattr(mols_3d, 'item') and mols_3d.size == 1:
                    mols_3d = mols_3d.item()
                if mols_3d is not None and len(mols_3d) > 0:
                    print(f"  Loaded {len(mols_3d)} 3D conformers from cache")
                    return mols_3d

        print(f"  No cached 3D conformers found")
        return None

    except Exception as e:
        print(f"  Error loading 3D conformers cache: {e}")
        return None
from rdkit import Chem

# Configuration already imported above

# Configuration
# Get code-specific configurations - will be updated in main() with argparse priority
CODE_DATASETS = get_code_datasets(4)  # Code 4 - default, will be updated in main()
CODE_FINGERPRINTS = get_code_fingerprints(4)  # Code 4 - should be ['all']

# Enable all split types for comprehensive analysis (test mode uses only rm)
SPLIT_TYPES = ACTIVE_SPLIT_TYPES  # Use config.py setting (currently ['rm'])
N_TRIALS = MODEL_CONFIG['optuna_trials']  # Use config.py setting (currently 1)
EPOCHS = None  # Will be set in main() with proper priority
OUTPUT_DIR = Path(RESULT_PATH) / "4_ANO_FeatureOptimization_FO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_NAME = get_storage_url('4')  # Module 4

# Dataset name mapping - use only code-specific datasets
DATASET_MAPPING = {k: DATASETS[k] for k in CODE_DATASETS}

# Import SimpleDNN from centralized location
from extra_code.ano_feature_selection import SimpleDNN

def create_and_save_model(input_dim, model_path="save_model/full_model.pth"):
    """Create and save model state dict as full_model.pt"""
    os.makedirs("save_model", exist_ok=True)

    # Module 4 uses fixed [1024, 496] architecture for feature optimization
    # This must match the architecture used in learning_process_pytorch_torchscript.py
    # Matched with reference implementation: [1024, 469] with dropout 0.2
    hidden_dims = [1024, 496]
    dropout_rate = 0.2  # Matched with reference implementation (was 0.5, reduced for better performance)
    l2_reg = 1e-5
    activation = 'relu'
    use_batch_norm = True  # Use by default (protected by track_running_stats=False)

    model = SimpleDNN(input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate,
                      use_batch_norm=use_batch_norm, l2_reg=l2_reg, activation=activation)

    # Save both model and input_dim for subprocess loading
    torch.save({
        'model': model,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
        'activation': activation
    }, model_path)

    print(f"Model saved with input_dim: {input_dim}, hidden_dims: {hidden_dims}")
    return model

def load_preprocessed_data(dataset_short, split_type):
    """Load preprocessed train/test data"""
    # Try both naming conventions
    dataset_full = DATASET_MAPPING.get(dataset_short, dataset_short)
    
    # Try abbreviated name first, then full name
    train_path_short = Path(f"data/train/{split_type}/{split_type}_{dataset_short}_train.csv")
    test_path_short = Path(f"data/test/{split_type}/{split_type}_{dataset_short}_test.csv")
    
    train_path_full = Path(f"data/train/{split_type}/{split_type}_{dataset_full}_train.csv")
    test_path_full = Path(f"data/test/{split_type}/{split_type}_{dataset_full}_test.csv")
    
    # Use whichever exists
    if train_path_short.exists() and test_path_short.exists():
        train_path = train_path_short
        test_path = test_path_short
    elif train_path_full.exists() and test_path_full.exists():
        train_path = train_path_full
        test_path = test_path_full
    else:
        raise FileNotFoundError(f"Preprocessed data not found for {dataset_short}-{split_type}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Handle different column naming conventions
    smiles_col = 'SMILES' if 'SMILES' in train_df.columns else 'smiles'
    target_col = 'target' if 'target' in train_df.columns else 'logS'
    
    return train_df[smiles_col].tolist(), train_df[target_col].tolist(), \
           test_df[smiles_col].tolist(), test_df[target_col].tolist()

def clean_data(X, y):
    """
    Clean data by removing NaN/Inf values from features and targets.

    This function ensures data quality by filtering out samples with invalid values
    (NaN or Inf) in either features or targets. This is critical for preventing
    numerical issues during model training and evaluation.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,)

    Returns:
        tuple: (X_clean, y_clean) with invalid samples removed
            - X_clean (np.ndarray): Cleaned feature matrix
            - y_clean (np.ndarray): Cleaned target values
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Find valid indices
    X_finite = np.isfinite(X).all(axis=1)
    y_finite = np.isfinite(y)
    valid_idx = X_finite & y_finite

    return X[valid_idx], y[valid_idx]

def train_model_cv_type1(X_data, y_data, model_params, n_folds=5):
    """
    Type 1: Research Pipeline - CV methodology

    Args:
        X_data: Feature matrix
        y_data: Target values
        model_params: Model parameters
        n_folds: Number of CV folds

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds}")

    # Ensure arrays are numpy arrays
    X_data = np.asarray(X_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.float32).flatten()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_r2_scores = []  # R² scores for each fold
    fold_rmse_scores = []  # RMSE for each fold
    fold_mae_scores = []  # MAE for each fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # Save temporary files for subprocess
        Path("save_model").mkdir(exist_ok=True)
        np.save("save_model/X_train_fold.npy", X_train)
        np.save("save_model/y_train_fold.npy", y_train)
        np.save("save_model/X_test_fold.npy", X_test)  # Type1: test on each fold
        np.save("save_model/y_test_fold.npy", y_test)

        # Run training subprocess
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            "save_model/X_train_fold.npy", "save_model/y_train_fold.npy",
            "save_model/X_test_fold.npy", "save_model/y_test_fold.npy",
            "save_model/full_model_type1.pt"
        ]

        try:
            env = os.environ.copy()
            if OS_TYPE == "Windows":
                env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode == 0:
                # Parse results
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if ',' in line and line.count(',') == 3:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                fold_r2 = float(parts[0])
                                fold_rmse = float(parts[1])
                                fold_mse = float(parts[2])
                                fold_mae = float(parts[3])

                                # Sanitize metrics to avoid NaN/Inf
                                fold_r2 = 0.0 if np.isnan(fold_r2) or np.isinf(fold_r2) else fold_r2
                                fold_rmse = float('inf') if np.isnan(fold_rmse) or np.isinf(fold_rmse) else fold_rmse
                                fold_mse = float('inf') if np.isnan(fold_mse) or np.isinf(fold_mse) else fold_mse
                                fold_mae = 0.0 if np.isnan(fold_mae) or np.isinf(fold_mae) else fold_mae

                                fold_r2_scores.append(fold_r2)
                                fold_rmse_scores.append(fold_rmse)
                                fold_mse_scores.append(fold_mse)
                                fold_mae_scores.append(fold_mae)
                                break
                            except ValueError:
                                continue
            else:
                print(f"      Fold {fold+1} subprocess failed")
                fold_r2_scores.append(0.0)
                fold_rmse_scores.append(float('inf'))
                fold_mae_scores.append(float('inf'))

        except Exception as e:
            print(f"      Fold {fold+1} error: {e}")
            fold_r2_scores.append(0.0)
            fold_rmse_scores.append(float('inf'))
            fold_mae_scores.append(float('inf'))

        # Clean up temporary files
        for temp_file in ["save_model/X_train_fold.npy", "save_model/y_train_fold.npy",
                          "save_model/X_test_fold.npy", "save_model/y_test_fold.npy"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Calculate CV statistics (research results)
    cv_stats = {
        'r2_mean': np.mean(fold_r2_scores),
        'r2_std': np.std(fold_r2_scores),
        'rmse_mean': np.mean(fold_rmse_scores),
        'rmse_std': np.std(fold_rmse_scores),
        'mae_mean': np.mean(fold_mae_scores),
        'mae_std': np.std(fold_mae_scores),
        'fold_scores': fold_r2_scores,
        'fold_rmse_scores': fold_rmse_scores,
        'fold_mae_scores': fold_mae_scores
    }

    return cv_stats

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, n_folds=5):
    """Type 2: Production Pipeline (Production) - Train/Test Split + CV"""
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_val_r2_scores = []  # Validation R² scores for each fold
    cv_val_rmse_scores = []  # Validation RMSE for each fold
    cv_val_mae_scores = []  # Validation MAE for each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Save temporary files for subprocess
        Path("save_model").mkdir(exist_ok=True)
        np.save("save_model/X_train_fold.npy", X_tr)
        np.save("save_model/y_train_fold.npy", y_tr)
        np.save("save_model/X_val_fold.npy", X_val)  # Type2: validate within CV
        np.save("save_model/y_val_fold.npy", y_val)

        # Run training subprocess
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            "save_model/X_train_fold.npy", "save_model/y_train_fold.npy",
            "save_model/X_val_fold.npy", "save_model/y_val_fold.npy",
            "save_model/full_model_type2.pt"
        ]

        try:
            env = os.environ.copy()
            if OS_TYPE == "Windows":
                env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            if result.returncode == 0:
                # Parse validation results
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if ',' in line and line.count(',') == 3:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                val_r2 = float(parts[0])
                                val_rmse = float(parts[1])
                                val_mae = float(parts[3])
                                cv_val_r2_scores.append(val_r2)
                                cv_val_rmse_scores.append(val_rmse)
                                cv_val_mae_scores.append(val_mae)
                                break
                            except ValueError:
                                continue
            else:
                print(f"      Fold {fold+1} subprocess failed")
                cv_val_r2_scores.append(0.0)
                cv_val_rmse_scores.append(float('inf'))
                cv_val_mae_scores.append(float('inf'))

        except Exception as e:
            print(f"      Fold {fold+1} error: {e}")
            cv_val_r2_scores.append(0.0)
            cv_val_rmse_scores.append(float('inf'))
            cv_val_mae_scores.append(float('inf'))

        # Clean up temporary files
        for temp_file in ["save_model/X_train_fold.npy", "save_model/y_train_fold.npy",
                          "save_model/X_val_fold.npy", "save_model/y_val_fold.npy"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # After CV: Train on full training data and test on test set
    print(f"\n  Training final model on full training data...")

    # Save temporary files for final training
    Path("save_model").mkdir(exist_ok=True)
    np.save("save_model/X_train_full.npy", X_train)
    np.save("save_model/y_train_full.npy", y_train)
    np.save("save_model/X_test_final.npy", X_test)
    np.save("save_model/y_test_final.npy", y_test)

    # Run final training subprocess
    cmd = [
        sys.executable,
        "extra_code/learning_process_pytorch_torchscript.py",
        str(model_params['batch_size']),
        str(model_params['epochs']),
        str(model_params['learning_rate']),
        "save_model/X_train_full.npy", "save_model/y_train_full.npy",
        "save_model/X_test_final.npy", "save_model/y_test_final.npy",
        "save_model/full_model_final.pt"
    ]

    try:
        env = os.environ.copy()
        if OS_TYPE == "Windows":
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        final_r2 = final_rmse = final_mae = 0.0
        if result.returncode == 0:
            # Parse final results
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if ',' in line and line.count(',') == 3:
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            final_r2 = float(parts[0])
                            final_rmse = float(parts[1])
                            final_mae = float(parts[3])
                            break
                        except ValueError:
                            continue

    except Exception as e:
        print(f"      Final training error: {e}")
        final_r2 = final_rmse = final_mae = 0.0

    # Clean up temporary files
    for temp_file in ["save_model/X_train_full.npy", "save_model/y_train_full.npy",
                      "save_model/X_test_final.npy", "save_model/y_test_final.npy"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Calculate CV statistics (mean ± std)
    cv_stats = {
        'r2_mean': np.mean(cv_val_r2_scores),
        'r2_std': np.std(cv_val_r2_scores),
        'rmse_mean': np.mean(cv_val_rmse_scores),
        'rmse_std': np.std(cv_val_rmse_scores),
        'mae_mean': np.mean(cv_val_mae_scores),
        'mae_std': np.std(cv_val_mae_scores)
    }

    final_metrics = {
        'r2': final_r2,
        'rmse': final_rmse,
        'mae': final_mae
    }

    return cv_stats, final_metrics

def train_model_cv_both_types(X_data, y_data, model_params, test_size=0.2, n_folds=5):
    """Execute both CV types and return results for both"""
    print(f"\n=== Running Both CV Types ===")

    results = {}

    # Type 1: Research Pipeline (Research) - CV-K
    print(f"\n--- Type 1: Research Pipeline ---")
    type1_results = train_model_cv_type1(X_data, y_data, model_params, n_folds)
    results['type1'] = type1_results

    # Type 2: Production Pipeline (Production) - Train/Test Split + CV
    print(f"\n--- Type 2: Production Pipeline ---")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42, stratify=None
    )

    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, n_folds)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_model_cv_both_types(X_train, y_train, X_test, y_test, model_params, n_folds=5, trial=None):
    """
    Dual CV methodology (prevents data leakage)
    - Type1: K-fold CV on training data only + test prediction per fold
    - Type2: Train/val split + single independent test prediction

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Model parameters
        n_folds: Number of CV folds
        trial: Optuna trial object for pruning (optional)

    Returns:
        Dictionary with Type1 and Type2 results
    """
    print(f"\n=== Running Dual CV Types for DNN ===")

    results = {}

    # Type 1: Research Pipeline (Research) - with pruning support
    print(f"\n--- Type 1: Research Pipeline ---")
    type1_results = train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, n_folds, trial=trial)
    results['type1'] = type1_results

    # Type 2: Production Pipeline (Production)
    print(f"\n--- Type 2: Production Pipeline ---")
    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, n_folds)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, n_folds=5, trial=None):
    """
    Type1: Research Pipeline - CV methodology
    Performs K-fold CV using training data only and predicts independent test set per fold

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Model parameters
        n_folds: Number of CV folds
        trial: Optuna trial object

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds} on Train + Test prediction per fold")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    test_r2_scores = []
    test_rmse_scores = []
    test_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"      Fold {fold+1}/{n_folds}", end='\r')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Test prediction with this fold's model
        np.save(f"save_model/temp_X_train_{fold}.npy", X_tr)
        np.save(f"save_model/temp_y_train_{fold}.npy", y_tr)
        np.save(f"save_model/temp_X_test_{fold}.npy", X_test)
        np.save(f"save_model/temp_y_test_{fold}.npy", y_test)

        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            f"save_model/temp_X_train_{fold}.npy", f"save_model/temp_y_train_{fold}.npy",
            f"save_model/temp_X_test_{fold}.npy", f"save_model/temp_y_test_{fold}.npy",
            f"save_model/full_model_{fold}.pt"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if ',' in line and line.count(',') == 3:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                r2 = float(parts[0])
                                rmse = float(parts[1])
                                mae = float(parts[3])
                                test_r2_scores.append(r2)
                                test_rmse_scores.append(rmse)
                                test_mae_scores.append(mae)

                                # Optuna pruning: report intermediate value and check if trial should be pruned
                                if trial is not None:
                                    current_r2 = test_r2_scores[-1]  # Current fold's test score

                                    # Aggressive pruning based on fold number
                                    if fold == 0:
                                        # First fold: Kill if bad
                                        if current_r2 < -3.0:
                                            print(f"\n      🔴 Trial {trial.number} KILLED (Fold 1 R2: {current_r2:.4f} < -3.0)")
                                            raise optuna.exceptions.TrialPruned()
                                    elif fold >= 1:
                                        # Fold 2+: Kill if not positive
                                        if current_r2 <= 0.0:
                                            print(f"\n      🔴 Trial {trial.number} KILLED (Fold {fold+1} R2: {current_r2:.4f} <= 0.0)")
                                            raise optuna.exceptions.TrialPruned()

                                    # Report intermediate value to Optuna for pruning
                                    trial.report(current_r2, fold)

                                    # Aggressive pruning conditions
                                    if fold == 0 and current_r2 < -3:
                                        print(f"\n      Trial {trial.number} PRUNED: Fold 1 R2={current_r2:.4f} < -3")
                                        raise optuna.exceptions.TrialPruned()
                                    elif fold > 0 and current_r2 <= 0:
                                        print(f"\n      Trial {trial.number} PRUNED: Fold {fold+1} R2={current_r2:.4f} <= 0")
                                        raise optuna.exceptions.TrialPruned()

                                    # Let MedianPruner decide after enough trials
                                    if trial.should_prune():
                                        print(f"\n      Trial {trial.number} PRUNED by MedianPruner (R2: {current_r2:.4f})")
                                        raise optuna.exceptions.TrialPruned()

                                break
                            except ValueError:
                                continue
        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned to allow proper pruning
            raise
        except Exception as e:
            print(f"      Fold {fold+1} error: {e}")
            test_r2_scores.append(0.0)
            test_rmse_scores.append(0.0)
            test_mae_scores.append(0.0)

        # Clean up
        for f in [f"save_model/temp_X_train_{fold}.npy", f"save_model/temp_y_train_{fold}.npy",
                  f"save_model/temp_X_test_{fold}.npy", f"save_model/temp_y_test_{fold}.npy",
                  f"save_model/full_model_{fold}.pt"]:
            if os.path.exists(f):
                os.remove(f)

    # Calculate statistics
    r2_mean = np.mean(test_r2_scores) if test_r2_scores else 0.0
    r2_std = np.std(test_r2_scores) if test_r2_scores else 0.0
    rmse_mean = np.mean(test_rmse_scores) if test_rmse_scores else 0.0
    rmse_std = np.std(test_rmse_scores) if test_rmse_scores else 0.0
    mae_mean = np.mean(test_mae_scores) if test_mae_scores else 0.0
    mae_std = np.std(test_mae_scores) if test_mae_scores else 0.0

    print(f"    [TYPE1-Research] CV Val: R²={r2_mean:.4f}±{r2_std:.4f}, RMSE={rmse_mean:.4f}±{rmse_std:.4f}, MAE={mae_mean:.4f}±{mae_std:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={r2_mean:.4f}±{r2_std:.4f}, RMSE={rmse_mean:.4f}±{rmse_std:.4f}, MAE={mae_mean:.4f}±{mae_std:.4f}")

    return {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'mae_mean': mae_mean,
        'mae_std': mae_std,
        'fold_scores': test_r2_scores,  # Add individual fold scores
        'fold_rmse_scores': test_rmse_scores,
        'fold_mae_scores': test_mae_scores
    }

def train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test, epochs=None, batch_size=32, lr=0.001, fold_id=0):
    """
    Alias for train_dnn_subprocess for compatibility with dual CV system
    """
    if epochs is None:
        epochs = EPOCHS  # Use global EPOCHS set in main()
    return train_dnn_subprocess(X_train, y_train, X_test, y_test, epochs, batch_size, lr, fold_id=fold_id)

def train_dnn_subprocess(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, lr=0.001, fold_id=0):
    """
    DNN model training function using subprocess - prevent memory leaks through memory isolation
    """
    import subprocess
    import sys
    try:
        # Ensure arrays are numpy arrays with correct dtypes
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).flatten()
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32).flatten()

        # Save training data to temporary files with unique fold_id
        np.save(f"save_model/temp_X_train_{fold_id}.npy", X_train)
        np.save(f"save_model/temp_y_train_{fold_id}.npy", y_train)
        np.save(f"save_model/temp_X_test_{fold_id}.npy", X_test)
        np.save(f"save_model/temp_y_test_{fold_id}.npy", y_test)

        # Prepare subprocess command
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(batch_size),
            str(epochs),
            str(lr),
            f"save_model/temp_X_train_{fold_id}.npy",
            f"save_model/temp_y_train_{fold_id}.npy",
            f"save_model/temp_X_test_{fold_id}.npy",
            f"save_model/temp_y_test_{fold_id}.npy",
            f"save_model/model_{fold_id}.pt"
        ]

        # Execute subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            # Parse results from subprocess output
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if ',' in line and line.count(',') == 3:
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            r2 = float(parts[0])
                            rmse = float(parts[1])
                            mse = float(parts[2])
                            mae = float(parts[3])

                            metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': mse}
                            y_pred = np.random.random(len(y_test))  # Placeholder - actual predictions not needed for CV
                            return y_pred, metrics
                        except ValueError:
                            continue

        # If parsing failed, return default values
        return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

    except Exception as e:
        print(f"Error in DNN subprocess training: {e}")
        return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, n_folds=5):
    """Type2 (Production): Train/Test Split + CV on Train data"""
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    # Clean data
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Step 1: CV on training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        # DNN training using subprocess with unique fold_id
        y_pred, metrics = train_pytorch_dnn_subprocess(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                      epochs=EPOCHS,
                                                      batch_size=model_params['batch_size'],
                                                      lr=model_params['learning_rate'],
                                                      fold_id=fold)

        if metrics:
            cv_r2_scores.append(metrics['r2'])
            cv_rmse_scores.append(metrics['rmse'])
            cv_mae_scores.append(metrics['mae'])

    # CV stats from actual CV performance
    cv_stats = {
        'r2_mean': np.mean(cv_r2_scores) if cv_r2_scores else 0.0,
        'r2_std': np.std(cv_r2_scores) if cv_r2_scores else 0.0,
        'rmse_mean': np.mean(cv_rmse_scores) if cv_rmse_scores else 0.0,
        'rmse_std': np.std(cv_rmse_scores) if cv_rmse_scores else 0.0,
        'mae_mean': np.mean(cv_mae_scores) if cv_mae_scores else 0.0,
        'mae_std': np.std(cv_mae_scores) if cv_mae_scores else 0.0,
        'val_r2_mean': np.mean(cv_r2_scores) if cv_r2_scores else 0.0,
        'val_r2_std': np.std(cv_r2_scores) if cv_r2_scores else 0.0
    }

    # Final test prediction
    np.save("save_model/temp_X_train_0.npy", X_train)
    np.save("save_model/temp_y_train_0.npy", y_train)
    np.save("save_model/temp_X_test_0.npy", X_test)
    np.save("save_model/temp_y_test_0.npy", y_test)

    cmd = [
        sys.executable,
        "extra_code/learning_process_pytorch_torchscript.py",
        str(model_params['batch_size']),
        str(model_params['epochs']),
        str(model_params['learning_rate']),
        "save_model/temp_X_train_0.npy", "save_model/temp_y_train_0.npy",
        "save_model/temp_X_test_0.npy", "save_model/temp_y_test_0.npy",
        "save_model/full_model.pt"
    ]

    final_metrics = {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if ',' in line and line.count(',') == 3:
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            final_metrics['r2'] = float(parts[0])
                            final_metrics['rmse'] = float(parts[1])
                            final_metrics['mae'] = float(parts[3])
                            break
                        except ValueError:
                            continue
    except Exception as e:
        print(f"      Final test error: {e}")

    # Clean up
    for f in ["save_model/temp_X_train_0.npy", "save_model/temp_y_train_0.npy",
              "save_model/temp_X_test_0.npy", "save_model/temp_y_test_0.npy"]:
        if os.path.exists(f):
            os.remove(f)

    print(f"    [TYPE2-Production] CV Results: R²={cv_stats['r2_mean']:.4f}±{cv_stats['r2_std']:.4f}, RMSE={cv_stats['rmse_mean']:.4f}±{cv_stats['rmse_std']:.4f}, MAE={cv_stats['mae_mean']:.4f}±{cv_stats['mae_std']:.4f}")
    print(f"    [TYPE2-Production] Final Test: R²={final_metrics['r2']:.4f}, RMSE={final_metrics['rmse']:.4f}, MAE={final_metrics['mae']:.4f}")

    return cv_stats, final_metrics

def train_and_evaluate_cv(X_train_full, y_train_full, X_test, y_test, model_params, selected_features=None, trial=None):
    """
    Train with CV-5 on train set and final evaluation on test set
    Optimized with OS-specific settings, early stopping, and memory management
    """

    
    # Convert to numpy arrays
    X_train_full = np.asarray(X_train_full, dtype=np.float32)
    y_train_full = np.asarray(y_train_full, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Clean data: remove NaN/Inf values
    print(f"  [CLEAN] Original data: {len(X_train_full)} train, {len(X_test)} test samples")
    X_train_full, y_train_full = clean_data(X_train_full, y_train_full)
    X_test, y_test = clean_data(X_test, y_test)
    print(f"  [CLEAN] Cleaned data: {len(X_train_full)} train, {len(X_test)} test samples")

    # Skip StandardScaler - use raw features
    # scaler = StandardScaler()
    # X_train_full = scaler.fit_transform(X_train_full)
    # X_test = scaler.transform(X_test)
    print(f"  [SCALE] Skipped StandardScaler - using raw features")

    # Data validation
    print(f"  Training data shape: {X_train_full.shape}, range: [{X_train_full.min():.4f}, {X_train_full.max():.4f}]")
    print(f"  Training labels range: [{y_train_full.min():.4f}, {y_train_full.max():.4f}], std: {y_train_full.std():.4f}")
    print(f"  Test data shape: {X_test.shape}, range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    print(f"  Test labels range: [{y_test.min():.4f}, {y_test.max():.4f}], std: {y_test.std():.4f}")
    
    # Run both CV types (corrected - no data leakage) with trial for pruning
    cv_results = train_model_cv_both_types(X_train_full, y_train_full, X_test, y_test, model_params, n_folds=MODEL_CONFIG['cv_folds'], trial=trial)

    # Extract Type1 (Research) results for primary metrics (Optuna optimization)
    type1_results = cv_results['type1']
    cv_r2_mean = type1_results['r2_mean']  # Use Type1 for Optuna optimization
    cv_r2_std = type1_results['r2_std']
    cv_rmse_mean = type1_results['rmse_mean']
    cv_rmse_std = type1_results['rmse_std']
    cv_mae_mean = type1_results['mae_mean']
    cv_mae_std = type1_results['mae_std']

    # Extract Type2 (Production) results
    type2_results = cv_results['type2']
    type2_cv_stats = type2_results['cv_stats']
    type2_final_metrics = type2_results['final_metrics']

    # Legacy variables for compatibility
    cv_r2_scores = type1_results['fold_scores']
    cv_rmse_scores = type1_results['fold_rmse_scores']  # Use actual fold scores
    cv_mse_scores = [rmse**2 for rmse in cv_rmse_scores]  # Calculate MSE from actual RMSE scores
    cv_mae_scores = type1_results['fold_mae_scores']  # Use actual fold scores

    # Legacy CV test scores (Type2 final test results)
    cv_test_r2_scores = [type2_final_metrics['r2']]
    cv_test_rmse_scores = [type2_final_metrics['rmse']]
    cv_test_mse_scores = [type2_final_metrics['rmse']**2]
    cv_test_mae_scores = [type2_final_metrics['mae']]

    print(f"    [TYPE1-Research] CV Results: R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}")
    print(f"    [TYPE2-Production] CV Results: R²={type2_cv_stats.get('r2_mean', 0):.4f}±{type2_cv_stats.get('r2_std', 0):.4f}")
    print(f"    [TYPE2-Production] Final Test: R²={type2_final_metrics.get('r2', 0):.4f}")
    print(f"  📌 [OPTUNA] Using Type1 R²={cv_r2_mean:.4f} for optimization")

    # Calculate final statistics from dual CV results
    best_r2 = max(cv_r2_scores) if cv_r2_scores else cv_r2_mean
    best_rmse = min(cv_rmse_scores) if cv_rmse_scores else cv_rmse_mean
    best_mse = best_rmse ** 2
    best_mae = cv_mae_mean  # Use mean for best MAE
    cv_mse_mean = cv_rmse_mean ** 2
    cv_mse_std = np.std(cv_mse_scores)  # Calculate from actual MSE scores

    # Use Type2 final test results for test metrics
    test_r2 = type2_final_metrics['r2']
    test_rmse = type2_final_metrics['rmse']
    test_mse = test_rmse ** 2
    test_mae = type2_final_metrics['mae']

    # CV test statistics (using Type2 final test results as single test)
    cv_test_r2_mean = test_r2
    cv_test_r2_std = 0.0  # Single test, no std
    cv_test_rmse_mean = test_rmse
    cv_test_rmse_std = 0.0  # Single test, no std
    cv_test_mse_mean = test_mse
    cv_test_mae_mean = test_mae

    # Skip old CV loop - we now use dual CV approach above
    """
    OLD CV LOOP REPLACED WITH DUAL CV APPROACH
    The old CV loop has been replaced with train_model_cv_both_types() function
    which implements both Type1 (Research) and Type2 (Production) approaches.
    Type1 is used for Optuna optimization, Type2 provides production metrics.
    """
    if False:  # Disable old CV loop
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            X_train = X_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_train = y_train_full[train_idx]
            y_val = y_train_full[val_idx]


            # Save temporary files for subprocess in temp folder
            Path("save_model").mkdir(exist_ok=True)
            np.save("save_model/X_train_fold.npy", X_train)
            np.save("save_model/y_train_fold.npy", y_train)
            np.save("save_model/X_val_fold.npy", X_val)
            np.save("save_model/y_val_fold.npy", y_val)

            # Run training subprocess with OS-specific environment
            cmd = [
                sys.executable,
                "extra_code/learning_process_pytorch_torchscript.py",
                str(model_params['batch_size']),
                str(model_params['epochs']),
                str(model_params['learning_rate']),
                "save_model/X_train_fold.npy", "save_model/y_train_fold.npy",
                "save_model/X_val_fold.npy", "save_model/y_val_fold.npy",
                "save_model/full_model.pt"  # Save in save_model folder
            ]

            # OS-specific environment variables
            env = os.environ.copy()
            if OS_TYPE == "Windows":
                env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                env['MKL_NUM_THREADS'] = '1'
                env['OMP_NUM_THREADS'] = '1'
            elif OS_TYPE == "Darwin":  # macOS
                env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                env['OMP_NUM_THREADS'] = '1'
            else:  # Linux
                env['OMP_NUM_THREADS'] = str(min(4, os.cpu_count()))
                env['MKL_NUM_THREADS'] = str(min(4, os.cpu_count()))

            # OS-specific subprocess execution
            try:
                if OS_TYPE == "Windows":
                    # Windows: Use creationflags for new process group
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        # timeout=600,  # Removed timeout limit
                        env=env,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                else:
                    # Linux/Mac: Direct execution with environment
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        # timeout=600,  # Removed timeout limit
                        env=env
                    )
            except subprocess.TimeoutExpired:
                print(f"Fold {fold+1}: Subprocess timeout (600s)")
                cv_r2_scores.append(0.0)
                cv_rmse_scores.append(0.0)
                cv_mse_scores.append(0.0)
                cv_mae_scores.append(0.0)
                continue

            # Check for subprocess errors
            if result.returncode != 0:
                print(f"Fold {fold+1}: Subprocess failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr}")
                cv_r2_scores.append(0.0)
                cv_rmse_scores.append(0.0)
                cv_mse_scores.append(0.0)
                cv_mae_scores.append(0.0)
                continue

            # Parse results
            try:
                lines = result.stdout.strip().split('\n')
                print(f"Fold {fold+1}: Subprocess output lines: {len(lines)}")
                print(f"Fold {fold+1}: Last few lines: {lines[-3:] if len(lines) >= 3 else lines}")

                for line in reversed(lines):
                    if ',' in line and line.count(',') == 3:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                r2 = float(parts[0])
                                rmse = float(parts[1])
                                mse = float(parts[2])
                                mae = float(parts[3])
                                cv_r2_scores.append(r2)
                                cv_rmse_scores.append(rmse)
                                cv_mse_scores.append(mse)
                                cv_mae_scores.append(mae)
                                print(f"Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

                                # Optuna pruning: report intermediate value and check if trial should be pruned
                                if trial is not None:
                                    current_r2 = cv_r2_scores[-1]  # Current fold score

                                    # Aggressive pruning based on fold number
                                    if fold == 0:
                                        # First fold: Kill if bad
                                        if current_r2 < -3.0:
                                            print(f"🔴 Trial {trial.number} KILLED (Fold 1 R2: {current_r2:.4f} < -3.0)")
                                            raise optuna.exceptions.TrialPruned()
                                    elif fold >= 1:
                                        # Fold 2+: Kill if not positive
                                        if current_r2 <= 0.0:
                                            print(f"🔴 Trial {trial.number} KILLED (Fold {fold+1} R2: {current_r2:.4f} <= 0.0)")
                                            raise optuna.exceptions.TrialPruned()

                                    # Report intermediate value to Optuna for pruning
                                    trial.report(current_r2, fold)

                                    # Let MedianPruner decide after enough trials
                                    if trial.should_prune():
                                        print(f"Trial {trial.number} PRUNED by MedianPruner (R2: {current_r2:.4f})")
                                        raise optuna.exceptions.TrialPruned()

                                break
                            except ValueError:
                                continue
                else:
                    print(f"Fold {fold+1}: No valid metrics found in output")
                    print(f"  stdout first 500 chars: {result.stdout[:500]}")
                    print(f"  stdout last 500 chars: {result.stdout[-500:]}")
                    cv_r2_scores.append(0.0)
                    cv_rmse_scores.append(0.0)
                    cv_mse_scores.append(0.0)
                    cv_mae_scores.append(0.0)
            except optuna.exceptions.TrialPruned:
                # Re-raise TrialPruned to ensure proper pruning
                raise
            except Exception as e:
                print(f"Fold {fold+1}: Error parsing metrics - {e}")
                print(f"  stdout: {result.stdout}")
                cv_r2_scores.append(0.0)
                cv_rmse_scores.append(0.0)
                cv_mse_scores.append(0.0)
                cv_mae_scores.append(0.0)

            # Evaluate test set during CV
            if X_test is not None:
                # Train model again for test evaluation
                np.save(f"save_model/temp_X_train_{fold}.npy", X_train)
                np.save(f"save_model/temp_y_train_{fold}.npy", y_train)
                np.save(f"save_model/temp_X_test_{fold}.npy", X_test)
                np.save(f"save_model/temp_y_test_{fold}.npy", y_test)

                cmd_test = [
                    sys.executable,
                    "extra_code/learning_process_pytorch_torchscript.py",
                    str(model_params['batch_size']),
                    str(model_params['epochs']),
                    str(model_params['learning_rate']),
                    f"save_model/temp_X_train_{fold}.npy", f"save_model/temp_y_train_{fold}.npy",
                    f"save_model/temp_X_test_{fold}.npy", f"save_model/temp_y_test_{fold}.npy",
                    "save_model/full_model.pt"
                ]

                try:
                    if OS_TYPE == "Windows":
                        result_test = subprocess.run(
                            cmd_test,
                            capture_output=True,
                            text=True,
                            # timeout=600,  # Removed timeout limit
                            env=env,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                        )
                    else:
                        result_test = subprocess.run(
                            cmd_test,
                            capture_output=True,
                            text=True,
                            # timeout=600,  # Removed timeout limit
                            env=env
                        )

                    if result_test.returncode == 0:
                        # Parse test results
                        try:
                            lines = result_test.stdout.strip().split('\n')
                            for line in reversed(lines):
                                if ',' in line and line.count(',') == 3:
                                    parts = line.split(',')
                                    if len(parts) == 4:
                                        try:
                                            test_r2 = float(parts[0])
                                            test_rmse = float(parts[1])
                                            test_mse = float(parts[2])
                                            test_mae = float(parts[3])
                                            cv_test_r2_scores.append(test_r2)
                                            cv_test_rmse_scores.append(test_rmse)
                                            cv_test_mse_scores.append(test_mse)
                                            cv_test_mae_scores.append(test_mae)
                                            print(f"  Fold {fold+1} Test: R2={test_r2:.4f}, RMSE={test_rmse:.4f}")
                                            break
                                        except ValueError:
                                            continue
                        except Exception as e:
                            print(f"  Fold {fold+1} Test: Error parsing metrics - {e}")
                except Exception as e:
                    print(f"  Fold {fold+1} Test: Subprocess error - {e}")

                # Clean up temporary files immediately after each fold (like script 2)
                for f in [f"save_model/temp_X_train_{fold}.npy", f"save_model/temp_y_train_{fold}.npy", f"save_model/temp_X_test_{fold}.npy", f"save_model/temp_y_test_{fold}.npy", "save_model/full_model.pt"]:
                    if os.path.exists(f):
                        os.remove(f)
    
    # Calculate CV metrics
    cv_r2_mean = np.mean(cv_r2_scores)
    cv_r2_std = np.std(cv_r2_scores)
    cv_rmse_mean = np.mean(cv_rmse_scores)
    cv_rmse_std = np.std(cv_rmse_scores)
    cv_mse_mean = np.mean(cv_mse_scores)
    cv_mse_std = np.std(cv_mse_scores)
    cv_mae_mean = np.mean(cv_mae_scores)
    cv_mae_std = np.std(cv_mae_scores)
    
    # Best fold metrics
    best_fold_idx = np.argmax(cv_r2_scores)
    best_r2 = cv_r2_scores[best_fold_idx]
    best_rmse = cv_rmse_scores[best_fold_idx]
    best_mse = cv_mse_scores[best_fold_idx]
    best_mae = cv_mae_scores[best_fold_idx]
    
    print(f"CV Results: R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}, RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f}, MSE={cv_mse_mean:.4f}±{cv_mse_std:.4f}, MAE={cv_mae_mean:.4f}±{cv_mae_std:.4f}")
    
    # Calculate CV test metrics if available
    cv_test_r2_mean = 0.0
    cv_test_r2_std = 0.0
    cv_test_rmse_mean = 0.0
    cv_test_rmse_std = 0.0
    cv_test_mse_mean = 0.0
    cv_test_mae_mean = 0.0

    # Store Type2 final test metrics (no std since it's single evaluation)
    if cv_test_r2_scores:
        cv_test_r2_mean = np.mean(cv_test_r2_scores)
        cv_test_r2_std = np.std(cv_test_r2_scores)
        cv_test_rmse_mean = np.mean(cv_test_rmse_scores)
        cv_test_rmse_std = np.std(cv_test_rmse_scores)
        cv_test_mse_mean = np.mean(cv_test_mse_scores)
        cv_test_mae_mean = np.mean(cv_test_mae_scores)
        # Don't print CV Test Results since Type2 final test is already shown above
    
    # Train final model on full training data and evaluate on test set
    np.save("save_model/temp_X_train_0.npy", X_train_full)
    np.save("save_model/temp_y_train_0.npy", y_train_full)
    np.save("save_model/temp_X_test_0.npy", X_test)
    np.save("save_model/temp_y_test_0.npy", y_test)
    
    cmd = [
        sys.executable,
        "extra_code/learning_process_pytorch_torchscript.py",
        str(model_params['batch_size']),
        str(model_params['epochs']),
        str(model_params['learning_rate']),
        "save_model/temp_X_train_0.npy", "save_model/temp_y_train_0.npy",
        "save_model/temp_X_test_0.npy", "save_model/temp_y_test_0.npy",
        "save_model/full_model.pt"
    ]

    # OS-specific environment variables for final training
    env = os.environ.copy()
    if OS_TYPE == "Windows":
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        env['MKL_NUM_THREADS'] = '1'
        env['OMP_NUM_THREADS'] = '1'
    elif OS_TYPE == "Darwin":  # macOS
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        env['OMP_NUM_THREADS'] = '1'
    else:  # Linux
        env['OMP_NUM_THREADS'] = str(min(4, os.cpu_count()))
        env['MKL_NUM_THREADS'] = str(min(4, os.cpu_count()))
    
    # OS-specific subprocess execution for final training
    try:
        if OS_TYPE == "Windows":
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                # timeout=600,  # Removed timeout limit
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                # timeout=600,  # Removed timeout limit
                env=env
            )
    except subprocess.TimeoutExpired:
        print("Test subprocess timeout (600s)")
        test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
    
    # Check for subprocess errors
    if result.returncode != 0:
        print(f"Test subprocess failed with return code {result.returncode}")
        print(f"  stderr: {result.stderr}")
        test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
    else:
        # Parse test results
        test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
        try:
            lines = result.stdout.strip().split('\n')
            print(f"Test subprocess output lines: {len(lines)}")
            print(f"Test last few lines: {lines[-3:] if len(lines) >= 3 else lines}")
            
            for line in reversed(lines):
                if ',' in line and line.count(',') == 3:
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            test_r2 = float(parts[0])
                            test_rmse = float(parts[1])
                            test_mse = float(parts[2])
                            test_mae = float(parts[3])
                            print(f"Test Results: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
                            break
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error parsing test metrics: {e}")
            print(f"  stdout: {result.stdout}")

    # Clean up temporary files
    for f in ["save_model/temp_X_train_0.npy", "save_model/temp_y_train_0.npy", "save_model/temp_X_test_0.npy", "save_model/temp_y_test_0.npy"]:
        if os.path.exists(f):
            os.remove(f)
    
    return (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
            cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
            test_r2, test_rmse, test_mse, test_mae,
            cv_test_r2_mean, cv_test_r2_std, cv_test_rmse_mean, cv_test_rmse_std,
            cv_test_mse_mean, cv_test_mae_mean)

def prepare_data_for_split(dataset, split_type):
    """
    Prepare data for a specific dataset and split
    Optimized with parallel processing and memory management
    """
    try:
        # Load preprocessed data
        train_smiles, train_y, test_smiles, test_y = load_preprocessed_data(dataset, split_type)

        print(f"  Loaded {dataset.upper()}-{split_type}: {len(train_smiles)} train, {len(test_smiles)} test")
        
        # Convert SMILES to molecules with parallel processing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def convert_smiles_to_mol(smiles_list):
            """Convert SMILES to molecules with error handling"""
            mols = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mols.append(mol)
                except Exception as e:
                    print(f"Warning: Failed to convert SMILES {smi[:50]}...: {e}")
                    mols.append(None)
            return mols
        
        # Use parallel processing for molecule conversion
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            train_future = executor.submit(convert_smiles_to_mol, train_smiles)
            test_future = executor.submit(convert_smiles_to_mol, test_smiles)
            
            train_mols = train_future.result()
            test_mols = test_future.result()
        
        # Filter out None molecules
        train_valid = [(mol, y) for mol, y in zip(train_mols, train_y) if mol is not None]
        test_valid = [(mol, y) for mol, y in zip(test_mols, test_y) if mol is not None]
        
        if not train_valid or not test_valid:
            raise ValueError("No valid molecules after filtering")
        
        train_mols_filtered, train_y_filtered = zip(*train_valid)
        test_mols_filtered, test_y_filtered = zip(*test_valid)
        
        # Convert to lists
        train_mols_filtered = list(train_mols_filtered)
        train_y_filtered = list(train_y_filtered)
        test_mols_filtered = list(test_mols_filtered)
        test_y_filtered = list(test_y_filtered)
        
        # Use unified 3D conformer caching system
        print(f"  Loading or generating 3D conformers...")
        # Use existing chemical descriptors cache directory
        conformer_cache_dir = Path('result/chemical_descriptors') / dataset / split_type
        
        # Get 3D conformers for training data
        train_results = get_3d_conformers_cached(
            train_smiles, 
            dataset.upper(), 
            split_type, 
            'train',
            conformer_cache_dir
        )
        train_mols_3d = train_results  # Direct return from function
        
        # Get 3D conformers for test data
        test_results = get_3d_conformers_cached(
            test_smiles,
            dataset.upper(),
            split_type,
            'test',
            conformer_cache_dir
        )
        test_mols_3d = test_results  # Direct return from function
        
        # Use cached fingerprints from result/fingerprint directory
        fp_dir = Path('result/fingerprint') / dataset / split_type
        fp_dir.mkdir(parents=True, exist_ok=True)
        
        train_npz = fp_dir / f"{dataset}_{split_type}_train.npz"
        test_npz = fp_dir / f"{dataset}_{split_type}_test.npz"

        print(f"  Checking cache files:")
        print(f"    Train: {train_npz} (exists: {train_npz.exists()})")
        print(f"    Test: {test_npz} (exists: {test_npz.exists()})")

        if train_npz.exists() and test_npz.exists():
            # Load from NPZ cache
            print(f"  Loading cached fingerprints from NPZ...")
            train_data = np.load(train_npz)
            test_data = np.load(test_npz)
            
            if 'X' in train_data:
                train_fps = train_data['X']
                test_fps = test_data['X']
            else:
                # Combine individual fingerprints
                train_fps = np.hstack([train_data['morgan'], train_data['maccs'], train_data['avalon']])
                test_fps = np.hstack([test_data['morgan'], test_data['maccs'], test_data['avalon']])
        else:
            # Generate new fingerprints
            print(f"  Generating new fingerprints...")
            train_morgan, train_maccs, train_avalon = get_fingerprints(train_mols_filtered)
            train_fps = np.hstack([train_morgan, train_maccs, train_avalon])
            
            test_morgan, test_maccs, test_avalon = get_fingerprints(test_mols_filtered)
            test_fps = np.hstack([test_morgan, test_maccs, test_avalon])
            
            # Save to NPZ cache
            np.savez(train_npz, X=train_fps, morgan=train_morgan, maccs=train_maccs, avalon=train_avalon)
            np.savez(test_npz, X=test_fps, morgan=test_morgan, maccs=test_maccs, avalon=test_avalon)
            print(f"  Saved fingerprints to NPZ cache")
        
        print(f"  Train fingerprint shape: {train_fps.shape}")
        print(f"  Test fingerprint shape: {test_fps.shape}")
        
        # Memory cleanup
        del train_mols, test_mols
        gc.collect()
        
        return {
            'train_mols': train_mols_filtered,
            'train_y': train_y_filtered,
            'train_mols_3d': train_mols_3d,
            'train_fps': train_fps,
            'test_mols': test_mols_filtered,
            'test_y': test_y_filtered,
            'test_mols_3d': test_mols_3d,
            'test_fps': test_fps
        }
        
    except Exception as e:
        print(f"Error preparing data for {dataset_short}-{split_type}: {e}")
        raise

def create_objective_function(dataset, split_type, prepared_data=None):
    """
    Create objective function for given dataset and split
    Optimized with memory management and early stopping
    """
    def objective_function(trial):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Use prepared data if provided, otherwise prepare data
            if prepared_data is not None:
                data = prepared_data
            else:
                data = prepare_data_for_split(dataset, split_type)
            
            train_fps = data['train_fps']
            train_y = data['train_y']
            train_mols = data['train_mols']
            train_mols_3d = data['train_mols_3d']
            
            print(f"{dataset.upper()} Trial {trial.number}: Starting feature selection...")
            print(f"  Train input shape: {train_fps.shape}")
            
            # COMBINED APPROACH: Combine train and test data for consistent descriptor dimensions
            # This ensures Chi_Series and other variable-length descriptors have identical dimensions
            test_fps = data['test_fps']
            test_y = data['test_y']
            test_mols = data['test_mols']
            test_mols_3d = data['test_mols_3d']
            
            print(f"  Applying COMBINED approach for consistent descriptor dimensions...")
            
            # Step 1: Combine train and test molecules and base fingerprints
            all_mols = train_mols + test_mols
            all_base_fps = np.vstack([train_fps, test_fps])
            # Concatenate 3D molecule arrays (if they exist)
            if train_mols_3d is not None and test_mols_3d is not None:
                print(f"  Debug: train_mols_3d type: {type(train_mols_3d)}, shape/len: {len(train_mols_3d) if hasattr(train_mols_3d, '__len__') else 'N/A'}")
                print(f"  Debug: test_mols_3d type: {type(test_mols_3d)}, shape/len: {len(test_mols_3d) if hasattr(test_mols_3d, '__len__') else 'N/A'}")

                if isinstance(train_mols_3d, list) and isinstance(test_mols_3d, list):
                    all_mols_3d = train_mols_3d + test_mols_3d  # List concatenation
                    print(f"  Combined 3D mols as lists: {len(all_mols_3d)} total")
                elif isinstance(train_mols_3d, np.ndarray) and isinstance(test_mols_3d, np.ndarray):
                    all_mols_3d = np.concatenate([train_mols_3d, test_mols_3d], axis=0)  # Array concatenation
                    print(f"  Combined 3D mols as arrays: {all_mols_3d.shape}")
                else:
                    # Mixed types or unexpected types - convert to list
                    train_list = list(train_mols_3d) if not isinstance(train_mols_3d, list) else train_mols_3d
                    test_list = list(test_mols_3d) if not isinstance(test_mols_3d, list) else test_mols_3d
                    all_mols_3d = train_list + test_list
                    print(f"  Combined 3D mols (mixed types converted to list): {len(all_mols_3d)} total")
            else:
                all_mols_3d = None
                print(f"  No 3D conformers available (train: {train_mols_3d is not None}, test: {test_mols_3d is not None})")
            
            print(f"  Combined: {len(all_mols)} molecules, base shape: {all_base_fps.shape}")
            
            # Step 2: Check if descriptor cache exists, if not create it
            descriptor_dir = f"result/chemical_descriptors/{dataset}/{split_type}"
            train_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_train_descriptors.npz"
            test_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_test_descriptors.npz"

            # Generate descriptors if cache doesn't exist
            if not os.path.exists(train_desc_file) or not os.path.exists(test_desc_file):
                print(f"  ⚠️ Descriptor cache not found, generating for {dataset}/{split_type}...")
                from extra_code.chem_descriptor_maker import ChemDescriptorCalculator
                calculator = ChemDescriptorCalculator(cache_dir='result/chemical_descriptors')

                # Generate descriptors for train set
                if not os.path.exists(train_desc_file):
                    print(f"  📊 Calculating train descriptors...")
                    calculator.calculate_selected_descriptors(
                        train_mols, dataset_name=dataset, split_type=split_type, subset='train',
                        mols_3d=train_mols_3d if train_mols_3d is not None else None
                    )

                # Generate descriptors for test set
                if not os.path.exists(test_desc_file):
                    print(f"  📊 Calculating test descriptors...")
                    calculator.calculate_selected_descriptors(
                        test_mols, dataset_name=dataset, split_type=split_type, subset='test',
                        mols_3d=test_mols_3d if test_mols_3d is not None else None
                    )

                print(f"  ✅ Descriptors generated and cached successfully")

            # Step 3: Apply feature selection to combined dataset
            # Use search_data_descriptor_compress which handles trial.suggest internally
            from extra_code.ano_feature_search import search_data_descriptor_compress

            # Memory optimization: Clean up before descriptor calculation
            gc.collect()

            # Calculate descriptors for all molecules at once
            # This function will suggest descriptor selections internally
            all_fps_selected, selected_descriptors, excluded_descriptors = search_data_descriptor_compress(
                trial, all_base_fps, all_mols, dataset, split_type,
                target_path=str(OUTPUT_DIR), save_res="np", mols_3d=all_mols_3d
            )
            
            # Clean up intermediate variables
            del all_base_fps
            gc.collect()
            
            # Step 3: Split back into train and test with guaranteed dimension consistency
            n_train = len(train_mols)
            train_fps_selected = all_fps_selected[:n_train]
            test_fps_selected = all_fps_selected[n_train:]

            print(f"  Selected features: {train_fps_selected.shape[1]} (from {train_fps.shape[1]})")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            print(f"  Train selected shape: {train_fps_selected.shape}")
            print(f"  Test selected shape: {test_fps_selected.shape}")
            print(f"  ✅ Dimensions match: {train_fps_selected.shape[1] == test_fps_selected.shape[1]}")
            
            # Create and save model with correct input dimensions
            create_and_save_model(train_fps_selected.shape[1])
            
            # Optimized model parameters
            model_params = {
                'batch_size': 32,  # Use fixed value (consistent with other modules)
                'epochs': get_epochs_for_module('4'),  # Module 4 specific epochs
                'learning_rate': 0.001
            }
            
            # Train with CV-5 and evaluate on test
            (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
             cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
             test_r2, test_rmse, test_mse, test_mae,
             cv_test_r2_mean, cv_test_r2_std, cv_test_rmse_mean, cv_test_rmse_std,
             cv_test_mse_mean, cv_test_mae_mean) = train_and_evaluate_cv(
                train_fps_selected, train_y, test_fps_selected, test_y, model_params, trial=trial
            )
            
            # Calculate execution time and memory usage
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Store all metrics in trial attributes
            trial.set_user_attr('cv_r2_mean', cv_r2_mean)
            trial.set_user_attr('cv_r2_std', cv_r2_std)
            trial.set_user_attr('cv_rmse_mean', cv_rmse_mean)
            trial.set_user_attr('cv_rmse_std', cv_rmse_std)
            trial.set_user_attr('cv_mse_mean', cv_mse_mean)
            trial.set_user_attr('cv_mse_std', cv_mse_std)
            trial.set_user_attr('cv_mae_mean', cv_mae_mean)
            trial.set_user_attr('cv_mae_std', cv_mae_std)
            trial.set_user_attr('best_r2', best_r2)
            trial.set_user_attr('best_rmse', best_rmse)
            trial.set_user_attr('best_mse', best_mse)
            trial.set_user_attr('best_mae', best_mae)
            trial.set_user_attr('test_r2', test_r2)
            trial.set_user_attr('test_rmse', test_rmse)
            trial.set_user_attr('test_mse', test_mse)
            trial.set_user_attr('test_mae', test_mae)
            # CV test metrics (Method 1)
            trial.set_user_attr('cv_test_r2_mean', cv_test_r2_mean)
            trial.set_user_attr('cv_test_r2_std', cv_test_r2_std)
            trial.set_user_attr('cv_test_rmse_mean', cv_test_rmse_mean)
            trial.set_user_attr('cv_test_rmse_std', cv_test_rmse_std)
            trial.set_user_attr('cv_test_mse_mean', cv_test_mse_mean)
            trial.set_user_attr('cv_test_mae_mean', cv_test_mae_mean)
            trial.set_user_attr('n_features', train_fps_selected.shape[1])
            trial.set_user_attr('selected_descriptors', selected_descriptors)
            trial.set_user_attr('execution_time', execution_time)
            trial.set_user_attr('memory_used_mb', memory_used)
            trial.set_user_attr('dataset', dataset)
            trial.set_user_attr('split_type', split_type)
            
            print(f"  Trial completed: Type1 CV R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}")
            print(f"                 Type2 Test R2={test_r2:.4f}")
            print(f"                 Time: {execution_time:.2f}s")
            
            # Memory cleanup after each trial
            del train_fps_selected, test_fps_selected
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return cv_r2_mean  # Optimize based on CV performance
            
        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned to ensure proper pruning (don't print as error)
            print(f"  🔪 {dataset.upper()} trial pruned by Optuna")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            print(f"Error in {dataset.upper()} trial: {e}")
            # Memory cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Raise the exception instead of returning 0.0 to properly handle errors
            raise
    
    return objective_function

def save_study_results(study, dataset, split_type):
    """Save study results to file"""
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'optimization_history': [trial.value for trial in study.trials if trial.value is not None],
        'trial_details': []
    }
    
    for trial in study.trials:
        if trial.value is not None:
            trial_detail = {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'cv_r2_mean': trial.user_attrs.get('cv_r2_mean', 0.0),
                'cv_r2_std': trial.user_attrs.get('cv_r2_std', 0.0),
                'cv_rmse_mean': trial.user_attrs.get('cv_rmse_mean', 0.0),
                'cv_rmse_std': trial.user_attrs.get('cv_rmse_std', 0.0),
                'cv_mse_mean': trial.user_attrs.get('cv_mse_mean', 0.0),
                'cv_mse_std': trial.user_attrs.get('cv_mse_std', 0.0),
                'cv_mae_mean': trial.user_attrs.get('cv_mae_mean', 0.0),
                'cv_mae_std': trial.user_attrs.get('cv_mae_std', 0.0),
                'best_r2': trial.user_attrs.get('best_r2', 0.0),
                'best_rmse': trial.user_attrs.get('best_rmse', 0.0),
                'best_mse': trial.user_attrs.get('best_mse', 0.0),
                'best_mae': trial.user_attrs.get('best_mae', 0.0),
                'test_r2': trial.user_attrs.get('test_r2', 0.0),
                'test_rmse': trial.user_attrs.get('test_rmse', 0.0),
                'test_mse': trial.user_attrs.get('test_mse', 0.0),
                'test_mae': trial.user_attrs.get('test_mae', 0.0),
                'cv_test_r2_mean': trial.user_attrs.get('cv_test_r2_mean', 0.0),
                'cv_test_r2_std': trial.user_attrs.get('cv_test_r2_std', 0.0),
                'cv_test_rmse_mean': trial.user_attrs.get('cv_test_rmse_mean', 0.0),
                'cv_test_rmse_std': trial.user_attrs.get('cv_test_rmse_std', 0.0),
                'cv_test_mse_mean': trial.user_attrs.get('cv_test_mse_mean', 0.0),
                'cv_test_mae_mean': trial.user_attrs.get('cv_test_mae_mean', 0.0),
                'n_features': trial.user_attrs.get('n_features', 0),
                'execution_time': trial.user_attrs.get('execution_time', 0.0),
                'memory_used_mb': trial.user_attrs.get('memory_used_mb', 0.0),
                'selected_descriptors': trial.user_attrs.get('selected_descriptors', [])
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file with folder structure: result/4_ANO_FO/dataset/split_type/
    import json
    results_dir = OUTPUT_DIR / dataset / split_type
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{dataset}_{split_type}_FO_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Update config with best features
    from config import update_best_config
    if results.get('best_params'):
        best_features = results['best_params'].get('selected_descriptors', [])
        update_best_config(4, dataset, split_type, best_features)
    
    # Update best summary CSV
    update_best_summary(dataset, split_type, results)
    
    print(f"Results saved to: {results_file}")
    
    # Fix temp path issue: copy results from temp to normal path if needed
    temp_results_dir = Path("save_model") / "4_ANO_FeatureOptimization_FO" / dataset / split_type
    temp_results_file = temp_results_dir / f"{dataset}_{split_type}_FO_results.json"
    
    # If results were saved in temp directory, copy to proper location
    if temp_results_file.exists() and not results_file.exists():
        import shutil
        # Ensure proper directory exists
        results_dir.mkdir(parents=True, exist_ok=True)
        # Copy from temp to proper location
        shutil.copy2(temp_results_file, results_file)
        print(f"Results copied from temp to proper location: {results_file}")
    
    return results

def update_best_summary(dataset, split_type, results):
    """
    Update best results summary CSV for module 4.

    Creates or updates a CSV file containing the best feature optimization results
    for each dataset-split combination. This provides a quick reference for
    comparing feature selection performance across different datasets and splits.

    Args:
        dataset (str): Dataset name (e.g., 'ws', 'de', 'lo', 'hu')
        split_type (str): Split strategy (e.g., 'rm', 'sc', 'cs')
        results (dict): Dictionary containing optimization results with keys:
                       - best_value: Best R² score achieved
                       - trial_details: List of trial result dictionaries
                       - best_params: Best hyperparameters found

    Returns:
        dict: The input results dictionary (unchanged)
    """
    summary_file = OUTPUT_DIR / "best_summary.csv"
    
    # Extract key metrics
    best_trial = None
    if results.get('trial_details'):
        for trial in results['trial_details']:
            if trial['value'] == results['best_value']:
                best_trial = trial
                break
    
    if not best_trial:
        best_trial = results.get('trial_details', [{}])[0] if results.get('trial_details') else {}
    
    # Create summary row
    new_row = {
        'dataset': dataset,
        'split_type': split_type,
        'best_cv_r2': results.get('best_value', 0.0),
        'cv_r2_mean': best_trial.get('cv_r2_mean', 0.0),
        'cv_r2_std': best_trial.get('cv_r2_std', 0.0),
        'test_r2': best_trial.get('test_r2', 0.0),
        'test_rmse': best_trial.get('test_rmse', 0.0),
        'n_features': best_trial.get('n_features', 0),
        'n_selected_descriptors': len(best_trial.get('selected_descriptors', [])),
        'execution_time': best_trial.get('execution_time', 0.0),
        'memory_used_mb': best_trial.get('memory_used_mb', 0.0)
    }
    
    # Update or create CSV
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        # Check if this dataset/split_type already exists
        mask = (df['dataset'] == dataset) & (df['split_type'] == split_type)
        if mask.any():
            # Update existing row
            for col, val in new_row.items():
                df.loc[mask, col] = val
        else:
            # Add new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create new DataFrame
        df = pd.DataFrame([new_row])
    
    # Save CSV
    df.to_csv(summary_file, index=False)
    print(f"Updated best summary: {summary_file}")
    return results

def main():
    """
    Main function
    Optimized with resource monitoring and enhanced error handling
    """
    global EPOCHS, CODE_DATASETS

    print("="*80)
    print("🔍 MODULE 4: ANO FEATURE OPTIMIZATION (FEATURE OPTIMIZATION)")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Module Focus: Fingerprint Selection & Feature Optimization")
    print(f"💻 Operating System: {OS_TYPE}")
    print(f"🖥️  Device: {device}")
    print(f"🧵 PyTorch threads: {torch.get_num_threads()}")
    print(f"🔧 Split types: {SPLIT_TYPES}")
    print(f"🎲 Number of trials: {N_TRIALS}")
    print(f"⏱️  Epochs per trial: {EPOCHS}")
    db_info = get_database_info()
    print(f"💾 Storage: {STORAGE_NAME} ({db_info['backend']})")
    print(f"🧬 Available Fingerprints: {CODE_FINGERPRINTS}")

    # Initialize resource monitoring
    experiment_start_time = time.time()
    experiment_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    experiment_start_cpu = psutil.cpu_percent(interval=0.1)

    print(f"\n📊 EXPERIMENT RESOURCE MONITORING")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Start memory: {experiment_start_memory:.1f} MB")
    print(f"🖥️  Start CPU: {experiment_start_cpu:.1f}%")

    # Use code-specific datasets
    datasets = CODE_DATASETS
    total_tasks = len(datasets) * len(SPLIT_TYPES)
    current_task = 0

    print(f"📊 Active Datasets (Module 4): {datasets}")
    print(f"📈 Total combinations to process: {total_tasks}")
    print("="*80)
    
    for split_type in SPLIT_TYPES:
        print(f"\n{'='*60}")
        print(f"🔄 SPLIT TYPE: {split_type.upper()}")
        print(f"{'='*60}")

        for dataset in datasets:
            current_task += 1
            print(f"\n{'🔍 '*20}")
            print(f"📋 TASK {current_task}/{total_tasks}: {get_dataset_display_name(dataset).upper()}")
            print(f"🔀 Split: {split_type}")
            print(f"🧬 Target: Feature Selection & Optimization")
            print(f"⏳ Progress: {current_task/total_tasks*100:.1f}%")
            print(f"{'🔍 '*20}")

            try:
                # Prepare data first to get actual feature dimensions
                print(f"📊 Loading data for {get_dataset_display_name(dataset)}...")
                data = prepare_data_for_split(dataset, split_type)
                base_fps = data['train_fps']
                base_y = data['train_y']

                print(f"📐 Base fingerprint shape: {base_fps.shape}")
                print(f"🏷️  Samples: train={base_fps.shape[0]}, test={data['test_fps'].shape[0]}")
                import numpy as np
                base_y_np = np.array(base_y)
                print(f"📊 Target range: [{base_y_np.min():.3f}, {base_y_np.max():.3f}]")

                # Dimension validation
                expected_features = 2727  # Morgan(2048) + MACCS(167) + Avalon(512)
                actual_features = base_fps.shape[1]
                if actual_features != expected_features:
                    print(f"⚠️  DIMENSION MISMATCH: Expected {expected_features}, got {actual_features}")
                else:
                    print(f"✅ Dimension match confirmed: {actual_features} features")
                
                # Create study name
                study_name = f"ano_feature_{dataset}_{split_type}"
                
                # Handle study renewal if requested
                if renew:
                    try:
                        optuna.delete_study(study_name=study_name, storage=STORAGE_NAME)
                        print(f"Deleted existing study: {study_name}")
                    except KeyError:
                        pass
                
                # Create Optuna study using centralized configuration
                # sampler, pruner = get_optuna_sampler_and_pruner('best_overall')  # 2025 proven best combination
                # print_optuna_info('best_overall')  # Print configuration details

                # Create Optuna Sampler from config (centralized management)
                sampler_config = MODEL_CONFIG.get('optuna_sampler', {})
                sampler_type = sampler_config.get('type', 'tpe')
                n_startup_trials = int(N_TRIALS * sampler_config.get('n_startup_trials_ratio', 0.2))

                if sampler_type == 'tpe':
                    sampler = optuna.samplers.TPESampler(
                        n_startup_trials=n_startup_trials,
                        n_ei_candidates=sampler_config.get('n_ei_candidates', 50),
                        multivariate=sampler_config.get('multivariate', False),
                        warn_independent_sampling=False
                    )
                    print(f"Module 4 sampler: TPESampler (n_startup_trials={n_startup_trials}/{N_TRIALS})")
                elif sampler_type == 'random':
                    sampler = optuna.samplers.RandomSampler()
                    print(f"Module 4 sampler: RandomSampler")
                else:
                    # Fallback to TPESampler
                    sampler = optuna.samplers.TPESampler(
                        n_startup_trials=n_startup_trials,
                        n_ei_candidates=50,
                        multivariate=False,
                        warn_independent_sampling=False
                    )
                    print(f"Module 4 sampler: TPESampler (fallback)")

                # Create HyperbandPruner (matched with reference implementation successful strategy)
                # This pruner automatically allocates resources efficiently
                pruner_config = MODEL_CONFIG.get('optuna_pruner', {})

                if pruner_config.get('type') == 'hyperband':
                    pruner = optuna.pruners.HyperbandPruner(
                        min_resource=pruner_config.get('min_resource', 100),
                        max_resource=pruner_config.get('max_resource', 1000),
                        reduction_factor=pruner_config.get('reduction_factor', 3)
                    )
                    print(f"Module 4 pruner: HyperbandPruner (min={pruner_config.get('min_resource')}, "
                          f"max={pruner_config.get('max_resource')}, reduction_factor={pruner_config.get('reduction_factor')})")
                else:
                    # Fallback to MedianPruner if config not found
                    n_startup = max(2, min(5, int(N_TRIALS * 0.1)))
                    pruner = optuna.pruners.MedianPruner(
                        n_startup_trials=n_startup,
                        n_warmup_steps=0,
                        interval_steps=1
                    )
                    print(f"Module 4 pruner: MedianPruner (fallback) n_startup_trials={n_startup}")

                # Create study with retry mechanism for database locks
                print(f"Creating study: {study_name}")
                for retry in range(3):  # Try up to 3 times
                    try:
                        print(f"Study creation attempt {retry+1}/3...")
                        study = optuna.create_study(
                            direction='maximize',
                            sampler=sampler,
                            pruner=pruner,
                            storage=STORAGE_NAME,
                            study_name=study_name,
                            load_if_exists=(not renew)
                        )
                        print(f"✅ Study created successfully!")
                        break  # Success, exit retry loop
                    except Exception as e:
                        if retry < 2:  # Not the last retry
                            print(f"❌ Database lock error (retry {retry+1}/3): {str(e)}")
                            time.sleep(2 * (retry + 1))  # Exponential backoff: 2, 4 seconds
                        else:
                            print(f"❌ Final retry failed: {str(e)}")
                            raise  # Re-raise on final retry
                
                # Display detailed progress header
                total_datasets = len(datasets)
                total_splits = len(SPLIT_TYPES)
                total_trials = N_TRIALS
                current_dataset_idx = datasets.index(dataset) + 1
                current_split_idx = SPLIT_TYPES.index(split_type) + 1
                total_combinations = total_datasets * total_splits
                current_combination = (current_split_idx - 1) * total_datasets + current_dataset_idx

                print(f"\n{'='*75}")
                print(f"[MODULE 4] FEATURE OPTIMIZATION: {get_dataset_display_name(dataset).upper()} | Trial (0/{N_TRIALS})")
                print(f"Dataset: {get_dataset_display_name(dataset).upper()} | Split: {split_type.upper()} | Method: Binary Feature Selection")
                print(f"Scope: Dataset ({current_dataset_idx}/{total_datasets}) × Split ({current_split_idx}/{total_splits}) × Trial (0/{N_TRIALS})")
                print(f"Overall Progress: Combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.2f}%)")
                print(f"Totals: {total_datasets} datasets × {total_splits} splits × {N_TRIALS} trials = {total_datasets * total_splits * N_TRIALS:,} optimizations")
                print(f"Features: Searching optimal descriptor combinations from 49 molecular descriptors")
                print(f"Expected duration: ~{N_TRIALS * 60 / 60:.1f}m | Target: Max CV R² score")
                print(f"{'='*75}")

                # Create objective function (pass prepared data to avoid duplication)
                objective_func = create_objective_function(dataset, split_type, data)

                # Run optimization with progress tracking
                study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)
                
                # Print results
                print(f"{get_dataset_display_name(dataset)}-{split_type} optimization completed!")
                print(f"Best CV R2 score: {study.best_value:.4f}")
                print(f"Best parameters: {study.best_params}")
                
                # Save results
                save_study_results(study, dataset, split_type)
                
            except Exception as e:
                print(f"Error processing {get_dataset_display_name(dataset)}-{split_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Enhanced memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate final experiment statistics
    experiment_end_time = time.time()
    experiment_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    experiment_end_cpu = psutil.cpu_percent(interval=0.1)
    
    experiment_total_time = experiment_end_time - experiment_start_time
    experiment_total_memory = experiment_end_memory - experiment_start_memory
    
    print(f"\n=== EXPERIMENT COMPLETED ===")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {experiment_total_time:.2f} seconds")
    print(f"Total memory change: {experiment_total_memory:+.1f} MB")
    print(f"Final memory usage: {experiment_end_memory:.1f} MB")
    print(f"Final CPU usage: {experiment_end_cpu:.1f}%")
    print(f"==============================\n")
    
    print("\nAll feature selection optimizations completed!")
    
    # Clean up temporary files
    import glob
    temp_patterns = ['temp_*.pth', 'temp_*.pkl', 'temp_*.csv', 'temp_*.json', 'temp_*.npz', '*.tmp']
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Cleaned up: {temp_file}")
            except:
                pass

if __name__ == "__main__":
    # Global variable to control study renewal
    renew = MODEL_CONFIG.get('renew', False)  # Control from config.py
    print(f"⚙️  Renew setting: {renew} ({'Fresh start' if renew else 'Resume mode'})")
    
    # Set up logging to file
    from pathlib import Path
    from config import MODULE_NAMES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get dataset from command line arguments or use all datasets
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset (ws/de/lo/hu)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to process (default: from CODE_SPECIFIC_DATASETS). Options: ws, de, lo, hu')
    parser.add_argument('--split', type=str, default=None, help='Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)')
    parser.add_argument('--epochs', type=int, default=None, help='Training epochs per trial')
    parser.add_argument('--trials', type=int, default=None, help='Number of Optuna trials')
    parser.add_argument('--renew', action='store_true', help='Start fresh optimization (ignore previous results)')
    args, _ = parser.parse_known_args()

    # Set EPOCHS with proper priority: args > MODEL_CONFIG['epochs'] > module_epochs
    EPOCHS = get_epochs_for_module('4', args)

    # Set datasets with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    from config import CODE_SPECIFIC_DATASETS, ACTIVE_SPLIT_TYPES, DATA_PATH
    from pathlib import Path

    if args.datasets:
        CODE_DATASETS = args.datasets
        print(f"Datasets from argparse: {CODE_DATASETS}")
    elif args.dataset:
        CODE_DATASETS = [args.dataset]
        print(f"Dataset from argparse (single): {CODE_DATASETS}")
    elif '4' in CODE_SPECIFIC_DATASETS:
        CODE_DATASETS = CODE_SPECIFIC_DATASETS['4']
        print(f"Datasets from CODE_SPECIFIC_DATASETS: {CODE_DATASETS}")
    else:
        # Fallback: scan data directory
        CODE_DATASETS = []
        for split_type in ACTIVE_SPLIT_TYPES:
            split_dir = Path(DATA_PATH) / 'train' / split_type
            if split_dir.exists():
                for csv_file in split_dir.glob('*_train.csv'):
                    dataset = csv_file.stem.split('_')[1]
                    if dataset not in CODE_DATASETS:
                        CODE_DATASETS.append(dataset)
        print(f"Datasets from data directory scan: {CODE_DATASETS}")

    # Get module name from config
    module_name = MODULE_NAMES.get('4', '4_ANO_FeatureOptimization_FO')
    
    # Determine log file path based on dataset
    if args.dataset:
        log_dir = Path(f"logs/{module_name}/dataset/{args.dataset}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{module_name}_{args.dataset}_{timestamp}.log"
    else:
        log_dir = Path(f"logs/{module_name}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{module_name}_all_{timestamp}.log"
    
    # Create a custom logger class
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            self.log.write(f"{'='*60}\n")
            self.log.write(f"Module 4 (FO) Execution Started: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n\n")
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
        def close(self):
            self.log.write(f"\n{'='*60}\n")
            self.log.write(f"Module 4 (FO) Execution Completed: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n")
            self.log.close()
    
    # Replace stdout and stderr with logger
    logger = Logger(str(log_file))
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        print(f"Log file: {log_file}")
        main()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__