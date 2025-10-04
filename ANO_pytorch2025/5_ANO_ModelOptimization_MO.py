#!/usr/bin/env python3
"""
ANO Model Optimization (MO) - ANO Framework Module 5
====================================================

PURPOSE:
Module 5 implements Model Optimization (MO), optimizing neural network architecture
while keeping features fixed to fingerprints only (no descriptors).

KEY FEATURES:
1. **Architecture Search**: Optimizes layers, hidden units, dropout, activation
2. **FlexibleDNNModel**: Dynamic architecture with 1-5 layers
3. **Fingerprints Only**: Uses Morgan+MACCS+Avalon (2727 features)
4. **No StandardScaler**: Binary fingerprints don't need normalization
5. **Optuna HPO**: Comprehensive hyperparameter optimization
6. **Dual CV Methods**: Type1 (Research) and Type2 (Production)

RECENT UPDATES (2024):
- Uses FlexibleDNNModel (not SimpleDNN) for dynamic architecture
- NO StandardScaler (fingerprints are already binary 0/1)
- Fixed get_epochs_for_module usage
- Default epochs: 30 (from config.py module_epochs['5'])
- Optuna trials: 1 (for quick testing, increase for better results)

OPTIMIZATION PARAMETERS:
- n_layers: 1-5 layers
- hidden_dims: 2-9999 units per layer
- activation: relu, gelu, silu, leaky_relu, elu, mish, selu
- dropout_rate: 0.1-0.5
- batch_norm: True/False
- learning_rate: 1e-4 to 1e-2
- batch_size: 32, 64, 128
- optimizer: Adam, AdamW, RAdam, NAdam
- weight_decay: 1e-5 to 1e-3

ARCHITECTURE STRATEGY:
1. Fixed input: 2727 fingerprint features
2. Variable depth: 1-5 hidden layers
3. Decreasing width: Each layer ≤ previous layer
4. Flexible activation functions
5. Optional batch normalization

OUTPUT STRUCTURE:
result/5_model_optimization/
├── {dataset}_{split}/
│   ├── best_architecture.json
│   ├── optimization_history.csv
│   ├── model_weights.pt
│   └── performance_metrics.json
└── architecture_comparison.png

USAGE:
python 5_ANO_ModelOptimization_MO.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --trials: Number of Optuna trials (default: 1)
  --epochs: Override epochs (default from config: 30)
  --renew: Start fresh optimization (ignore previous results)
"""

import os
import sys
from datetime import datetime
import time
import gc
import json
import subprocess

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")
import numpy as np
import pandas as pd
import optuna
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Module 5 uses Optuna - DO NOT fix seeds for optimization diversity

# Fingerprint size constants
MORGAN_FP_SIZE = 2048
MACCS_FP_SIZE = 167
AVALON_FP_SIZE = 512
TOTAL_FP_SIZE = MORGAN_FP_SIZE + MACCS_FP_SIZE + AVALON_FP_SIZE  # 2727

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
# Seeds are intentionally NOT set to allow Optuna to explore different initialization
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from rdkit import Chem

# Import from molecular_loader_with_fp and NPZ cache
from extra_code.mol_fps_maker import get_fingerprints, get_fingerprints_cached

# Import flexible DNN model
from extra_code.ano_feature_selection import FlexibleDNNModel

# Import configuration from config.py
from config import (
    DATASETS, SPLIT_TYPES as ALL_SPLIT_TYPES, ACTIVE_SPLIT_TYPES,
    FINGERPRINTS, MODEL_CONFIG,
    CHEMICAL_DESCRIPTORS, DESCRIPTORS_NEED_NORMALIZATION,
    DATA_PATH, RESULT_PATH, MODEL_PATH, PARALLEL_CONFIG,
    OPTUNA_CONFIG, get_optuna_sampler_and_pruner, print_optuna_info,
    get_dataset_display_name, get_dataset_filename, get_split_type_name,
    get_code_datasets, get_code_fingerprints,
    get_storage_url, get_database_info, get_epochs_for_module
)

# Configuration
# Get code-specific configurations - will be updated in main() with argparse priority
CODE_DATASETS = get_code_datasets(5)  # Code 5 - default, will be updated in main()
CODE_FINGERPRINTS = get_code_fingerprints(5)  # Code 5 - should be ['all']

# Enable all split types for comprehensive analysis (test mode uses only rm)
SPLIT_TYPES = ACTIVE_SPLIT_TYPES  # Use config.py setting (currently ['rm'])
N_TRIALS = MODEL_CONFIG['optuna_trials']  # Use config.py setting (currently 1)
EPOCHS = None  # Will be set in main() with proper priority
OUTPUT_DIR = Path(RESULT_PATH) / "5_ANO_ModelOptimization_MO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_NAME = get_storage_url('5')  # Module 5

# Dataset name mapping - use only code-specific datasets
DATASET_MAPPING = {k: DATASETS[k] for k in CODE_DATASETS}

def load_preprocessed_data(dataset_short, split_type):
    """
    Load preprocessed train/test data
    
    This function loads the pre-split train and test datasets that were created
    in the preprocessing step. The data is already split according to various
    splitting strategies (random, scaffold, etc.) to ensure fair evaluation.
    
    Args:
        dataset_short: Short dataset name ('ws', 'de', 'lo', 'hu')
        split_type: Split strategy ('rm', 'sc', 'cs', etc.)
    
    Returns:
        Tuple of (train_smiles, train_targets, test_smiles, test_targets)
    """
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

def prepare_data_for_split(dataset_short, split_type):
    """
    Prepare data for a specific dataset and split
    
    This function converts SMILES strings to molecular fingerprints for both
    train and test sets. It combines three types of fingerprints (Morgan, MACCS,
    Avalon) to create a comprehensive molecular representation.
    
    Processing steps:
    1. Load preprocessed SMILES and target values
    2. Convert SMILES to RDKit molecule objects
    3. Filter out invalid molecules
    4. Generate three types of fingerprints
    5. Concatenate all fingerprints into single feature vector
    
    Args:
        dataset_short: Short dataset name
        split_type: Data splitting strategy
    
    Returns:
        Dictionary with train/test fingerprints and targets
    """
    try:
        # Load preprocessed data
        train_smiles, train_y, test_smiles, test_y = load_preprocessed_data(dataset_short, split_type)
        
        print(f"  Loaded {dataset_short.upper()}-{split_type}: {len(train_smiles)} train, {len(test_smiles)} test")
        
        # Convert SMILES to molecules
        train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
        test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        
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
        
        # Calculate fingerprints with NPZ caching (all: morgan + maccs + avalon)
        train_morgan, train_maccs, train_avalon = get_fingerprints_cached(
            train_mols_filtered,
            dataset_short,
            split_type,
            'train',
            module_name='5_ANO_ModelOptimization_MO'
        )
        train_fps = np.hstack([train_morgan, train_maccs, train_avalon])
        
        test_morgan, test_maccs, test_avalon = get_fingerprints_cached(
            test_mols_filtered,
            dataset_short,
            split_type,
            'test',
            module_name='5_ANO_ModelOptimization_MO'
        )
        test_fps = np.hstack([test_morgan, test_maccs, test_avalon])
        
        print(f"  Train fingerprint shape: {train_fps.shape}")
        print(f"  Test fingerprint shape: {test_fps.shape}")
        
        return {
            'train_fps': train_fps,
            'train_y': train_y_filtered,
            'test_fps': test_fps,
            'test_y': test_y_filtered
        }
        
    except Exception as e:
        print(f"Error preparing data for {dataset_short}-{split_type}: {e}")
        raise

def clean_data(X, y):
    """
    Clean data by removing NaN/Inf values from features and targets.

    This function ensures data quality by filtering out samples with invalid values
    (NaN or Inf) in either features or targets. This prevents numerical issues
    during model training and evaluation.

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

def train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds=5, trial=None):
    """
    Type1: Research Pipeline

    Performs K-fold CV on training data only and predicts independent test set per fold.
    This approach prevents data leakage by keeping the test set completely separate during
    cross-validation, providing unbiased performance estimates.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Independent test features
        y_test (np.ndarray): Independent test targets
        model_params (dict): Model hyperparameters (batch_size, epochs, learning_rate)
        hidden_dims (list): Hidden layer dimensions
        dropout_rate (float): Dropout probability
        n_folds (int): Number of CV folds. Defaults to 5.
        trial (optuna.Trial, optional): Optuna trial for pruning. Defaults to None.

    Returns:
        dict: CV statistics including mean/std for R², RMSE, MSE, MAE
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds} on Train + Test prediction per fold")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    test_r2_scores = []  # Test R² scores from each fold
    test_rmse_scores = []  # Test RMSE from each fold
    test_mae_scores = []  # Test MAE from each fold
    test_mse_scores = []  # Test MSE from each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Save temporary files for subprocess
        Path("save_model").mkdir(exist_ok=True)
        np.save(f"save_model/X_train_{fold}.npy", X_train_fold)
        np.save(f"save_model/y_train_{fold}.npy", y_train_fold)
        np.save(f"save_model/X_test_{fold}.npy", X_test)  # Type1: predict independent test set each fold
        np.save(f"save_model/y_test_{fold}.npy", y_test)

        # Save architecture for subprocess to read
        with open("save_model/temp_architecture.json", "w") as f:
            json.dump({"hidden_dims": hidden_dims, "dropout_rate": dropout_rate}, f)

        # Run training subprocess (standard 8 arguments)
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            f"save_model/X_train_{fold}.npy", f"save_model/y_train_{fold}.npy",
            f"save_model/X_test_{fold}.npy", f"save_model/y_test_{fold}.npy",
            "save_model/full_model_type1.pt"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Parse results
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if ',' in line and line.count(',') == 3:
                        parts = line.split(',')
                        if len(parts) == 4:
                            try:
                                test_r2 = float(parts[0])
                                test_rmse = float(parts[1])
                                test_mse = float(parts[2])  # MSE from subprocess
                                test_mae = float(parts[3])

                                # Sanitize metrics to avoid NaN/Inf
                                test_r2 = 0.0 if np.isnan(test_r2) or np.isinf(test_r2) else test_r2
                                test_rmse = float('inf') if np.isnan(test_rmse) or np.isinf(test_rmse) else test_rmse
                                test_mse = float('inf') if np.isnan(test_mse) or np.isinf(test_mse) else test_mse
                                test_mae = 0.0 if np.isnan(test_mae) or np.isinf(test_mae) else test_mae

                                test_r2_scores.append(test_r2)
                                test_rmse_scores.append(test_rmse)
                                test_mse_scores.append(test_mse)
                                test_mae_scores.append(test_mae)

                                # Report intermediate value for pruning
                                if trial is not None:
                                    # Aggressive pruning based on fold number
                                    if fold == 0:
                                        # First fold: Kill if R² < -3
                                        if test_r2 < -3.0:
                                            print(f"🔴 Trial {trial.number} KILLED (Fold 1 R2: {test_r2:.4f} < -3.0)")
                                            import optuna
                                            raise optuna.exceptions.TrialPruned()
                                    else:
                                        # Fold 2+: Kill if R² ≤ 0
                                        if test_r2 <= 0.0:
                                            print(f"🔴 Trial {trial.number} KILLED (Fold {fold+1} R2: {test_r2:.4f} <= 0.0)")
                                            import optuna
                                            raise optuna.exceptions.TrialPruned()

                                    # Report to Optuna for MedianPruner
                                    trial.report(test_r2, fold)

                                    # Let MedianPruner decide after enough trials
                                    if trial.should_prune():
                                        print(f"Trial {trial.number} PRUNED by MedianPruner (R2: {test_r2:.4f})")
                                        import optuna
                                        raise optuna.exceptions.TrialPruned()

                                break
                            except ValueError:
                                continue
            else:
                print(f"      Fold {fold+1} subprocess failed with return code {result.returncode}")
                if result.stderr:
                    print(f"      Error: {result.stderr[:500]}")  # Print first 500 chars of error
                test_r2_scores.append(0.0)
                test_rmse_scores.append(float('inf'))
                test_mse_scores.append(float('inf'))
                test_mae_scores.append(float('inf'))

        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned immediately
            raise
        except Exception as e:
            print(f"      Fold {fold+1} error: {e}")
            test_r2_scores.append(0.0)
            test_rmse_scores.append(float('inf'))
            test_mse_scores.append(float('inf'))
            test_mae_scores.append(float('inf'))

        # Clean up temporary files
        for temp_file in [f"save_model/X_train_{fold}.npy", f"save_model/y_train_{fold}.npy",
                          f"save_model/X_test_{fold}.npy", f"save_model/y_test_{fold}.npy"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Calculate CV statistics (research results - average of test predictions across folds)
    cv_stats = {
        'r2_mean': np.mean(test_r2_scores),
        'r2_std': np.std(test_r2_scores),
        'rmse_mean': np.mean(test_rmse_scores),
        'rmse_std': np.std(test_rmse_scores),
        'mse_mean': np.mean(test_mse_scores),
        'mse_std': np.std(test_mse_scores),
        'mae_mean': np.mean(test_mae_scores),
        'mae_std': np.std(test_mae_scores),
        'test_scores': test_r2_scores,
        'fold_scores': test_r2_scores,  # Add fold_scores for compatibility
        'fold_rmse_scores': test_rmse_scores,
        'fold_mae_scores': test_mae_scores
    }

    return cv_stats

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds=5, trial=None):
    """
    Type2: Production Pipeline

    Splits training data into train/val for CV, then makes final prediction on independent test set.
    This approach is closer to production deployment where you train on all available data and
    evaluate on a separate test set only once.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Independent test features
        y_test (np.ndarray): Independent test targets
        model_params (dict): Model hyperparameters (batch_size, epochs, learning_rate)
        hidden_dims (list): Hidden layer dimensions
        dropout_rate (float): Dropout probability
        n_folds (int): Number of CV folds. Defaults to 5.
        trial (optuna.Trial, optional): Optuna trial for pruning. Defaults to None.

    Returns:
        tuple: (cv_stats, final_metrics) where cv_stats contains validation metrics
               and final_metrics contains test set performance
    """
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_val_r2_scores = []  # Validation R² scores for each fold
    cv_val_rmse_scores = []  # Validation RMSE for each fold
    cv_val_mse_scores = []  # Validation MSE for each fold
    cv_val_mae_scores = []  # Validation MAE for each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Save temporary files for subprocess
        Path("save_model").mkdir(exist_ok=True)
        np.save(f"save_model/X_train_{fold}.npy", X_tr)
        np.save(f"save_model/y_train_{fold}.npy", y_tr)
        np.save(f"save_model/X_val_{fold}.npy", X_val)  # Type2: validate within CV
        np.save(f"save_model/y_val_{fold}.npy", y_val)

        # Save architecture for subprocess to read
        with open("save_model/temp_architecture.json", "w") as f:
            json.dump({"hidden_dims": hidden_dims, "dropout_rate": dropout_rate}, f)

        # Run training subprocess (standard 8 arguments)
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            f"save_model/X_train_{fold}.npy", f"save_model/y_train_{fold}.npy",
            f"save_model/X_val_{fold}.npy", f"save_model/y_val_{fold}.npy",
            "save_model/full_model_type2.pt"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

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
                                val_mse = float(parts[2])
                                val_mae = float(parts[3])
                                cv_val_r2_scores.append(val_r2)
                                cv_val_rmse_scores.append(val_rmse)
                                cv_val_mse_scores.append(val_mse)
                                cv_val_mae_scores.append(val_mae)
                                break
                            except ValueError:
                                continue
            else:
                print(f"      Fold {fold+1} subprocess failed")
                cv_val_r2_scores.append(0.0)
                cv_val_rmse_scores.append(float('inf'))
                cv_val_mse_scores.append(float('inf'))
                cv_val_mae_scores.append(float('inf'))

        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned immediately
            raise
        except Exception as e:
            print(f"      Fold {fold+1} error: {e}")
            cv_val_r2_scores.append(0.0)
            cv_val_rmse_scores.append(float('inf'))
            cv_val_mse_scores.append(float('inf'))
            cv_val_mae_scores.append(float('inf'))

        # Clean up temporary files
        for temp_file in [f"save_model/X_train_{fold}.npy", f"save_model/y_train_{fold}.npy",
                          f"save_model/X_val_{fold}.npy", f"save_model/y_val_{fold}.npy"]:
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

    # Save architecture for subprocess to read
    with open("save_model/temp_architecture.json", "w") as f:
        json.dump({"hidden_dims": hidden_dims, "dropout_rate": dropout_rate}, f)

    # Run final training subprocess (standard 8 arguments)
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
        result = subprocess.run(cmd, capture_output=True, text=True)

        final_r2 = final_rmse = final_mse = final_mae = 0.0
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
                            final_mse = float(parts[2])
                            final_mae = float(parts[3])
                            break
                        except ValueError:
                            continue

    except Exception as e:
        print(f"      Final training error: {e}")
        final_r2 = final_rmse = final_mse = final_mae = 0.0

    # Clean up temporary files
    for temp_file in ["save_model/X_train_full.npy", "save_model/y_train_full.npy",
                      "save_model/X_test_final.npy", "save_model/y_test_final.npy"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Calculate CV statistics (mean ± std)
    cv_stats = {
        'mean_r2': np.mean(cv_val_r2_scores),
        'std_r2': np.std(cv_val_r2_scores),
        'mean_rmse': np.mean(cv_val_rmse_scores),
        'std_rmse': np.std(cv_val_rmse_scores),
        'mean_mse': np.mean(cv_val_mse_scores),
        'std_mse': np.std(cv_val_mse_scores),
        'mean_mae': np.mean(cv_val_mae_scores),
        'std_mae': np.std(cv_val_mae_scores)
    }

    final_metrics = {
        'test_r2': final_r2,
        'test_rmse': final_rmse,
        'test_mse': final_mse,
        'test_mae': final_mae
    }

    return cv_stats, final_metrics

def train_model_cv_both_types(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds=5, trial=None):
    """
    Dual CV approach preventing data leakage

    Runs both Type1 (Research) and Type2 (Production) pipelines to provide
    comprehensive model evaluation from both perspectives.

    - Type1: K-fold CV on training data only + test prediction per fold
    - Type2: Train/val split for CV + single independent test prediction

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        model_params (dict): Model hyperparameters
        hidden_dims (list): Hidden layer dimensions
        dropout_rate (float): Dropout probability
        n_folds (int): Number of CV folds
        trial (optuna.Trial, optional): Optuna trial

    Returns:
        dict: Results from both Type1 and Type2 pipelines
    """
    print(f"\n=== Running Both CV Types ===")

    results = {}

    # Type1: Research Pipeline - K-fold CV on train only + test prediction per fold
    print(f"    [TYPE1-Research] Research Pipeline - CV-5")
    type1_results = train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds, trial)
    results['type1'] = type1_results

    # Type2: Production Pipeline - Train/val split + single independent test prediction
    print(f"    [TYPE2-Production] Production Pipeline - CV-5 on Train + Final Test")
    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds, trial)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_and_evaluate_cv(X_train_full, y_train_full, X_test, y_test, model_params, hidden_dims, dropout_rate, activation='relu', optimizer_name='adam', trial=None):
    """
    Train with CV-5 on train set and final evaluation on test set.

    This function implements the core training and evaluation logic using 5-fold
    cross-validation on the training set, followed by final evaluation on the
    hold-out test set. It uses a subprocess to handle the actual training to
    ensure clean memory management.

    Training process:
    1. Split training data into 5 folds
    2. For each fold:
       - Train on 4 folds, validate on 1 fold
       - Save temporary data files
       - Call learning subprocess
       - Parse and store metrics
    3. Calculate mean and std of CV metrics
    4. Train final model on full training set
    5. Evaluate on hold-out test set

    Args:
        X_train_full (np.ndarray): Full training features (n_samples x n_features)
        y_train_full (np.ndarray): Full training targets (n_samples,)
        X_test (np.ndarray): Test features (n_test_samples x n_features)
        y_test (np.ndarray): Test targets (n_test_samples,)
        model_params (dict): Dictionary with batch_size, epochs, learning_rate, weight_decay
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout probability (0.0 to 1.0)
        activation (str, optional): Activation function name. Defaults to 'relu'.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        trial (optuna.Trial, optional): Optuna trial for pruning. Defaults to None.

    Returns:
        tuple: 16 metrics as (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std,
               best_rmse, cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std,
               best_mae, test_r2, test_rmse, test_mse, test_mae)
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

    # Run both CV types with corrected methodology (no data leakage)
    cv_results = train_model_cv_both_types(X_train_full, y_train_full, X_test, y_test, model_params, hidden_dims, dropout_rate, n_folds=5, trial=trial)

    # Extract Type1 (Research) results for primary metrics (Optuna optimization)
    type1_results = cv_results['type1']
    cv_r2_mean = type1_results['r2_mean']  # Use Type1 for Optuna optimization
    cv_r2_std = type1_results['r2_std']
    cv_rmse_mean = type1_results['rmse_mean']
    cv_rmse_std = type1_results['rmse_std']
    cv_mse_mean = type1_results['mse_mean']
    cv_mse_std = type1_results['mse_std']
    cv_mae_mean = type1_results['mae_mean']
    cv_mae_std = type1_results['mae_std']

    # Extract Type2 (Production) results
    type2_results = cv_results['type2']
    type2_cv_stats = type2_results['cv_stats']
    type2_final_metrics = type2_results['final_metrics']

    # Legacy variables for compatibility
    cv_r2_scores = type1_results['fold_scores']
    cv_rmse_scores = [cv_rmse_mean] * 5  # Placeholder for compatibility
    cv_mse_scores = [cv_rmse_mean**2] * 5  # Placeholder for compatibility
    cv_mae_scores = [cv_mae_mean] * 5  # Placeholder for compatibility

    print(f"    [TYPE1-Research] CV Val: R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f}, MAE={cv_mae_mean:.4f}±{cv_mae_std:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={cv_r2_mean:.4f}±{cv_r2_std:.4f}, RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f}, MAE={cv_mae_mean:.4f}±{cv_mae_std:.4f}")
    print(f"    [TYPE2-Production] CV Results: R²={type2_cv_stats.get('mean_r2', 0):.4f}±{type2_cv_stats.get('std_r2', 0):.4f}, RMSE={type2_cv_stats.get('mean_rmse', 0):.4f}±{type2_cv_stats.get('std_rmse', 0):.4f}, MSE={type2_cv_stats.get('mean_mse', 0):.4f}±{type2_cv_stats.get('std_mse', 0):.4f}, MAE={type2_cv_stats.get('mean_mae', 0):.4f}±{type2_cv_stats.get('std_mae', 0):.4f}")
    print(f"    [TYPE2-Production] Final Test: R²={type2_final_metrics.get('test_r2', 0):.4f}, RMSE={type2_final_metrics.get('test_rmse', 0):.4f}, MSE={type2_final_metrics.get('test_mse', 0):.4f}, MAE={type2_final_metrics.get('test_mae', 0):.4f}")
    print(f"  📌 [OPTUNA] Using Type1 R²={cv_r2_mean:.4f} for optimization")

    # Calculate final statistics from dual CV results
    best_r2 = max(cv_r2_scores) if cv_r2_scores else cv_r2_mean
    best_rmse = min(cv_rmse_scores) if cv_rmse_scores else cv_rmse_mean
    best_mse = best_rmse ** 2
    best_mae = cv_mae_mean  # Use mean for best MAE
    cv_mse_mean = cv_rmse_mean ** 2
    cv_mse_std = type1_results['mse_std']  # Use actual MSE std from TYPE1

    # Use Type2 final test results for test metrics
    test_r2 = type2_final_metrics['test_r2']
    test_rmse = type2_final_metrics['test_rmse']
    test_mse = type2_final_metrics['test_mse']  # Use actual MSE from subprocess
    test_mae = type2_final_metrics['test_mae']

    """
    OLD CV LOOP REPLACED WITH DUAL CV APPROACH
    The old CV loop has been replaced with train_model_cv_both_types() function
    which implements both Type1 (Research) and Type2 (Production) approaches.
    Type1 is used for Optuna optimization, Type2 provides production metrics.
    """


    return (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
            cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
            test_r2, test_rmse, test_mse, test_mae)

# Using FlexibleDNNModel from ano_feature_selection module

def create_objective_function(dataset_name, split_type):
    """
    Create objective function for given dataset and split
    
    This factory function creates a closure that serves as the Optuna objective
    function for hyperparameter optimization. The objective function suggests
    hyperparameters, trains a model, and returns the CV performance.
    
    Hyperparameters optimized:
    1. n_layers: Number of hidden layers (2-4)
    2. hidden_dim_i: Dimension of each hidden layer
    3. dropout_rate: Dropout probability (0.1-0.5)
    4. learning_rate: Learning rate (1e-4 to 1e-2, log scale)
    5. batch_size: Training batch size (16, 32, or 64)
    
    The function ensures architectural constraints:
    - First layer can be large (256, 512, 1024)
    - Subsequent layers must be <= previous layer size
    - This creates a funnel architecture that progressively
      compresses information toward the output
    
    Args:
        dataset_name: Dataset identifier ('ws', 'de', 'lo', 'hu')
        split_type: Data splitting strategy
    
    Returns:
        Objective function for Optuna optimization
    """
    def objective_function(trial):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Prepare data for this specific split
            data = prepare_data_for_split(dataset_name, split_type)

            train_fps = data['train_fps']
            train_y = data['train_y']
            test_fps = data['test_fps']
            test_y = data['test_y']
            
            print(f"\n🎯 {dataset_name.upper()} Trial {trial.number}: Starting structure optimization...")
            print(f"📊 Dataset: {get_dataset_display_name(dataset_name)}")
            print(f"🔀 Split Type: {split_type}")
            print(f"🧬 Fingerprints: {CODE_FINGERPRINTS}")
            print(f"📐 Train shape: {train_fps.shape}")
            print(f"📐 Test shape: {test_fps.shape}")
            print(f"🎲 Trial: {trial.number + 1}/{N_TRIALS}")

            # Dimension validation (assume standard combined fingerprint size)
            actual_features = train_fps.shape[1]
            if actual_features != TOTAL_FP_SIZE:
                print(f"⚠️  DIMENSION MISMATCH: Expected {TOTAL_FP_SIZE}, got {actual_features}")
            else:
                print(f"✅ Dimension match confirmed: {actual_features} features")

            print(f"🏷️  Train samples: {train_fps.shape[0]}, Test samples: {test_fps.shape[0]}")
            train_y_np = np.array(train_y)
            print(f"📊 Target range: [{train_y_np.min():.3f}, {train_y_np.max():.3f}]")
            
            # Get hyperparameter ranges from config
            from config import DNN_HYPERPARAMETERS
            hp_config = DNN_HYPERPARAMETERS['search_space']
            
            # Suggest hyperparameters for structure optimization
            # Key innovation: Dynamic layer count (1-5 layers)
            n_layers = trial.suggest_int('n_layers',
                                        hp_config['n_layers'][0],
                                        hp_config['n_layers'][1])
            hidden_dims = []
            
            # Suggest activation function from config
            activation = trial.suggest_categorical('activation', hp_config['activation'])
            
            # Suggest hidden dimensions for each layer with decreasing pattern
            for i in range(n_layers):
                if i == 0:
                    # First layer can use full range
                    hidden_dim = trial.suggest_int(f'hidden_dim_{i}',
                                                 hp_config['hidden_dims'][0],  # min: 2
                                                 hp_config['hidden_dims'][1])  # max: 9999
                else:
                    # Subsequent layers must be smaller than previous layer
                    prev_dim = hidden_dims[i-1]
                    if prev_dim <= hp_config['hidden_dims'][0]:
                        # If previous layer is already minimum, use minimum
                        hidden_dim = hp_config['hidden_dims'][0]
                    else:
                        # Choose from min to previous layer size
                        hidden_dim = trial.suggest_int(f'hidden_dim_{i}',
                                                     hp_config['hidden_dims'][0],  # min: 2
                                                     prev_dim)  # max: previous layer
                hidden_dims.append(hidden_dim)
            
            # Add final output layer with 1 unit for regression
            hidden_dims.append(1)
            
            # Regularization parameters from config
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            weight_decay = trial.suggest_float('weight_decay',
                                              hp_config['weight_decay'][0],
                                              hp_config['weight_decay'][1], log=True)  # L2 regularization
            
            # Training parameters from config
            batch_size = trial.suggest_categorical('batch_size', hp_config['batch_size'])
            
            # Optimizer parameters from config
            optimizer_name = trial.suggest_categorical('optimizer', hp_config['optimizer'])
            learning_rate = trial.suggest_categorical('learning_rate', hp_config['learning_rate'])
            
            # Optimizer-specific parameters
            if optimizer_name == 'sgd':
                momentum = trial.suggest_float('momentum', 0.0, 0.99)
                nesterov = trial.suggest_categorical('nesterov', [True, False])
            elif optimizer_name in ['adam', 'adamw']:
                beta1 = trial.suggest_float('beta1', 0.5, 0.999)
                beta2 = trial.suggest_float('beta2', 0.9, 0.999)
                eps = trial.suggest_float('eps', 1e-10, 1e-6, log=True)
            
            # Additional training parameters
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            
            # Learning rate scheduler
            scheduler_name = trial.suggest_categorical('scheduler', ['none', 'step', 'exponential', 'cosine', 'reduce_on_plateau'])
            if scheduler_name == 'step':
                step_size = trial.suggest_int('step_size', 10, 50)
                gamma = trial.suggest_float('gamma', 0.1, 0.9)
            elif scheduler_name == 'exponential':
                gamma = trial.suggest_float('gamma', 0.9, 0.99)
            
            # Model architecture is now passed to subprocess via pickle file
            # No need to create model here since subprocess will handle it
            
            # Model parameters
            model_params = {
                'batch_size': batch_size,
                'epochs': EPOCHS,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            }
            
            # Train with CV-5 and evaluate on test
            (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
             cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
             test_r2, test_rmse, test_mse, test_mae) = train_and_evaluate_cv(
                train_fps, train_y, test_fps, test_y, model_params, hidden_dims, dropout_rate, activation, optimizer_name, trial
            )
            
            # Calculate execution time and memory usage
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Store all metrics in trial attributes (sanitize values)
            import math
            def sanitize_value(val):
                if val is None or math.isnan(val) or math.isinf(val):
                    return 0.0
                return float(val)

            trial.set_user_attr('cv_r2_mean', sanitize_value(cv_r2_mean))
            trial.set_user_attr('cv_r2_std', sanitize_value(cv_r2_std))
            trial.set_user_attr('cv_rmse_mean', sanitize_value(cv_rmse_mean))
            trial.set_user_attr('cv_rmse_std', sanitize_value(cv_rmse_std))
            trial.set_user_attr('cv_mse_mean', sanitize_value(cv_mse_mean))
            trial.set_user_attr('cv_mse_std', sanitize_value(cv_mse_std))
            trial.set_user_attr('cv_mae_mean', sanitize_value(cv_mae_mean))
            trial.set_user_attr('cv_mae_std', sanitize_value(cv_mae_std))
            trial.set_user_attr('best_r2', sanitize_value(best_r2))
            trial.set_user_attr('best_rmse', sanitize_value(best_rmse))
            trial.set_user_attr('best_mse', sanitize_value(best_mse))
            trial.set_user_attr('best_mae', sanitize_value(best_mae))
            trial.set_user_attr('test_r2', sanitize_value(test_r2))
            trial.set_user_attr('test_rmse', sanitize_value(test_rmse))
            trial.set_user_attr('test_mse', sanitize_value(test_mse))
            trial.set_user_attr('test_mae', sanitize_value(test_mae))
            trial.set_user_attr('n_layers', int(n_layers))
            trial.set_user_attr('hidden_dims', str(hidden_dims))
            trial.set_user_attr('dropout_rate', sanitize_value(dropout_rate))
            trial.set_user_attr('learning_rate', sanitize_value(learning_rate))
            trial.set_user_attr('batch_size', int(batch_size))
            trial.set_user_attr('activation', str(activation))
            trial.set_user_attr('weight_decay', sanitize_value(weight_decay))
            trial.set_user_attr('n_features', int(train_fps.shape[1]))
            trial.set_user_attr('execution_time', sanitize_value(execution_time))
            trial.set_user_attr('memory_used_mb', sanitize_value(memory_used))
            trial.set_user_attr('dataset', str(dataset_name))
            trial.set_user_attr('split_type', str(split_type))
            
            print(f"  Trial completed: Type1 CV R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}")
            print(f"                 Type2 Test R2={test_r2:.4f}")
            print(f"                 Time: {execution_time:.2f}s")
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return cv_r2_mean  # Optimize based on CV performance

        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned to ensure proper pruning (don't treat as error)
            raise
        except Exception as e:
            print(f"Error in {dataset_name.upper()} trial: {e}")
            # Memory cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
            
    return objective_function

def save_study_results(study, dataset, split_type):
    """
    Save study results to file
    
    This function extracts and saves all optimization results including:
    - Best hyperparameters found
    - Optimization history
    - Detailed metrics for each trial
    - Resource usage (time and memory)
    
    The results are saved in JSON format for easy analysis and visualization.
    
    Args:
        study: Optuna study object
        dataset: Dataset name
        split_type: Split strategy
    
    Returns:
        Dictionary containing all results
    """
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
                'cv_method2_test_r2': trial.user_attrs.get('cv_method2_test_r2', 0.0),
                'cv_method2_test_rmse': trial.user_attrs.get('cv_method2_test_rmse', 0.0),
                'n_layers': trial.user_attrs.get('n_layers', 0),
                'hidden_dims': trial.user_attrs.get('hidden_dims', '[]'),
                'dropout_rate': trial.user_attrs.get('dropout_rate', 0.0),
                'learning_rate': trial.user_attrs.get('learning_rate', 0.0),
                'batch_size': trial.user_attrs.get('batch_size', 0),
                'execution_time': trial.user_attrs.get('execution_time', 0.0),
                'memory_used_mb': trial.user_attrs.get('memory_used_mb', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file with folder structure: result/5_ANO_MO/dataset/split_type/
    results_dir = OUTPUT_DIR / dataset / split_type
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{dataset}_{split_type}_MO_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Update config with best structure
    from config import update_best_config
    if results.get('best_params'):
        best_structure = {
            'n_layers': results['best_params'].get('n_layers'),
            'hidden_dims': results['best_params'].get('hidden_dims', []),
            'dropout_rate': results['best_params'].get('dropout_rate'),
            'activation': results['best_params'].get('activation', 'relu'),
            'learning_rate': results['best_params'].get('learning_rate'),
            'batch_size': results['best_params'].get('batch_size')
        }
        update_best_config(5, dataset, split_type, best_structure)
    
    # Update best summary CSV
    update_best_summary(dataset, split_type, results)
    
    print(f"Results saved to: {results_file}")
    
    # Fix temp path issue: copy results from temp to normal path if needed
    temp_results_dir = Path("save_model") / "5_ANO_ModelOptimization_MO" / dataset / split_type
    temp_results_file = temp_results_dir / f"{dataset}_{split_type}_MO_results.json"
    
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
    Update best results summary CSV for module 5.

    Creates or updates a CSV file containing the best optimization results
    for each dataset-split combination. This provides a quick reference
    for comparing performance across different datasets and splits.

    Args:
        dataset (str): Dataset name (e.g., 'ws', 'de', 'lo', 'hu')
        split_type (str): Split strategy (e.g., 'rm', 'sc', 'cs')
        results (dict): Dictionary containing optimization results with keys:
                       - best_value: Best R² score achieved
                       - trial_details: List of trial result dictionaries

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
        'n_layers': best_trial.get('n_layers', 0),
        'hidden_dims': str(best_trial.get('hidden_dims', [])),
        'dropout_rate': best_trial.get('dropout_rate', 0.0),
        'learning_rate': best_trial.get('learning_rate', 0.0),
        'batch_size': best_trial.get('batch_size', 0),
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
    Main function for ANO structure optimization

    This is the entry point for the structure optimization module. It:
    1. Iterates through all datasets and split types
    2. Creates Optuna studies for hyperparameter optimization
    3. Runs trials to find optimal network architectures
    4. Saves results for later analysis

    The optimization focuses on finding the best neural network structure
    while keeping the molecular features fixed (all fingerprints).
    """

    # Parse command line arguments with priority over config
    import argparse
    parser = argparse.ArgumentParser(description='ANO Module 5: Model Architecture Optimization')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to process (ws, de, lo, hu). Default: all datasets from config')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets to process (default: from CODE_SPECIFIC_DATASETS). Options: ws, de, lo, hu')
    parser.add_argument('--split', type=str, default=None,
                       help='Specific split type (rm, ac, cl, etc.). Default: all splits from config')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of optimization trials. Default: from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs per trial. Default: from config')
    parser.add_argument('--fingerprints', type=str, nargs='+', default=None,
                       help='Fingerprint types to use. Default: from config')
    parser.add_argument('--renew', action='store_true',
                       help='Delete existing studies and start fresh')

    args = parser.parse_args()

    # Apply argparse overrides (argparse has highest priority)
    global N_TRIALS, EPOCHS, SPLIT_TYPES, CODE_DATASETS, CODE_FINGERPRINTS, renew

    # Override trials if specified
    if args.trials is not None:
        N_TRIALS = args.trials

    # Override epochs if specified
    # Set EPOCHS with proper priority: args > MODEL_CONFIG['epochs'] > module_epochs
    EPOCHS = get_epochs_for_module('5', args)

    # Set datasets with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    from config import CODE_SPECIFIC_DATASETS, ACTIVE_SPLIT_TYPES, DATA_PATH
    from pathlib import Path

    if args.datasets:
        CODE_DATASETS = args.datasets
        print(f"Datasets from argparse: {CODE_DATASETS}")
    elif args.dataset is not None:
        if args.dataset not in ['ws', 'de', 'lo', 'hu']:
            raise ValueError(f"Invalid dataset: {args.dataset}. Must be one of: ws, de, lo, hu")
        CODE_DATASETS = [args.dataset]
        print(f"Dataset from argparse (single): {CODE_DATASETS}")
    elif '5' in CODE_SPECIFIC_DATASETS:
        CODE_DATASETS = CODE_SPECIFIC_DATASETS['5']
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

    # Override split types if specified
    if args.split is not None:
        SPLIT_TYPES = [args.split]

    # Override fingerprints if specified
    if args.fingerprints is not None:
        CODE_FINGERPRINTS = args.fingerprints

    # Override renew setting
    if args.renew:
        renew = True

    print("="*80)
    print("🚀 MODULE 5: ANO STRUCTURE OPTIMIZATION (MODEL OPTIMIZATION)")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Module Focus: Neural Network Architecture Optimization")
    print(f"🔧 Split types: {SPLIT_TYPES}")
    print(f"🎲 Number of trials: {N_TRIALS}")
    print(f"⏱️  Epochs per trial: {EPOCHS}")
    print(f"💾 Storage: {STORAGE_NAME}")
    print(f"🧬 Fingerprints: {CODE_FINGERPRINTS}")
    if args.dataset:
        print(f"📊 Target Dataset: {args.dataset.upper()} (argparse override)")
    if args.split:
        print(f"🎯 Target Split: {args.split.upper()} (argparse override)")

    # Use code-specific datasets (potentially overridden by argparse)
    datasets = CODE_DATASETS
    total_tasks = len(datasets) * len(SPLIT_TYPES)
    current_task = 0

    print(f"📊 Active Datasets (Module 5): {datasets}")
    print(f"📈 Total combinations to process: {total_tasks}")
    print("="*80)

    for split_type in SPLIT_TYPES:
        print(f"\n{'='*60}")
        print(f"🔄 SPLIT TYPE: {split_type.upper()}")
        print(f"{'='*60}")

        for dataset in datasets:
            current_task += 1
            print(f"\n{'🏗️ '*20}")
            print(f"📋 TASK {current_task}/{total_tasks}: {get_dataset_display_name(dataset).upper()}")
            print(f"🔀 Split: {split_type}")
            print(f"🧬 Fingerprints: {CODE_FINGERPRINTS}")
            print(f"⏳ Progress: {current_task/total_tasks*100:.1f}%")
            print(f"{'🏗️ '*20}")
            
            try:
                # Create study name
                study_name = f"ano_structure_{dataset}_{split_type}"
                
                # Handle study renewal if requested
                if renew:
                    try:
                        optuna.delete_study(study_name=study_name, storage=STORAGE_NAME)
                        print(f"Deleted existing study: {study_name}")
                    except KeyError:
                        pass
                
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
                    print(f"Module 5 sampler: TPESampler (n_startup_trials={n_startup_trials}/{N_TRIALS})")
                elif sampler_type == 'random':
                    sampler = optuna.samplers.RandomSampler()
                    print(f"Module 5 sampler: RandomSampler")
                else:
                    # Fallback to TPESampler
                    sampler = optuna.samplers.TPESampler(
                        n_startup_trials=n_startup_trials,
                        n_ei_candidates=50,
                        multivariate=False,
                        warn_independent_sampling=False
                    )
                    print(f"Module 5 sampler: TPESampler (fallback)")

                # Create HyperbandPruner for Module 5 (matched with Module 4 strategy)
                pruner_config = MODEL_CONFIG.get('optuna_pruner', {})
                if pruner_config.get('type') == 'hyperband':
                    pruner = optuna.pruners.HyperbandPruner(
                        min_resource=pruner_config.get('min_resource', 100),
                        max_resource=pruner_config.get('max_resource', 1000),
                        reduction_factor=pruner_config.get('reduction_factor', 3)
                    )
                    print(f"Module 5 pruner: HyperbandPruner (min_resource={pruner_config.get('min_resource', 100)}, "
                          f"max_resource={pruner_config.get('max_resource', 1000)}, "
                          f"reduction_factor={pruner_config.get('reduction_factor', 3)})")
                else:
                    # Fallback to MedianPruner
                    n_startup = max(3, min(10, int(N_TRIALS * 0.2)))
                    pruner = optuna.pruners.MedianPruner(
                        n_startup_trials=n_startup,
                        n_warmup_steps=0,
                        interval_steps=1
                    )
                    print(f"Module 5 pruner: MedianPruner (n_startup_trials={n_startup})")

                # Create study with retry mechanism for database locks
                for retry in range(3):  # Try up to 3 times
                    try:
                        study = optuna.create_study(
                            direction='maximize',
                            sampler=sampler,
                            pruner=pruner,
                            storage=STORAGE_NAME,
                            study_name=study_name,
                            load_if_exists=(not renew)
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        if retry < 2:  # Not the last retry
                            print(f"Database lock error (retry {retry+1}/3): {str(e)}")
                            time.sleep(2 * (retry + 1))  # Exponential backoff: 2, 4 seconds
                        else:
                            raise  # Re-raise on final retry
                
                # Display detailed progress header for Module 5
                total_datasets = len(datasets)
                total_splits = len(SPLIT_TYPES)
                current_dataset_idx = datasets.index(dataset) + 1
                current_split_idx = SPLIT_TYPES.index(split_type) + 1
                total_combinations = total_datasets * total_splits
                current_combination = (current_split_idx - 1) * total_datasets + current_dataset_idx

                print(f"\n{'='*75}")
                print(f"[MODULE 5] STRUCTURE OPTIMIZATION: {get_dataset_display_name(dataset).upper()} | Trial (0/{N_TRIALS})")
                print(f"Dataset: {get_dataset_display_name(dataset).upper()} | Split: {split_type.upper()} | Method: Neural Architecture Search")
                print(f"Scope: Dataset ({current_dataset_idx}/{total_datasets}) × Split ({current_split_idx}/{total_splits}) × Trial (0/{N_TRIALS})")
                print(f"Overall Progress: Combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.2f}%)")
                print(f"Totals: {total_datasets} datasets × {total_splits} splits × {N_TRIALS} trials = {total_datasets * total_splits * N_TRIALS:,} optimizations")
                print(f"Features: Fixed fingerprints (Morgan+MACCS+Avalon) with dynamic architecture (2-9999 units)")
                print(f"Expected duration: ~{N_TRIALS * 90 / 60:.1f}m | Target: Max CV R² with optimal DNN structure")
                print(f"{'='*75}")

                # Create objective function
                objective_func = create_objective_function(dataset, split_type)

                # Run optimization
                study.optimize(
                    objective_func,
                    n_trials=N_TRIALS,
                    # timeout=1800,  # Removed timeout limit
                    show_progress_bar=True
                )
                
                # Print results
                print(f"{get_dataset_display_name(dataset)}-{split_type} optimization completed!")
                print(f"Best CV R2 score: {study.best_value:.4f}")
                print(f"Best parameters: {study.best_params}")
                
                # Save results
                save_study_results(study, dataset, split_type)
                
            except Exception as e:
                print(f"Error processing {get_dataset_display_name(dataset)}-{split_type}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
    
    print("\nAll structure optimizations completed!")
    
    # Clean up temporary files
    import glob
    temp_patterns = ['save_model/temp_*.pth', 'temp_*.pkl', 'temp_*.csv', 'temp_*.json', 'temp_*.npz', '*.tmp', 'save_model/temp_model_cv*.pt']
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
    # renew = True
    
    # Set up logging to file
    from pathlib import Path
    from config import MODULE_NAMES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Note: argparse is now handled in main() function
    
    # Get module name from config
    module_name = MODULE_NAMES.get('5', '5_ANO_ModelOptimization_MO')
    
    # Determine log file path (args will be parsed in main)
    log_dir = Path(f"logs/{module_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module_name}_all_{timestamp}.log"
    
    # Create a custom logger class
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            self.log.write(f"{'='*60}\n")
            self.log.write(f"Module 5 Execution Started: {datetime.now()}\n")
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
            self.log.write(f"Module 5 Execution Completed: {datetime.now()}\n")
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