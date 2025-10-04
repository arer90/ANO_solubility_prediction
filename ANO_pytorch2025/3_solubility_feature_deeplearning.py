"""
Feature-based Deep Learning Analysis - ANO Framework Module 3
=============================================================

PURPOSE:
This module analyzes the impact of molecular descriptors on deep learning performance.
It combines fingerprints with chemical descriptors to evaluate feature importance.

KEY FEATURES:
1. **Feature Combination**: Fingerprints + 49 molecular descriptors
2. **Deep Learning Focus**: SimpleDNN with standardized architecture
3. **Dual CV Methods**: Type1 (Research) and Type2 (Production) validation
4. **3D Descriptor Support**: Handles 3D conformer-based descriptors
5. **Feature Importance**: Analyzes descriptor contribution to predictions
6. **No StandardScaler**: Raw features used directly

RECENT UPDATES (2024):
- Fixed import: Now uses SimpleDNN from ano_feature_selection
- Removed StandardScaler - using raw features directly
- Fixed get_epochs_for_module usage
- Default epochs: 100 (from config.py module_epochs['3'])
- Improved plot generation with proper data visualization

FEATURE TYPES:
- Fingerprints: Morgan (2048), MACCS (167), Avalon (512)
- 2D Descriptors: MolWeight, MolLogP, TPSA, NumRotatableBonds, etc.
- 3D Descriptors: Asphericity, RadiusOfGyration, PMI, NPR, etc.
- Total: ~49 molecular descriptors + fingerprint features

OUTPUT STRUCTURE:
result/3_feature_deeplearning/
├── plots/
│   ├── {dataset}_{split}_{fingerprint}_with_baseline.png
│   ├── {dataset}_{split}_{fingerprint}_without_baseline.png
│   └── descriptor_performance.png
├── models/
│   └── {dataset}_{split}_{fingerprint}_model.pt
└── consolidated_results.csv

USAGE:
python 3_solubility_feature_deeplearning.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --fingerprint: Specific fingerprint (morgan/maccs/avalon/all)
  --epochs: Override epochs (default from config: 100)
"""

import os
import sys
import numpy as np
import pandas as pd
# Support both subprocess and direct SimpleDNN training methods
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gc
import time
import json
import psutil
import pickle
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler  # Not used anymore
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import argparse

# Import SimpleDNN from centralized location
from extra_code.ano_feature_selection import SimpleDNN
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")

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

# Import data loading functions from mol_fps_maker and chem_descriptor_maker
from extra_code.mol_fps_maker import (
    load_split_data,
    get_fingerprints_cached,
    get_fingerprints_combined
)
# ChemDescriptorCalculator removed - use pre-generated descriptors from chem_descriptor_maker.py

# Import SimpleDNN from centralized location (like scripts 4-9)
from extra_code.ano_feature_selection import SimpleDNN

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import training method configuration
from config import MODEL_CONFIG
TRAINING_METHOD = MODEL_CONFIG.get('training_method', 'subprocess')  # Default to subprocess

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Import configuration
from config import (
    DATASETS, SPLIT_TYPES, ACTIVE_SPLIT_TYPES,
    FINGERPRINTS, MODEL_CONFIG,
    CHEMICAL_DESCRIPTORS, DESCRIPTORS_NEED_NORMALIZATION,
    DATA_PATH, RESULT_PATH, MODEL_PATH, PARALLEL_CONFIG, RESTART_CONFIG,
    DATASET_DISPLAY_NAMES, get_dataset_display_name, get_dataset_filename,
    get_split_type_name, get_code_datasets, get_code_fingerprints,
    get_epochs_for_module, format_descriptor_name_for_display
)

# Add extra imports for 3D descriptors
# from rdkit.Chem import Descriptors3D  # Removed - not needed
from pathlib import Path

# Helper functions to match the expected API
def extract_xy_from_data(data, feature_names=None):
    """Extract X (features) and y (target) from data DataFrame"""
    if feature_names:
        X = data[feature_names].values
    else:
        # Assume last column is target
        X = data.iloc[:, :-1].values
    y = data['y'].values if 'y' in data.columns else data.iloc[:, -1].values
    return X, y

def build_fingerprints_for_splits(all_splits, dataset_name, fingerprint_names):
    """Build fingerprints for all splits of a dataset"""
    fps_dict = {}
    for split_name, split_data in all_splits[dataset_name].items():
        smiles_list = split_data['smiles'].tolist()
        fps_combined = get_fingerprints_combined(
            smiles_list=smiles_list,
            fingerprint_names=fingerprint_names,
            dataset_name=dataset_name,
            split_name=split_name,
            module_number=3
        )
        fps_dict[split_name] = fps_combined
    return fps_dict

def get_fingerprints_for_dataset(dataset_name, fingerprint_names):
    """Get fingerprints for a specific dataset"""
    all_splits = load_split_data()
    if dataset_name not in all_splits:
        raise ValueError(f"Dataset {dataset_name} not found")
    return build_fingerprints_for_splits(all_splits, dataset_name, fingerprint_names)


# Configuration - Use config.py settings properly
DATASET_MAP = {v: k for k, v in DATASETS.items()}  # Reverse mapping

# Get configurations from config.py
# Default datasets - will be overridden by argparse or CODE_SPECIFIC_DATASETS in main()
CODE_DATASETS = ['ws', 'de', 'lo', 'hu']  # Default - will be updated in main()
CODE_FINGERPRINTS = get_code_fingerprints(3)  # Get all 7 fingerprints for Code 3

# DNN baseline configuration
SKIP_DNN_BASELINE = False  # Set to True to skip DNN baseline calculation
PARALLEL_BASELINE = True   # Try parallel processing for baseline calculation

# 3. Use ACTIVE_SPLIT_TYPES from config.py
SPLIT_TYPES_TO_USE = ACTIVE_SPLIT_TYPES  # Use config value

# Convert to full names for processing
ACTIVE_DATASET_NAMES = [DATASETS[d] for d in CODE_DATASETS]
ACTIVE_SPLIT_NAMES = [SPLIT_TYPES[s] for s in SPLIT_TYPES_TO_USE]

ACTIVE_FP_NAMES = CODE_FINGERPRINTS

CV = MODEL_CONFIG['cv_folds']
# EPOCHS will be set in main() using get_epochs_for_module
EPOCHS = 30  # Default value, will be overridden in main()
BATCHSIZE = MODEL_CONFIG['batch_size']
LR = MODEL_CONFIG['learning_rate']
TARGET_PATH = os.path.join(RESULT_PATH, '3_solubility_feature_deeplearning')
os.makedirs(TARGET_PATH, exist_ok=True)

# Removed generate_fingerprint and generate_single_fingerprint functions
# Module 3 now uses precomputed fingerprints from extra_code.mol_fps_maker
# Following Module 4's pattern

def load_data(dataset, split_type):
    """Load train and test data from split files"""
    # split_type is already the folder name (e.g., 'rm')
    split_folder = split_type
        
    # Map dataset abbreviations to actual filenames
    dataset_file_map = {
        'ws': 'ws496_logS',
        'de': 'delaney-processed', 
        'lo': 'Lovric2020_logS0',
        'hu': 'huusk'
    }
    
    dataset_filename = dataset_file_map.get(dataset, dataset)
    train_path = os.path.join(DATA_PATH, 'train', split_folder, f'{split_folder}_{dataset_filename}_train.csv')
    test_path = os.path.join(DATA_PATH, 'test', split_folder, f'{split_folder}_{dataset_filename}_test.csv')
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Files not found: {train_path} or {test_path}")
        return None
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract smiles and labels
    train_smiles = train_df['smiles'].values
    train_y = train_df['target'].values
    test_smiles = test_df['smiles'].values
    test_y = test_df['target'].values
    
    return {
        'train_smiles': train_smiles,
        'train_y': train_y,
        'test_smiles': test_smiles,
        'test_y': test_y
    }

def train_model_direct(X_train, y_train, X_val, y_val, batch_size=32, epochs=100, lr=0.001, architecture=None):
    """
    Train model directly without subprocess (like scripts 4-9)

    Result: Returns model performance metrics (R², RMSE, MAE) after direct training
    This method provides faster execution and easier debugging compared to subprocess method

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training (default 32)
        # epochs: Number of training epochs (default 100) - commented out for result analysis
        lr: Learning rate for optimizer (default 0.001)
        architecture: [input_dim, hidden_dims..., output_dim]

    Returns:
        tuple: (r2_score, rmse, mae) - Performance metrics on validation set
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Create model
    input_dim = X_train.shape[1]
    if architecture is None:
        architecture = [input_dim, 1024, 496, 1]

    # Extract hidden dims from architecture (exclude input and output dims)
    hidden_dims = architecture[1:-1] if len(architecture) > 2 else [1024, 496]

    model = SimpleDNN(input_dim=input_dim, hidden_dims=hidden_dims, use_batch_norm=True, l2_reg=1e-4).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # Early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    patience = 75  # Increased patience for more stable training
    patience_counter = 0

    # Training loop with early stopping
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        mse_loss = criterion(outputs, y_train_tensor)

        # Add L2 regularization loss
        l2_loss = model.get_l2_loss()
        total_loss = mse_loss + l2_loss

        total_loss.backward()
        optimizer.step()

        # Validation every epoch for early stopping
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

            scheduler.step(val_loss)
        model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor).cpu().numpy()
        r2 = r2_score(y_val, val_outputs)
        rmse = np.sqrt(mean_squared_error(y_val, val_outputs))
        mae = mean_absolute_error(y_val, val_outputs)

    return r2, rmse, mae

def train_model_subprocess(X_train, y_train, X_val, y_val, batch_size=32, epochs=100, lr=0.001, architecture=None):
    """
    Train model using subprocess (original method)

    Result: Returns model performance metrics after subprocess training with TorchScript
    This method provides memory isolation and model persistence for complex training scenarios

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training (default 32)
        # epochs: Number of training epochs (default 100) - commented out for result analysis
        lr: Learning rate for optimizer (default 0.001)
        architecture: [input_dim, hidden_dims..., output_dim]

    Returns:
        tuple: (r2_score, rmse, mae) - Performance metrics parsed from subprocess output
    """

    # Prepare training data for subprocess
    train_data = {
        'X': X_train,
        'y': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'batch_size': batch_size,
        # 'epochs': epochs,  # Training epochs parameter - commented out for result analysis
        'epochs': epochs,  # Result: Number of training iterations passed to subprocess
        'lr': lr,
        'architecture': architecture if architecture is not None else [X_train.shape[1], 1024, 496, 1]  # Dynamic architecture
    }

    # Ensure save_model directory exists
    os.makedirs('save_model', exist_ok=True)

    # Use consistent filenames without timestamp
    train_file = 'save_model/module3_train_data.pkl'
    model_file = os.path.join('save_model', 'full_model.pt')

    temp_files = [train_file, model_file]  # Track for cleanup

    try:
        # Save training data
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)

        # Run subprocess training with memory isolation
        cmd = [sys.executable, os.path.join('extra_code', 'learning_process_pytorch_torchscript.py'),
               train_file, model_file]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())  # Removed timeout limit

        # Parse R² from output
        r2_value = 0.0
        rmse_value = 0.0
        mae_value = 0.0

        if result.returncode == 0:
            # Only show key subprocess information - simplified output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Training with' in line or 'Early stopping' in line or 'Final Results' in line:
                    print(f"    {line}")

            # Parse final results
            if "Final Results - R²:" in result.stdout:
                try:
                    r2_line = [line for line in result.stdout.split('\n') if 'Final Results - R²:' in line][0]
                    parts = r2_line.split('R²:')[1].split(',')
                    r2_value = float(parts[0].strip())
                    if len(parts) > 1:
                        rmse_value = float(parts[1].split('RMSE:')[1].strip())
                    if len(parts) > 2:
                        mae_value = float(parts[2].split('MAE:')[1].strip())

                    # Sanitize metrics to avoid NaN/Inf
                    r2_value = 0.0 if np.isnan(r2_value) or np.isinf(r2_value) else r2_value
                    rmse_value = float('inf') if np.isnan(rmse_value) or np.isinf(rmse_value) else rmse_value
                    mae_value = 0.0 if np.isnan(mae_value) or np.isinf(mae_value) else mae_value
                    print(f"  → Parsed: R²={r2_value:.4f}, RMSE={rmse_value:.4f}, MAE={mae_value:.4f}")
                except Exception as e:
                    print(f"  Error parsing subprocess output: {e}")
                    r2_value = 0.0
            else:
                print("  No 'Final Results' found in subprocess output")
        else:
            print(f"  Subprocess failed (returncode={result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr}")  # Show full error
            if result.stdout:
                print(f"  Output: {result.stdout}")  # Show full output

    except subprocess.TimeoutExpired:
        print("Subprocess timeout (600s)")
        r2_value = rmse_value = mae_value = 0.0
    except Exception as e:
        print(f"Subprocess error: {e}")
        r2_value = rmse_value = mae_value = 0.0
    finally:
        # Comprehensive cleanup
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return r2_value, rmse_value, mae_value

def train_model_adaptive(X_train, y_train, X_val, y_val, batch_size=32, epochs=100, lr=0.001, architecture=None):
    """
    Adaptive training function that uses the method specified in config.py
    Enhanced with improved memory management

    Result: Returns performance metrics using either direct or subprocess training method
    Method selection based on TRAINING_METHOD configuration for flexibility

    Args:
        # epochs: Number of training epochs (default 100) - commented out for result analysis

    Returns:
        tuple: (r2, rmse, mae) - Performance metrics from selected training method
    """
    # Pre-training memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Explicit garbage collection
    gc.collect()

    # Ensure clean model initialization
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        if TRAINING_METHOD == 'direct':
            print(f"  Using direct SimpleDNN training method")
            return train_model_direct(X_train, y_train, X_val, y_val, batch_size, epochs, lr, architecture)
        else:  # 'subprocess' or any other value defaults to subprocess
            print(f"  Using subprocess training method (TorchScript + memory isolation)")
            return train_model_subprocess(X_train, y_train, X_val, y_val, batch_size, epochs, lr, architecture)
    except Exception as e:
        print(f"Training failed: {e}, trying direct method...")
        # Fallback to direct method if subprocess fails
        return train_model_direct(X_train, y_train, X_val, y_val, batch_size, epochs, lr, architecture)
    finally:
        # Post-training cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def Normalization(descriptor):
    """Normalization function from notebook"""
    descriptor = np.asarray(descriptor)
    epsilon = 1e-10
    max_value = 1e15
    descriptor = np.clip(descriptor, -max_value, max_value)
    descriptor_custom = np.where(np.abs(descriptor) < epsilon, epsilon, descriptor)
    descriptor_log = np.sign(descriptor_custom) * np.log1p(np.abs(descriptor_custom))
    descriptor_log = np.nan_to_num(descriptor_log, nan=0.0, posinf=0.0, neginf=0.0)
    del epsilon
    gc.collect()
    return descriptor_log

def evaluate_model_cv_type1(X_data, y_data, n_folds=5):
    """
    Type 1: Research Pipeline - CV methodology

    Args:
        X_data: Feature matrix
        y_data: Target values
        n_folds: Number of CV folds

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds}")

    # Ensure arrays are numpy arrays
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []  # R² scores for each fold
    fold_rmse_scores = []  # RMSE for each fold
    fold_mae_scores = []  # MAE for each fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # Skip StandardScaler - use raw features
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X_train
        X_test_scaled = X_test

        # Train model using configured method
        input_dim = X_train_scaled.shape[1]
        architecture = [input_dim, 1024, 496, 1]
        r2_value, rmse, mae = train_model_adaptive(
            X_train_scaled, y_train, X_test_scaled, y_test,
            batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
        )

        fold_scores.append(r2_value)
        fold_rmse_scores.append(rmse)
        fold_mae_scores.append(mae)

    # Calculate CV statistics (research results)
    cv_stats = {
        'r2_mean': np.mean(fold_scores),
        'r2_std': np.std(fold_scores),
        'rmse_mean': np.mean(fold_rmse_scores),
        'rmse_std': np.std(fold_rmse_scores),
        'mae_mean': np.mean(fold_mae_scores),
        'mae_std': np.std(fold_mae_scores),
        'fold_scores': fold_scores
    }

    return cv_stats

def train_model_cv_type2(X_train, y_train, X_test, y_test, n_folds=5):
    """Type 2: Production Pipeline (Production) - Train/Test Split + CV"""
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_val_scores = []  # Validation R² scores for each fold
    cv_val_rmse_scores = []  # Validation RMSE for each fold
    cv_val_mae_scores = []  # Validation MAE for each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Skip StandardScaler - use raw features
        # scaler = StandardScaler()
        # X_tr_scaled = scaler.fit_transform(X_tr)
        # X_val_scaled = scaler.transform(X_val)
        X_tr_scaled = X_tr
        X_val_scaled = X_val

        # Train model on this fold and validate (NOT test)
        input_dim = X_tr_scaled.shape[1]
        architecture = [input_dim, 1024, 496, 1]
        val_r2, val_rmse, val_mae = train_model_adaptive(
            X_tr_scaled, y_tr, X_val_scaled, y_val,
            batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
        )

        cv_val_scores.append(val_r2)
        cv_val_rmse_scores.append(val_rmse)
        cv_val_mae_scores.append(val_mae)

    # After CV: Train on full training data and test on test set
    print(f"\n  Training final model on full training data...")

    # Skip StandardScaler - use raw features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Final model training
    input_dim = X_train_scaled.shape[1]
    architecture = [input_dim, 1024, 496, 1]
    final_r2, final_rmse, final_mae = train_model_adaptive(
        X_train_scaled, y_train, X_test_scaled, y_test,
        batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
    )

    # Calculate CV statistics (mean ± std)
    cv_stats = {
        'val_r2_mean': np.mean(cv_val_scores),
        'val_r2_std': np.std(cv_val_scores),
        'val_rmse_mean': np.mean(cv_val_rmse_scores),
        'val_rmse_std': np.std(cv_val_rmse_scores),
        'val_mae_mean': np.mean(cv_val_mae_scores),
        'val_mae_std': np.std(cv_val_mae_scores)
    }

    final_metrics = {
        'r2': final_r2,
        'rmse': final_rmse,
        'mae': final_mae
    }

    return cv_stats, final_metrics

def evaluate_model_cv_both_types(X_data, y_data, test_size=0.2, n_folds=5):
    """Execute both CV types and return results for both"""
    print(f"\n=== Running Both CV Types ===")

    results = {}

    # Type 1: Research Pipeline (Research) - CV-K
    print(f"\n--- Type 1: Research Pipeline ---")
    type1_results = evaluate_model_cv_type1(X_data, y_data, n_folds)
    results['type1'] = type1_results

    # Type 2: Production Pipeline (Production) - Train/Test Split + CV
    print(f"\n--- Type 2: Production Pipeline ---")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42, stratify=None
    )

    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, n_folds)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def calculate_dnn_baseline(fingerprints, y_true, test_fingerprints=None, test_y=None):
    """Calculate DNN baseline using ONLY fingerprints (no descriptors)"""
    baseline_start = time.time()
    print("\n" + "="*60)
    print("  🔍 DNN BASELINE EVALUATION (Fingerprint Only)")
    print("="*60)
    
    # Dual CV evaluation for baseline - Both Type1 and Type2
    baseline_cv_results = evaluate_model_cv_both_types(fingerprints, y_true, test_size=0.2, n_folds=CV)

    # Extract Type1 (Research) results for baseline metrics
    baseline_type1_results = baseline_cv_results['type1']
    baseline_cv_mean = baseline_type1_results['r2_mean']  # Use Type1 for primary baseline
    baseline_cv_std = baseline_type1_results['r2_std']

    # Extract Type2 (Production) results for baseline
    baseline_type2_results = baseline_cv_results['type2']
    baseline_type2_cv_stats = baseline_type2_results['cv_stats']
    baseline_type2_final_metrics = baseline_type2_results['final_metrics']

    print(f"    [TYPE1-Research] CV Val: R²={baseline_cv_mean:.4f}±{baseline_cv_std:.4f}, RMSE={baseline_type1_results['rmse_mean']:.4f}±{baseline_type1_results['rmse_std']:.4f}, MAE={baseline_type1_results['mae_mean']:.4f}±{baseline_type1_results['mae_std']:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={baseline_cv_mean:.4f}±{baseline_cv_std:.4f}, RMSE={baseline_type1_results['rmse_mean']:.4f}±{baseline_type1_results['rmse_std']:.4f}, MAE={baseline_type1_results['mae_mean']:.4f}±{baseline_type1_results['mae_std']:.4f}")
    print(f"    [TYPE2-Production] CV: R²={baseline_type2_cv_stats['val_r2_mean']:.4f}±{baseline_type2_cv_stats['val_r2_std']:.4f}, RMSE={baseline_type2_cv_stats['val_rmse_mean']:.4f}±{baseline_type2_cv_stats['val_rmse_std']:.4f}, MAE={baseline_type2_cv_stats['val_mae_mean']:.4f}±{baseline_type2_cv_stats['val_mae_std']:.4f}")
    print(f"    [TYPE2-Production] Test: R²={baseline_type2_final_metrics['r2']:.4f}, RMSE={baseline_type2_final_metrics['rmse']:.4f}, MAE={baseline_type2_final_metrics['mae']:.4f}")
    
    # Test evaluation if test set provided
    baseline_test_r2 = 0.0
    if test_fingerprints is not None and test_y is not None:
        # Skip StandardScaler - use raw features
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(fingerprints)
        # X_test_scaled = scaler.transform(test_fingerprints)
        X_train_scaled = fingerprints
        X_test_scaled = test_fingerprints

        # Train final model using configured method (subprocess by default)
        # Architecture dynamically adapts to input dimension
        input_dim = X_train_scaled.shape[1]
        architecture = [input_dim, 1024, 496, 1]
        baseline_test_r2, baseline_test_rmse, baseline_test_mae = train_model_adaptive(
            X_train_scaled, y_true, X_test_scaled, test_y,
            batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
        )
    
    baseline_time = time.time() - baseline_start

    # Final baseline results
    print("\n" + "🏆 BASELINE FINAL RESULTS")
    print("="*60)
    if test_fingerprints is not None and test_y is not None:
        test_mse = baseline_test_rmse ** 2
        print(f"  📈 CV-5 Performance:  R² = {baseline_cv_mean:.4f} ± {baseline_cv_std:.4f}")
        print(f"  🎯 Test Performance:  R² = {baseline_test_r2:.4f}, RMSE = {baseline_test_rmse:.4f}")
        print(f"                        MAE = {baseline_test_mae:.4f}, MSE = {test_mse:.4f}")
    else:
        print(f"  📈 CV-5 Performance:  R² = {baseline_cv_mean:.4f} ± {baseline_cv_std:.4f}")
    print("="*60)

    return baseline_cv_mean, baseline_cv_std, baseline_test_r2, baseline_time

def calculate_dnn_baseline_worker(args):
    """Worker function for parallel DNN baseline calculation"""
    json_file, dataset, split_type, fingerprint, worker_id = args
    
    print(f"[Worker {worker_id}] Processing {dataset}-{split_type}-{fingerprint}...")
    start_time = time.time()
    
    try:
        # Load data
        test_file = os.path.join(DATA_PATH, 'test', split_type, f'{split_type}_{dataset}_test.csv')
        train_file = os.path.join(DATA_PATH, 'train', split_type, f'{split_type}_{dataset}_train.csv')
        
        if not os.path.exists(test_file) or not os.path.exists(train_file):
            print(f"[Worker {worker_id}] Data files not found")
            return None
            
        test_df = pd.read_csv(test_file)
        train_df = pd.read_csv(train_file)
        
        # Get precomputed fingerprints
        train_mols = [Chem.MolFromSmiles(s) for s in train_df['smiles']]
        test_mols = [Chem.MolFromSmiles(s) for s in test_df['smiles']]
        
        # Filter out None molecules
        train_valid = [(mol, idx) for idx, mol in enumerate(train_mols) if mol is not None]
        test_valid = [(mol, idx) for idx, mol in enumerate(test_mols) if mol is not None]
        
        train_mols_valid = [m for m, _ in train_valid]
        test_mols_valid = [m for m, _ in test_valid]
        
        # Get precomputed fingerprints using get_fingerprints_combined
        fp_type = 'all' if fingerprint == 'morgan+maccs+avalon' else fingerprint
        
        train_fps_valid = get_fingerprints_combined(
            train_mols_valid,
            dataset,  # Dataset name (mol_fps_maker converts to lowercase internally)
            split_type,
            'train',
            fingerprint_type=fp_type,
            module_name='3_solubility_feature_deeplearning'
        )
        
        test_fps_valid = get_fingerprints_combined(
            test_mols_valid,
            dataset,  # Dataset name (mol_fps_maker converts to lowercase internally)
            split_type,
            'test',
            fingerprint_type=fp_type,
            module_name='3_solubility_feature_deeplearning'
        )
        
        # Create full arrays with zeros for invalid molecules
        fp_dim = train_fps_valid.shape[1] if len(train_fps_valid) > 0 else 2048
        train_fps = np.zeros((len(train_mols), fp_dim))
        test_fps = np.zeros((len(test_mols), fp_dim))
        
        for i, (_, idx) in enumerate(train_valid):
            train_fps[idx] = train_fps_valid[i]
        
        for i, (_, idx) in enumerate(test_valid):
            test_fps[idx] = test_fps_valid[i]
        
        X_train = np.array(train_fps)
        X_test = np.array(test_fps)
        y_train = train_df['target'].values
        y_test = test_df['target'].values
        
        # CV-5 evaluation with DNN
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Skip StandardScaler - use raw features
            # scaler = StandardScaler()
            # X_tr_scaled = scaler.fit_transform(X_tr)
            # X_val_scaled = scaler.transform(X_val)
            X_tr_scaled = X_tr
            X_val_scaled = X_val
            
            # Train model using configured method
            # Architecture dynamically adapts to input dimension
            input_dim = X_tr_scaled.shape[1]
            architecture = [input_dim, 1024, 496, 1]
            r2_value, rmse, mae = train_model_adaptive(
                X_tr_scaled, y_tr, X_val_scaled, y_val,
                batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
            )
            cv_scores.append(r2_value)
        
        cv_r2 = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Train final model
        # Skip StandardScaler - use raw features
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X_train
        X_test_scaled = X_test

        # Train final model using configured method
        # Architecture dynamically adapts to input dimension
        input_dim = X_train_scaled.shape[1]
        architecture = [input_dim, 1024, 496, 1]
        test_r2, test_rmse, test_mae = train_model_adaptive(
            X_train_scaled, y_train, X_test_scaled, y_test,
            batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
        )
        
        # No cleanup needed for direct training method
        
        elapsed = time.time() - start_time
        print(f"[Worker {worker_id}] Completed in {elapsed:.1f}s: CV={cv_r2:.3f}, Test={test_r2:.3f}")
        
        return {
            'json_file': json_file,
            'dataset': dataset,
            'split_type': split_type,
            'fingerprint': fingerprint,
            'dnn_baseline': {
                'cv_r2': cv_r2,
                'cv_std': cv_std,
                'test_r2': test_r2,
                'time': elapsed
            }
        }
        
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        return None

def detect_parallel_capability():
    """Detect if parallel processing is available on current OS"""
    import platform
    system = platform.system()
    
    print(f"\nDetecting parallel processing capability...")
    print(f"Operating System: {system}")
    print(f"CPU cores: {mp.cpu_count()}")
    
    if system == "Windows":
        print("Windows detected - will attempt parallel but may fall back to sequential")
        return True, "windows"
    elif system == "Linux":
        print("Linux detected - parallel processing available")
        return True, "linux"
    elif system == "Darwin":  # macOS
        print("macOS detected - parallel processing available")
        return True, "macos"
    else:
        print(f"Unknown OS {system} - using sequential processing")
        return False, "unknown"

def add_dnn_baselines_sequential():
    """Add DNN baselines sequentially (fallback method)"""
    print(f"\nAdding DNN baselines using sequential processing...")
    print("=" * 80)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(TARGET_PATH) if f.endswith('_results.json')]
    tasks = []
    
    for json_file in json_files:
        if json_file.startswith('results_') or json_file == 'all_results_intermediate.json':
            continue
        
        json_path = os.path.join(TARGET_PATH, json_file)
        
        # Load JSON to check if baseline exists
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check restart configuration
        if 'dnn_baseline' in data:
            if RESTART_CONFIG['mode'] == 'restart' or 3 in RESTART_CONFIG.get('force_restart_modules', []):
                print(f"Reprocessing {json_file} (restart mode)")
            else:
                print(f"Skipping {json_file} (already has baseline)")
                continue
        
        # Parse filename
        parts = json_file.replace('_results.json', '').split('_')
        if len(parts) < 3:
            continue
        
        fingerprint = parts[-1]
        split_type = parts[-2]
        dataset = '_'.join(parts[:-2])
        
        tasks.append((json_file, dataset, split_type, fingerprint, len(tasks) + 1))
    
    print(f"Found {len(tasks)} files needing DNN baseline")
    
    if not tasks:
        print("No files need processing")
        return
    
    # Process sequentially
    start_time = time.time()
    results = []
    
    for i, task in enumerate(tasks, 1):
        print(f"\nProgress: {i}/{len(tasks)}")
        result = calculate_dnn_baseline_worker(task)
        results.append(result)
    
    # Update JSON files with results
    success_count = 0
    for result in results:
        if result:
            json_path = os.path.join(TARGET_PATH, result['json_file'])
            
            # Load original JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Add DNN baseline
            data['dnn_baseline'] = result['dnn_baseline']
            
            # Create organized folder
            output_dir = os.path.join(TARGET_PATH, result['dataset'], result['split_type'], result['fingerprint'])
            os.makedirs(output_dir, exist_ok=True)
            
            # Save in organized location
            organized_path = os.path.join(output_dir, result['json_file'])
            with open(organized_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            success_count += 1
            print(f"Updated: {result['json_file']}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Successfully updated {success_count}/{len(tasks)} files")
    if len(tasks) > 0:
        print(f"Speed: {elapsed/len(tasks):.1f} seconds per combination")

def add_dnn_baselines_parallel(n_workers=4):
    """Add DNN baselines to all JSON files using parallel processing with OS detection"""
    can_parallel, os_type = detect_parallel_capability()
    
    if not can_parallel:
        print("Falling back to sequential processing...")
        return add_dnn_baselines_sequential()
    
    print(f"\nAdding DNN baselines using {n_workers} parallel workers on {os_type}...")
    print("=" * 80)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(TARGET_PATH) if f.endswith('_results.json')]
    tasks = []
    
    for json_file in json_files:
        if json_file.startswith('results_') or json_file == 'all_results_intermediate.json':
            continue
        
        json_path = os.path.join(TARGET_PATH, json_file)
        
        # Load JSON to check if baseline exists
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check restart configuration
        if 'dnn_baseline' in data:
            if RESTART_CONFIG['mode'] == 'restart' or 3 in RESTART_CONFIG.get('force_restart_modules', []):
                print(f"Reprocessing {json_file} (restart mode)")
            else:
                print(f"Skipping {json_file} (already has baseline)")
                continue
        
        # Parse filename
        parts = json_file.replace('_results.json', '').split('_')
        if len(parts) < 3:
            continue
        
        fingerprint = parts[-1]
        split_type = parts[-2]
        dataset = '_'.join(parts[:-2])
        
        tasks.append((json_file, dataset, split_type, fingerprint, len(tasks) + 1))
    
    print(f"Found {len(tasks)} files needing DNN baseline")
    
    if not tasks:
        print("No files need processing")
        return
    
    # Process in parallel
    start_time = time.time()
    
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(calculate_dnn_baseline_worker, tasks)
    except Exception as e:
        print(f"\n!!! Parallel processing failed: {e}")
        print("!!! Switching to sequential processing...")
        return add_dnn_baselines_sequential()
    
    # Update JSON files with results
    success_count = 0
    for result in results:
        if result:
            json_path = os.path.join(TARGET_PATH, result['json_file'])
            
            # Load original JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Add DNN baseline
            data['dnn_baseline'] = result['dnn_baseline']
            
            # Create organized folder
            output_dir = os.path.join(TARGET_PATH, result['dataset'], result['split_type'], result['fingerprint'])
            os.makedirs(output_dir, exist_ok=True)
            
            # Save in both locations
            organized_path = os.path.join(output_dir, result['json_file'])
            with open(organized_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            success_count += 1
            print(f"Updated: {result['json_file']}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Successfully updated {success_count}/{len(tasks)} files")
    print(f"Speed: {elapsed/len(tasks):.1f} seconds per combination")

def evaluate_descriptor_cv5(fingerprints, descriptor, descriptor_name, y_true,
                           test_fingerprints=None, test_descriptor=None, test_y=None,
                           r2_list=None, time_list=None, memory_list=None,
                           dataset="default", split_type="default", desc_idx=None, total_descriptors=None, fingerprint="morgan",
                           combo_info=None):
    """Evaluate descriptor with CV-5 and optional test set evaluation"""
    desc_start = time.time()

    # Optional resource tracking with psutil
    try:
        import psutil
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024
        start_cpu_percent = process.cpu_percent()
        psutil_available = True
    except ImportError:
        start_mem = 0
        start_cpu_percent = 0
        psutil_available = False

    print(f"\n" + "="*90)
    desc_count_str = f" ({desc_idx}/{total_descriptors})" if desc_idx and total_descriptors else ""
    remaining_desc = f" | {total_descriptors - desc_idx} remaining" if desc_idx and total_descriptors else ""
    progress_percent = f" | {(desc_idx/total_descriptors)*100:.1f}%" if desc_idx and total_descriptors else ""

    print(f"  [MODULE 3] DESCRIPTOR EVALUATION: {descriptor_name}{desc_count_str}{remaining_desc}{progress_percent}")
    print(f"  Dataset: {dataset.upper()} | Split: {split_type.upper()} | Fingerprint: {fingerprint}")

    # Add comprehensive scope information
    if combo_info:
        total_combos = combo_info.get('total_combinations', 0)
        current_combo = combo_info.get('current_combination', 0)
        total_datasets = combo_info.get('total_datasets', 0)
        total_splits = combo_info.get('total_splits', 0)
        total_fingerprints = combo_info.get('total_fingerprints', 0)

        # Get current position indices
        current_dataset_idx = combo_info.get('current_dataset_idx', 0)
        current_split_idx = combo_info.get('current_split_idx', 0)
        current_fingerprint_idx = combo_info.get('current_fingerprint_idx', 0)

        # Calculate total evaluations across all combinations
        total_evaluations = total_combos * total_descriptors if total_descriptors else 0
        current_evaluation = ((current_combo - 1) * total_descriptors + desc_idx) if desc_idx and total_descriptors else 0

        print(f"  Scope: Dataset ({current_dataset_idx}/{total_datasets}) × Split ({current_split_idx}/{total_splits}) × Fingerprint ({current_fingerprint_idx}/{total_fingerprints}) × Descriptor ({desc_idx}/{total_descriptors})")
        print(f"  Overall Progress: Evaluation {current_evaluation:,}/{total_evaluations:,} ({(current_evaluation/total_evaluations)*100:.2f}%)")
        print(f"  Totals: {total_datasets} datasets × {total_splits} splits × {total_fingerprints} fingerprints × {total_descriptors} descriptors = {total_evaluations:,} evaluations")

    print(f"  Features: Fingerprint({fingerprint}) + {descriptor_name}")

    # Add timing information if available (pass from calling function)
    if hasattr(evaluate_descriptor_cv5, '_avg_desc_time') and evaluate_descriptor_cv5._avg_desc_time > 0:
        if desc_idx and total_descriptors:
            remaining_time = evaluate_descriptor_cv5._avg_desc_time * (total_descriptors - desc_idx)
            if remaining_time < 60:
                eta_str = f"{remaining_time:.1f}s"
            elif remaining_time < 3600:
                eta_str = f"{remaining_time/60:.1f}m"
            else:
                eta_str = f"{remaining_time/3600:.1f}h"
            print(f"  Current Combo ETA: {eta_str} | Avg: {evaluate_descriptor_cv5._avg_desc_time:.1f}s/desc")

    print("="*90)
    
    # Prepare features
    if isinstance(descriptor, list):
        descriptor = np.array(descriptor)
    
    # Ensure descriptor is 2D (n_samples, n_features)
    if len(descriptor.shape) == 1:
        descriptor = descriptor.reshape(-1, 1)
    # If descriptor is already 2D (like PMI with shape (396, 3)), keep as is
    
    # Combine fingerprints and descriptor
    X = np.hstack([fingerprints, descriptor])
    print(f"    🔗 Combined features: FP({fingerprints.shape[1]}) + Desc({descriptor.shape[1]}) = {X.shape[1]} dimensions")

    # Dual CV evaluation - Both Type1 and Type2
    cv_results = evaluate_model_cv_both_types(X, y_true, test_size=0.2, n_folds=CV)

    # Extract Type1 (Research) results for primary metrics
    type1_results = cv_results['type1']
    cv_mean = type1_results['r2_mean']  # Use Type1 for primary evaluation
    cv_std = type1_results['r2_std']

    # Extract Type2 (Production) results
    type2_results = cv_results['type2']
    type2_cv_stats = type2_results['cv_stats']
    type2_final_metrics = type2_results['final_metrics']

    print(f"    [TYPE1-Research] CV Results: R²={cv_mean:.4f}±{cv_std:.4f}, RMSE={type1_results['rmse_mean']:.4f}±{type1_results['rmse_std']:.4f}, MSE={type1_results['rmse_mean']**2:.4f}±{(type1_results['rmse_std']**2):.4f}, MAE={type1_results['mae_mean']:.4f}±{type1_results['mae_std']:.4f}")
    print(f"    [TYPE2-Production] CV Results: R²={type2_cv_stats['val_r2_mean']:.4f}±{type2_cv_stats['val_r2_std']:.4f}, RMSE={type2_cv_stats['val_rmse_mean']:.4f}±{type2_cv_stats['val_rmse_std']:.4f}, MSE={type2_cv_stats['val_rmse_mean']**2:.4f}±{(type2_cv_stats['val_rmse_std']**2):.4f}, MAE={type2_cv_stats['val_mae_mean']:.4f}±{type2_cv_stats['val_mae_std']:.4f}")
    print(f"    [TYPE2-Production] Final Test: R²={type2_final_metrics['r2']:.4f}, RMSE={type2_final_metrics['rmse']:.4f}, MSE={type2_final_metrics['rmse']**2:.4f}, MAE={type2_final_metrics['mae']:.4f}")
    print(f"    📌 [OPTUNA] Using Type1 R²={cv_mean:.4f} for optimization")
    
    # Test set evaluation if provided
    test_score = 0.0
    if test_fingerprints is not None and test_y is not None:
        # Prepare test features
        if isinstance(test_descriptor, list):
            test_descriptor = np.array(test_descriptor)
        
        # Ensure descriptor is 2D
        if len(test_descriptor.shape) == 1:
            test_descriptor = test_descriptor.reshape(-1, 1)
        
        X_test = np.hstack([test_fingerprints, test_descriptor])
        
        # Train on full training set using configured method
        # Skip StandardScaler - use raw features
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X)
        # X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X
        X_test_scaled = X_test

        # Train model using adaptive method
        # Architecture dynamically adapts to input dimension
        input_dim = X_train_scaled.shape[1]
        architecture = [input_dim, 1024, 496, 1]
        test_score, test_rmse, test_mae = train_model_adaptive(
            X_train_scaled, y_true, X_test_scaled, test_y,
            batch_size=BATCHSIZE, epochs=EPOCHS, lr=LR, architecture=architecture
        )
        
        
        # No cleanup needed for direct training method
    
    desc_end = time.time()
    desc_time = desc_end - desc_start

    # Calculate resource usage (if psutil available)
    if psutil_available:
        end_mem = process.memory_info().rss / 1024 / 1024
        end_cpu_percent = process.cpu_percent()
        mem_used = end_mem - start_mem
        avg_cpu_usage = (start_cpu_percent + end_cpu_percent) / 2
    else:
        mem_used = 0
        avg_cpu_usage = 0

    # Final results for this descriptor
    print(f"\n  🧪 DESCRIPTOR RESULTS: {descriptor_name}")
    print("  " + "-"*50)
    if test_fingerprints is not None and test_y is not None:
        test_mse = test_rmse ** 2
        print(f"    📊 CV-5 Performance:  R² = {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    🎯 Test Performance:  R² = {test_score:.4f}, RMSE = {test_rmse:.4f}")
        print(f"                          MAE = {test_mae:.4f}, MSE = {test_mse:.4f}")
    else:
        print(f"    📊 CV-5 Performance:  R² = {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"    ⏱️  Resource Usage:     Time = {desc_time:.2f}s, Memory = {mem_used:.1f}MB, CPU = {avg_cpu_usage:.1f}%")
    print("  " + "-"*50)

    return cv_mean, cv_std, test_score, desc_time, mem_used, avg_cpu_usage

def evaluate_simple_descriptor(fps, mols, y_true, test_fps, test_mols, test_y,
                             descriptor_func, descriptor_name, r2_list, time_list, memory_list,
                             dataset="default", split_type="default"):
    """Evaluate simple descriptor that can be directly calculated"""
    try:
        # Calculate descriptor for train set
        descriptor = []
        for mol in mols:
            val = descriptor_func(mol)
            descriptor.append(val)
        
        # Convert to numpy array - handles both single values and lists
        descriptor = np.array(descriptor)
        
        # Calculate descriptor for test set
        test_descriptor = None
        if test_mols:
            test_descriptor = []
            for mol in test_mols:
                val = descriptor_func(mol)
                test_descriptor.append(val)
            test_descriptor = np.array(test_descriptor)
        
        # Check if normalization is needed
        if descriptor_name in DESCRIPTORS_NEED_NORMALIZATION:
            descriptor = Normalization(descriptor)
            if test_descriptor is not None:
                test_descriptor = Normalization(test_descriptor)
        
        # Evaluate with CV-5
        cv_mean_r2, cv_std_r2, test_r2, desc_time, mem_used, cpu_used = evaluate_descriptor_cv5(
            fps, descriptor, descriptor_name, y_true, test_fps, test_descriptor, test_y,
            dataset=dataset, split_type=split_type, fingerprint="morgan")
        
        r2_list[descriptor_name] = (cv_mean_r2, cv_std_r2, test_r2)
        time_list[descriptor_name] = desc_time
        memory_list[descriptor_name] = mem_used
        
    except Exception as e:
        print(f"  ❌ Error evaluating {descriptor_name}: {str(e)}")
        print(f"  {descriptor_name}: CV R² = 0.0000 ± 0.0000, Test R² = 0.0000 (ERROR)")
        r2_list[descriptor_name] = (0.0, 0.0, 0.0)
        time_list[descriptor_name] = 0.0
        memory_list[descriptor_name] = 0.0

def descriptors_list(fps, mols, y_true, test_fps=None, test_mols=None, test_y=None,
                    dataset="default", split_type="default", fingerprint="morgan", combo_info=None):
    """Load precomputed descriptors and evaluate them"""
    r2_list = {}
    time_list = {}
    memory_list = {}
    
    print(f"\nEvaluating descriptors for {dataset} dataset with {split_type} split using {fingerprint} fingerprint...")
    print(f"Dataset location: {dataset}/{split_type}")

    # Load precomputed descriptors
    descriptor_dir = f"result/chemical_descriptors/{dataset}/{split_type}"
    train_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_train_descriptors.npz"
    test_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_test_descriptors.npz"
    
    if not os.path.exists(train_desc_file) or not os.path.exists(test_desc_file):
        print(f"  Warning: Precomputed descriptors not found at {descriptor_dir}")
        print(f"⚠️ Descriptor cache not found, generating for {dataset}/{split_type}...")

        # Generate descriptors automatically using ChemDescriptorCalculator
        from extra_code.chem_descriptor_maker import ChemDescriptorCalculator
        calculator = ChemDescriptorCalculator(cache_dir='result/chemical_descriptors')

        # Calculate descriptors for train set
        if not os.path.exists(train_desc_file):
            print(f"  📊 Calculating train descriptors...")
            calculator.calculate_selected_descriptors(
                mols, dataset_name=dataset, split_type=split_type, subset='train'
            )

        # Calculate descriptors for test set
        if not os.path.exists(test_desc_file) and test_mols is not None:
            print(f"  📊 Calculating test descriptors...")
            calculator.calculate_selected_descriptors(
                test_mols, dataset_name=dataset, split_type=split_type, subset='test'
            )

        # Check again after generation
        if not os.path.exists(train_desc_file) or not os.path.exists(test_desc_file):
            print("  ❌ Failed to generate descriptors, skipping evaluation...")
            return r2_list, time_list, memory_list
        else:
            print(f"  ✅ Descriptors generated and cached successfully")
    else:
        print(f"✅ Using cached descriptors from {dataset}/{split_type}")

    # Load precomputed descriptor data
    train_data = np.load(train_desc_file, allow_pickle=True)
    test_data = np.load(test_desc_file, allow_pickle=True)
    
    # Use descriptor names from precomputed data (exclude metadata keys)
    descriptor_names = [key for key in train_data.keys() if key not in ['train_mols', 'test_mols', 'descriptor_array', '3d_conformers']]

    # Initialize timing tracking for descriptors
    desc_times = []

    # Initialize average time tracking
    if not hasattr(evaluate_descriptor_cv5, '_avg_desc_time'):
        evaluate_descriptor_cv5._avg_desc_time = 0

    # Iterate through each descriptor and evaluate
    for desc_idx, desc_name in enumerate(descriptor_names, 1):
        if desc_name not in train_data or desc_name not in test_data:
            continue
            
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Get precomputed descriptor values
            train_desc = train_data[desc_name]
            test_desc = test_data[desc_name]
            
            # Reshape if needed
            if len(train_desc.shape) == 1:
                train_desc = train_desc.reshape(-1, 1)
            if len(test_desc.shape) == 1:
                test_desc = test_desc.reshape(-1, 1)
            
            # Combine with fingerprints
            X_train = np.hstack([fps, train_desc])
            X_test = np.hstack([test_fps, test_desc]) if test_fps is not None else None

            # Calculate and update average descriptor time for ETA
            if desc_times:
                avg_time = sum(desc_times) / len(desc_times)
                evaluate_descriptor_cv5._avg_desc_time = avg_time

            # Evaluate
            cv_mean, cv_std, test_r2, desc_time, desc_mem, cpu_used = evaluate_descriptor_cv5(
                fps, train_desc, desc_name, y_true,
                test_fps, test_desc, test_y,
                dataset=dataset, split_type=split_type,
                desc_idx=desc_idx, total_descriptors=len(descriptor_names), fingerprint=fingerprint,
                combo_info=combo_info
            )

            # Track timing for future ETA calculations
            desc_times.append(desc_time)

            # Store results as tuple (cv_mean, cv_std, test_r2)
            r2_list[desc_name] = (cv_mean, cv_std, test_r2)
            time_list[desc_name] = desc_time
            memory_list[desc_name] = desc_mem

            print(f"  {desc_name}: CV R² = {cv_mean:.4f} ± {cv_std:.4f}, Test R² = {test_r2:.4f} (time: {desc_time:.2f}s)")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {desc_name}: {e}")
            print(f"  {desc_name}: CV R² = 0.0000 ± 0.0000, Test R² = 0.0000 (ERROR)")
            r2_list[desc_name] = (0.0, 0.0, 0.0)
            time_list[desc_name] = 0.0
            memory_list[desc_name] = 0.0
            continue
    
    return r2_list, time_list, memory_list


def retry_failed_descriptors(failed_descriptor_names, train_fps, train_mols, train_y,
                            test_fps, test_mols, test_y, dataset, split_type, fingerprint):
    """Retry specific descriptors that failed (R² ≤ 0.0)"""
    retry_r2_dict = {}
    retry_time_dict = {}
    retry_memory_dict = {}

    print(f"\n🔄 RETRYING FAILED DESCRIPTORS:")

    # Load precomputed descriptor data
    descriptor_dir = f"result/chemical_descriptors/{dataset}/{split_type}"
    train_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_train_descriptors.npz"
    test_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_test_descriptors.npz"

    if not os.path.exists(train_desc_file) or not os.path.exists(test_desc_file):
        print(f"  ❌ No precomputed descriptors found, cannot retry")
        return retry_r2_dict, retry_time_dict, retry_memory_dict

    # Load precomputed descriptor data
    train_data = np.load(train_desc_file, allow_pickle=True)
    test_data = np.load(test_desc_file, allow_pickle=True)

    # Retry each failed descriptor individually
    for desc_idx, desc_name in enumerate(failed_descriptor_names, 1):
        print(f"  🔄 [{desc_idx}/{len(failed_descriptor_names)}] Retrying {desc_name}...")

        if desc_name not in train_data or desc_name not in test_data:
            print(f"    ❌ {desc_name} not found in precomputed data")
            retry_r2_dict[desc_name] = (0.0, 0.0, 0.0)
            retry_time_dict[desc_name] = 0.0
            retry_memory_dict[desc_name] = 0.0
            continue

        try:
            # Get descriptor data
            train_desc = train_data[desc_name]
            test_desc = test_data[desc_name]

            # Convert to proper format
            if isinstance(train_desc, list):
                train_desc = np.array(train_desc)
            if isinstance(test_desc, list):
                test_desc = np.array(test_desc)

            # Evaluate with more robust error handling
            cv_mean, cv_std, test_r2, desc_time, desc_mem, cpu_used = evaluate_descriptor_cv5(
                train_fps, train_desc, desc_name, train_y,
                test_fps, test_desc, test_y,
                dataset=dataset, split_type=split_type, desc_idx=desc_idx,
                total_descriptors=len(failed_descriptor_names), fingerprint=fingerprint
            )

            # Store results
            retry_r2_dict[desc_name] = (cv_mean, cv_std, test_r2)
            retry_time_dict[desc_name] = desc_time
            retry_memory_dict[desc_name] = desc_mem

            if cv_mean > 0.0:
                print(f"    ✅ {desc_name}: CV R² = {cv_mean:.4f} ± {cv_std:.4f}, Test R² = {test_r2:.4f}")
            else:
                print(f"    ❌ {desc_name}: Still failed (CV R² = {cv_mean:.4f})")

        except Exception as e:
            print(f"    ❌ {desc_name}: Retry failed with error: {e}")
            retry_r2_dict[desc_name] = (0.0, 0.0, 0.0)
            retry_time_dict[desc_name] = 0.0
            retry_memory_dict[desc_name] = 0.0

    print(f"🔄 Retry completed: {len([d for d in retry_r2_dict.values() if d[0] > 0.0])}/{len(failed_descriptor_names)} successful")
    return retry_r2_dict, retry_time_dict, retry_memory_dict


def process_combination(dataset, split_type, fingerprint, x_map, y_map, fingerprint_map, skip_baseline=False, retry_failed=False, combo_info=None):
    """Process a single dataset-split-fingerprint combination using preprocessed data

    Args:
        retry_failed (bool): If True, retry descriptors with R² ≤ 0.0
    """
    print(f"\n{'='*80}")
    print(f"📊 PROCESSING COMBINATION")
    print(f"   Dataset: {get_dataset_display_name(dataset)} ({dataset})")
    print(f"   Split: {SPLIT_TYPES.get(split_type, split_type)} ({split_type})")
    print(f"   Fingerprint: {fingerprint}")
    print(f"{'='*80}")
    
    combo_start = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        # Get preprocessed data from maps
        train_key = f"{dataset}_train"
        test_key = f"{dataset}_test"
        
        # Check if data exists
        if split_type not in x_map or train_key not in x_map[split_type]:
            print(f"Skipping {dataset} - {split_type}: data not found")
            return None
        
        # Get train and test data
        train_smiles = x_map[split_type][train_key]
        train_y = y_map[split_type][train_key]
        test_smiles = x_map[split_type][test_key]
        test_y = y_map[split_type][test_key]
        
        # Optimized molecular processing using ThreadPoolExecutor
        def convert_smiles_to_mol(smiles):
            return Chem.MolFromSmiles(smiles)
        
        # Use ThreadPoolExecutor for parallel molecular conversion
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            train_future = executor.submit(lambda: [convert_smiles_to_mol(s) for s in train_smiles])
            test_future = executor.submit(lambda: [convert_smiles_to_mol(s) for s in test_smiles])
            
            train_mols = train_future.result()
            test_mols = test_future.result()
        
        # Filter out None molecules for training
        train_valid_idx = [i for i, mol in enumerate(train_mols) if mol is not None]
        train_mols = [train_mols[i] for i in train_valid_idx]
        train_y = train_y[train_valid_idx]
        
        # Filter out None molecules for testing
        test_valid_idx = [i for i, mol in enumerate(test_mols) if mol is not None]
        test_mols = [test_mols[i] for i in test_valid_idx]
        test_y = test_y[test_valid_idx]
        
        print(f"Loaded {len(train_mols)} train and {len(test_mols)} test molecules")
        
        # Get precomputed fingerprints using get_fingerprints_combined
        # This follows Module 4's pattern - use precomputed data
        print(f"  Loading precomputed {fingerprint} fingerprints...")
        
        # Convert fingerprint type for get_fingerprints_combined
        # 'all' means morgan+maccs+avalon
        fp_type = fingerprint
        if fingerprint == 'morgan+maccs+avalon':
            fp_type = 'all'
        
        # Get precomputed fingerprints from mol_fps_maker
        train_fingerprints = get_fingerprints_combined(
            train_mols, 
            dataset,  # Dataset name (mol_fps_maker converts to lowercase internally)
            split_type, 
            'train',
            fingerprint_type=fp_type,
            module_name='3_solubility_feature_deeplearning'
        )
        
        test_fingerprints = get_fingerprints_combined(
            test_mols,
            dataset,  # Dataset name (mol_fps_maker converts to lowercase internally)
            split_type,
            'test', 
            fingerprint_type=fp_type,
            module_name='3_solubility_feature_deeplearning'
        )
        
        print(f"  Loaded precomputed fingerprints: train shape {train_fingerprints.shape}, test shape {test_fingerprints.shape}")
        
        # Calculate DNN baseline (default behavior)
        if not skip_baseline:
            baseline_cv, baseline_cv_std, baseline_test, baseline_time = calculate_dnn_baseline(
                train_fingerprints, train_y, test_fingerprints, test_y
            )
        else:
            print("  ⏭️ Skipping DNN baseline calculation (skip_baseline=True)")
            print("  Baseline: CV R² = 0.0000 ± 0.0000, Test R² = 0.0000 (SKIPPED)")
            baseline_cv, baseline_cv_std, baseline_test, baseline_time = 0.0, 0.0, 0.0, 0.0
        
        # Calculate all descriptors with test evaluation
        r2_dict, time_dict, memory_dict = descriptors_list(
            train_fingerprints, train_mols, train_y,
            test_fingerprints, test_mols, test_y,
            dataset, split_type, fingerprint, combo_info
        )

        # If retry_failed is enabled, retry descriptors with R² ≤ 0.0
        if retry_failed:
            failed_descriptors = [desc for desc, (cv_r2, cv_std, test_r2) in r2_dict.items()
                                 if cv_r2 <= 0.0]
            if failed_descriptors:
                print(f"\n🔄 RETRYING {len(failed_descriptors)} FAILED DESCRIPTORS:")
                for desc in failed_descriptors[:5]:  # Show first 5
                    print(f"   - {desc}")
                if len(failed_descriptors) > 5:
                    print(f"   ... and {len(failed_descriptors) - 5} more")

                # Retry failed descriptors one by one
                retry_r2_dict, retry_time_dict, retry_memory_dict = retry_failed_descriptors(
                    failed_descriptors, train_fingerprints, train_mols, train_y,
                    test_fingerprints, test_mols, test_y, dataset, split_type, fingerprint
                )

                # Update results with successful retries
                for desc in failed_descriptors:
                    if desc in retry_r2_dict and retry_r2_dict[desc][0] > 0.0:
                        print(f"   ✅ {desc}: {r2_dict[desc][0]:.4f} → {retry_r2_dict[desc][0]:.4f}")
                        r2_dict[desc] = retry_r2_dict[desc]
                        time_dict[desc] = retry_time_dict[desc]
                        memory_dict[desc] = retry_memory_dict[desc]
                    else:
                        print(f"   ❌ {desc}: Still failed after retry")
        
        # Record process stats
        combo_end = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Find best descriptor based on CV score
        if r2_dict:
            best_descriptor = max(r2_dict.items(), key=lambda x: x[1][0])
        else:
            print(f"  ⚠️ WARNING: No descriptors evaluated successfully for {dataset}-{split_type}-{fingerprint}")
            print(f"  None: CV R² = 0.0000 ± 0.0000, Test R² = 0.0000 (NO DESCRIPTORS)")
            best_descriptor = ('None', (0.0, 0.0, 0.0))
        
        # Use display name from config
        display_name = DATASET_DISPLAY_NAMES.get(DATASET_MAP.get(dataset, dataset), dataset)
        
        # Create result
        result = {
            'dataset': dataset,
            'dataset_display_name': display_name,
            'split_type': split_type,
            'fingerprint': fingerprint,
            'n_train_molecules': len(train_mols),
            'n_test_molecules': len(test_mols),
            'dnn_baseline': {
                'cv_r2': baseline_cv,
                'cv_std': baseline_cv_std,
                'test_r2': baseline_test,
                'time': baseline_time
            },
            'descriptors': {k: {
                'cv_mean_r2': v[0], 
                'cv_std_r2': v[1],
                'test_r2': v[2],
                'cv_test_mean_r2': v[3] if len(v) > 3 else v[2],
                'cv_test_std_r2': v[4] if len(v) > 4 else 0.0
            } for k, v in r2_dict.items()},
            'descriptor_times': time_dict,
            'descriptor_memory': memory_dict,
            'best_descriptor': best_descriptor[0],
            'best_cv_r2': best_descriptor[1][0],
            'best_cv_std': best_descriptor[1][1],
            'best_test_r2': best_descriptor[1][2],
            'total_time': combo_end - combo_start,
            'avg_cpu_percent': psutil.Process().cpu_percent(interval=0.1),
            'max_memory_mb': end_memory - start_memory
        }
        
        # Save individual result in organized folder structure
        # Create folder structure: dataset/split_type/fingerprint/
        output_dir = os.path.join(TARGET_PATH, dataset, split_type, fingerprint)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in the organized folder
        output_file = os.path.join(output_dir, f'{dataset}_{split_type}_{fingerprint}_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nBest descriptor: {best_descriptor[0]}")
        print(f"  CV Val R²: {best_descriptor[1][0]:.4f} ± {best_descriptor[1][1]:.4f}")
        if len(best_descriptor[1]) > 4:
            print(f"  CV Test R²: {best_descriptor[1][3]:.4f} ± {best_descriptor[1][4]:.4f}")
        print(f"  Final Test R²: {best_descriptor[1][2]:.4f}")
        print(f"Processing time: {combo_end - combo_start:.2f} seconds")
        print(f"Memory used: {end_memory - start_memory:.2f} MB")
        print(f"Result created successfully with {len(r2_dict)} descriptors evaluated")
        print(f"Saving to {output_file}...")
        
        return result
        
    except Exception as e:
        print(f"Error processing {dataset} - {split_type} - {fingerprint}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_comparison_plots():
    """Generate separate plots with and without baseline for all results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("Generating comparison plots for all JSON files...")

    # Find all JSON files in organized folders
    plot_count = 0
    for dataset in os.listdir(TARGET_PATH):
        dataset_path = os.path.join(TARGET_PATH, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split_type in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split_type)
            if not os.path.isdir(split_path):
                continue

            for fingerprint in os.listdir(split_path):
                fp_path = os.path.join(split_path, fingerprint)
                if not os.path.isdir(fp_path):
                    continue

                # Find JSON file
                json_files = [f for f in os.listdir(fp_path) if f.endswith('_results.json')]
                if not json_files:
                    continue
                    
                json_path = os.path.join(fp_path, json_files[0])
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Handle both direct and nested JSON structure
                if 'result' in data:
                    result_data = data['result']
                else:
                    result_data = data

                # Extract descriptor performances
                if 'descriptors' not in result_data:
                    continue

                descriptors = result_data['descriptors']

                # Ensure all 49 descriptors from config are included
                from config import CHEMICAL_DESCRIPTORS
                filtered_descriptors = {}

                # Add all config descriptors first (with defaults if missing)
                for desc_name in CHEMICAL_DESCRIPTORS:
                    if desc_name in descriptors:
                        v = descriptors[desc_name]
                        cv_mean = v.get('cv_mean_r2', 0.0)
                        cv_std = v.get('cv_std_r2', 0.0)
                        test_r2 = v.get('test_r2', 0.0)
                    else:
                        # Missing descriptor - use zeros
                        cv_mean = cv_std = test_r2 = 0.0

                    filtered_descriptors[desc_name] = {
                        'cv_mean_r2': cv_mean,
                        'cv_std_r2': cv_std,
                        'test_r2': test_r2
                    }

                # Add any extra descriptors found in JSON but not in config
                for k, v in descriptors.items():
                    if k not in ['descriptor_array', '3d_conformers'] and k not in CHEMICAL_DESCRIPTORS:
                        cv_mean = v.get('cv_mean_r2', 0.0)
                        cv_std = v.get('cv_std_r2', 0.0)
                        test_r2 = v.get('test_r2', 0.0)

                        filtered_descriptors[k] = {
                            'cv_mean_r2': cv_mean,
                            'cv_std_r2': cv_std,
                            'test_r2': test_r2
                        }

                sorted_desc = sorted(filtered_descriptors.items(),
                                   key=lambda x: x[1]['cv_mean_r2'],
                                   reverse=True)  # ALL descriptors, not just top 10
                
                names = [d[0] for d in sorted_desc]  # Full descriptor names
                cv_scores = [d[1]['cv_mean_r2'] for d in sorted_desc]
                cv_stds = [d[1]['cv_std_r2'] for d in sorted_desc]
                test_scores = [d[1]['test_r2'] for d in sorted_desc]

                print(f"  Generated plots for: {dataset}/{split_type}/{fingerprint}")
                print(f"    Total descriptors: {len(names)} (expecting 49)")
                print(f"    CV R² range: {min(cv_scores):.3f} - {max(cv_scores):.3f}")
                print(f"    CV Std range: {min(cv_stds):.3f} - {max(cv_stds):.3f}")
                print(f"    Test R² range: {min(test_scores):.3f} - {max(test_scores):.3f}")

                # Check for zero or negative values (failed descriptors)
                failed_descriptors = [(names[i], cv_scores[i]) for i in range(len(names))
                                     if cv_scores[i] <= 0.0]
                if failed_descriptors:
                    print(f"    ⚠️ Warning: {len(failed_descriptors)} descriptors with R² ≤ 0 (failed):")
                    for desc_name, r2_val in failed_descriptors[:5]:  # Show first 5
                        print(f"      - {desc_name}: R² = {r2_val:.4f}")
                    if len(failed_descriptors) > 5:
                        print(f"      ... and {len(failed_descriptors) - 5} more")
                
                # Create plots directory
                plot_dir = os.path.join(fp_path, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                
                # Plot 1: WITH baseline (only if baseline exists)
                if 'dnn_baseline' in result_data and result_data['dnn_baseline']:
                    # Adjust figure size based on number of descriptors (much wider for readability)
                    fig_width = max(40, len(names) * 1.0)  # Even wider spacing for large text
                    fig1, ax1 = plt.subplots(figsize=(fig_width, 14))  # Even taller and wider

                    # Bar positions with wider spacing
                    x = np.arange(len(names))
                    bar_width = 0.45  # Even wider bars for large text
                    
                    bars1 = ax1.bar(x - bar_width/2, cv_scores, bar_width, label='CV R²', color='skyblue',
                                   yerr=cv_stds, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.8})  # Enhanced error bars
                    bars2 = ax1.bar(x + bar_width/2, test_scores, bar_width, label='Test R²', color='orange')
                    
                    # Add R² values inside bars (rotated 90 degrees)
                    for i, (bar1, bar2, cv_val, cv_std, test_val) in enumerate(zip(bars1, bars2, cv_scores, cv_stds, test_scores)):
                        # CV R² value with ± std inside CV bar (from bottom)
                        height1 = bar1.get_height()
                        if height1 > 0.05:  # Show if bar height > 0.05
                            # Position from bottom of bar (more readable)
                            y_pos1 = height1 * 0.1  # 10% from bottom
                            ax1.text(bar1.get_x() + bar1.get_width()/2., y_pos1,
                                    f'{cv_val:.3f}±{cv_std:.3f}',
                                    ha='center', va='bottom', fontsize=16, rotation=90,
                                    color='black', fontweight='bold')

                        # Test R² value inside Test bar (from bottom)
                        height2 = bar2.get_height()
                        if height2 > 0.05:  # Show if bar height > 0.05
                            # Position from bottom of bar (more readable)
                            y_pos2 = height2 * 0.1  # 10% from bottom
                            ax1.text(bar2.get_x() + bar2.get_width()/2., y_pos2,
                                    f'{test_val:.3f}',
                                    ha='center', va='bottom', fontsize=16, rotation=90,
                                    color='black', fontweight='bold')
                    
                    baseline_cv = result_data['dnn_baseline']['cv_r2']
                    baseline_test = result_data['dnn_baseline']['test_r2']
                    ax1.axhline(baseline_cv, color='red', linestyle='--', 
                              label=f'Baseline CV: {baseline_cv:.3f}')
                    ax1.axhline(baseline_test, color='darkred', linestyle='--',
                              label=f'Baseline Test: {baseline_test:.3f}')
                    
                    ax1.set_xlabel('Chemical Descriptors', fontsize=18, fontweight='bold')
                    ax1.set_ylabel('R² Score', fontsize=18, fontweight='bold')
                    ax1.set_title(f'DNN Performance Enhancement with Individual Chemical Descriptors\n{dataset.upper()} Dataset - {split_type.upper()} Split - {fingerprint.capitalize()} Fingerprint', fontsize=20, fontweight='bold')
                    ax1.set_xticks(x)
                    # Show ALL descriptor names with enhanced visibility
                    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=14,
                                       fontweight='bold', color='black')
                    ax1.legend(fontsize=16)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim([0, 1])
                    
                    plt.tight_layout()
                    # Adjust bottom margin for rotated X-axis labels
                    plt.subplots_adjust(bottom=0.15)
                    plot_path_with = os.path.join(plot_dir,
                        f'{dataset}_{split_type}_{fingerprint}_with_baseline.png')
                    plt.savefig(plot_path_with, dpi=200, bbox_inches='tight')
                    plt.close()
                
                # Plot 2: WITHOUT baseline
                # Adjust figure size based on number of descriptors (much wider for readability)
                fig_width = max(40, len(names) * 1.0)  # Even wider spacing for large text
                fig2, ax2 = plt.subplots(figsize=(fig_width, 14))  # Even taller and wider

                x = np.arange(len(names))
                bar_width = 0.45  # Even wider bars for large text

                bars1 = ax2.bar(x - bar_width/2, cv_scores, bar_width, label='CV R²', color='skyblue',
                               yerr=cv_stds, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.8})  # Enhanced error bars
                bars2 = ax2.bar(x + bar_width/2, test_scores, bar_width, label='Test R²', color='orange')
                
                # Add R² values inside bars (rotated 90 degrees)
                for i, (bar1, bar2, cv_val, cv_std, test_val) in enumerate(zip(bars1, bars2, cv_scores, cv_stds, test_scores)):
                    # CV R² value with ± std inside CV bar (from bottom)
                    height1 = bar1.get_height()
                    if height1 > 0.05:  # Show if bar height > 0.05
                        # Position from bottom of bar (more readable)
                        y_pos1 = height1 * 0.1  # 10% from bottom
                        ax2.text(bar1.get_x() + bar1.get_width()/2., y_pos1,
                                f'{cv_val:.3f}±{cv_std:.3f}',
                                ha='center', va='bottom', fontsize=16, rotation=90,
                                color='black', fontweight='bold')

                    # Test R² value inside Test bar (from bottom)
                    height2 = bar2.get_height()
                    if height2 > 0.05:  # Show if bar height > 0.05
                        # Position from bottom of bar (more readable)
                        y_pos2 = height2 * 0.1  # 10% from bottom
                        ax2.text(bar2.get_x() + bar2.get_width()/2., y_pos2,
                                f'{test_val:.3f}',
                                ha='center', va='bottom', fontsize=16, rotation=90,
                                color='black', fontweight='bold')
                
                ax2.set_xlabel('Chemical Descriptors', fontsize=18, fontweight='bold')
                ax2.set_ylabel('R² Score', fontsize=18, fontweight='bold')
                ax2.set_title(f'DNN Performance Enhancement with Individual Chemical Descriptors\n{dataset.upper()} Dataset - {split_type.upper()} Split - {fingerprint.capitalize()} Fingerprint', fontsize=20, fontweight='bold')
                ax2.set_xticks(x)
                # Show ALL descriptor names with enhanced visibility
                ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=14,
                                   fontweight='bold', color='black')
                ax2.legend(fontsize=16)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, 1])
                
                plt.tight_layout()
                # Adjust bottom margin for rotated X-axis labels
                plt.subplots_adjust(bottom=0.15)
                plot_path_without = os.path.join(plot_dir,
                    f'{dataset}_{split_type}_{fingerprint}_without_baseline.png')
                plt.savefig(plot_path_without, dpi=200, bbox_inches='tight')
                plt.close()
                
                plot_count += 2  # We generate 2 plots per JSON
                print(f"  Generated plots for: {dataset}/{split_type}/{fingerprint}")
    
    print(f"\nGenerated {plot_count} comparison plots (with and without baseline)")

def create_fingerprint_combination_plot(dataset, split_type, fingerprint, results, output_dir):
    """Create visualization after each fingerprint combination completes"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Create output directory - same location as JSON results
    plot_dir = Path(output_dir) / dataset / split_type / fingerprint
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract metrics
    models = []
    cv_scores = []
    test_scores = []

    # Check if 'result' key exists (for saved JSON format)
    if 'result' in results:
        data = results['result']
    else:
        data = results

    # Add baseline DNN results
    if 'dnn_baseline' in data:
        models.append('DNN Baseline')
        cv_scores.append(data['dnn_baseline'].get('cv_r2', 0))
        test_scores.append(data['dnn_baseline'].get('test_r2', 0))

    # Add descriptor results
    if 'descriptors' in data:
        # Collect all descriptor scores
        desc_scores = []
        for desc_name, metrics in data['descriptors'].items():
            desc_scores.append({
                'name': desc_name,
                'cv': metrics.get('cv_mean_r2', 0),
                'test': metrics.get('test_r2', 0)
            })

        # Sort by test score and take top 10
        desc_scores.sort(key=lambda x: x['test'], reverse=True)
        for desc in desc_scores[:10]:
            # Shorten long descriptor names for display
            display_name = desc['name']
            if len(display_name) > 15:
                display_name = display_name[:12] + '...'
            models.append(display_name)
            cv_scores.append(desc['cv'])
            test_scores.append(desc['test'])

    # Plot 1: Bar chart of model performance
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV R²', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.8)

    # Color bars based on performance
    for bar, score in zip(bars1, cv_scores):
        color = '#2ecc71' if score > 0.6 else '#3498db' if score > 0.5 else '#f39c12' if score > 0.4 else '#e74c3c'
        bar.set_color(color)

    for bar, score in zip(bars2, test_scores):
        color = '#27ae60' if score > 0.6 else '#2980b9' if score > 0.5 else '#d68910' if score > 0.4 else '#a93226'
        bar.set_color(color)

    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title(f'{dataset.upper()} - {split_type.upper()} - {fingerprint}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

    # Plot 2: Scatter plot of CV vs Test scores
    ax2.scatter(cv_scores, test_scores, s=100, alpha=0.6, edgecolors='black', linewidth=1)

    # Add diagonal line
    max_val = max(max(cv_scores, default=0), max(test_scores, default=0))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')

    # Add labels for each point
    for i, model in enumerate(models):
        ax2.annotate(model[:10], (cv_scores[i], test_scores[i]),
                    fontsize=8, ha='center', va='bottom')

    ax2.set_xlabel('CV R² Score', fontsize=12)
    ax2.set_ylabel('Test R² Score', fontsize=12)
    ax2.set_title('CV vs Test Performance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # Add summary text
    fig.suptitle(f'Module 3 Results: {dataset.upper()}/{split_type}/{fingerprint}\n' +
                 f'Best Test R²: {max(test_scores, default=0):.4f}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save plot
    plot_filename = f"{dataset}_{split_type}_{fingerprint}_results.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    📊 Saved plot: {plot_path}")

    return str(plot_path)

def create_descriptor_visualization(all_results, output_dir):
    """Create visualization showing config.py settings"""
    import seaborn as sns
    sns.set_style('whitegrid')
    
    # Aggregate descriptor performance
    descriptor_stats = {}
    
    for result in all_results:
        if 'descriptor_results' in result:
            for desc_name, metrics in result['descriptor_results'].items():
                if desc_name not in descriptor_stats:
                    descriptor_stats[desc_name] = {'cv_r2': [], 'test_r2': []}
                descriptor_stats[desc_name]['cv_r2'].append(metrics.get('cv_mean_r2', 0))
                descriptor_stats[desc_name]['test_r2'].append(metrics.get('test_r2', 0))
    
    # Prepare data for all CHEMICAL_DESCRIPTORS from config.py
    cv_means = []
    test_means = []
    
    for desc_name in CHEMICAL_DESCRIPTORS:
        if desc_name in descriptor_stats:
            cv_means.append(np.mean(descriptor_stats[desc_name]['cv_r2']))
            test_means.append(np.mean(descriptor_stats[desc_name]['test_r2']))
        else:
            cv_means.append(0)
            test_means.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(30, 10))
    x = np.arange(len(CHEMICAL_DESCRIPTORS))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_means, width, label='CV R²', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, test_means, width, label='Test R²', alpha=0.8, edgecolor='black')
    
    # Color based on performance
    for bar, score in zip(bars1, cv_means):
        color = '#2ecc71' if score > 0.55 else '#3498db' if score > 0.50 else '#f39c12' if score > 0.45 else '#e74c3c'
        bar.set_color(color)
    
    for bar, score in zip(bars2, test_means):
        color = '#27ae60' if score > 0.55 else '#2980b9' if score > 0.50 else '#d68910' if score > 0.45 else '#a93226'
        bar.set_color(color)
    
    # Title showing config.py settings clearly
    title = f'49 Chemical Descriptors Performance\n'
    title += f'CODE_SPECIFIC_DATASETS[\'3\']: {CODE_DATASETS} | '
    title += f'ACTIVE_SPLIT_TYPES: {len(SPLIT_TYPES_TO_USE)} types | '
    title += f'CODE_SPECIFIC_FINGERPRINTS[\'3\']: {CODE_FINGERPRINTS}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlabel('Descriptor', fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}.{format_descriptor_name_for_display(d)[:10]}' for i, d in enumerate(CHEMICAL_DESCRIPTORS)],
                       rotation=90, ha='right', fontsize=9)
    
    ax.axhline(y=0.4445, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'descriptor_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: {plot_path}")

def generate_csv_summary():
    """Generate CSV summary of all results"""
    print("\n📊 GENERATING CSV SUMMARY...")

    all_results = []
    baseline_results = []

    for dataset in os.listdir(TARGET_PATH):
        dataset_path = os.path.join(TARGET_PATH, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split_type in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split_type)
            if not os.path.isdir(split_path):
                continue

            for fingerprint in os.listdir(split_path):
                fp_path = os.path.join(split_path, fingerprint)
                if not os.path.isdir(fp_path):
                    continue

                # Find JSON file
                json_files = [f for f in os.listdir(fp_path) if f.endswith('_results.json')]
                if not json_files:
                    continue

                json_path = os.path.join(fp_path, json_files[0])
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Handle both direct and nested JSON structure
                    if 'result' in data:
                        result_data = data['result']
                    else:
                        result_data = data

                    combination_id = f"{dataset}_{split_type}_{fingerprint}"

                    # Extract baseline data
                    if 'dnn_baseline' in result_data:
                        baseline = result_data['dnn_baseline']
                        baseline_results.append({
                            'combination_id': combination_id,
                            'dataset': dataset,
                            'split_type': split_type,
                            'fingerprint': fingerprint,
                            'baseline_cv_r2': baseline.get('cv_r2', 0),
                            'baseline_cv_std': baseline.get('cv_std', 0),
                            'baseline_test_r2': baseline.get('test_r2', 0),
                            'baseline_time': baseline.get('time', 0),
                            'n_train': result_data.get('n_train_molecules', 0),
                            'n_test': result_data.get('n_test_molecules', 0),
                            'completed_time': data.get('completed_time', ''),
                            'total_time': data.get('total_time', 0)
                        })

                    # Extract descriptor data
                    if 'descriptors' in result_data:
                        descriptors = result_data['descriptors']
                        for desc_name, desc_data in descriptors.items():
                            if desc_name in ['descriptor_array', '3d_conformers']:
                                continue

                            all_results.append({
                                'combination_id': combination_id,
                                'dataset': dataset,
                                'split_type': split_type,
                                'fingerprint': fingerprint,
                                'descriptor': desc_name,
                                'cv_mean_r2': desc_data.get('cv_mean_r2', 0),
                                'cv_std_r2': desc_data.get('cv_std_r2', 0),
                                'test_r2': desc_data.get('test_r2', 0),
                                'time_seconds': desc_data.get('time', 0),
                                'memory_mb': desc_data.get('memory_usage', 0),
                                'cpu_percent': desc_data.get('cpu_usage', 0),
                                'baseline_cv_r2': result_data.get('dnn_baseline', {}).get('cv_r2', 0),
                                'n_train': result_data.get('n_train_molecules', 0),
                                'n_test': result_data.get('n_test_molecules', 0)
                            })

                except Exception as e:
                    print(f"❌ Error reading {json_path}: {e}")

    # Save CSV files
    if all_results:
        df_descriptors = pd.DataFrame(all_results)
        csv_path = os.path.join(TARGET_PATH, 'module3_descriptor_results.csv')
        df_descriptors.to_csv(csv_path, index=False)
        print(f"✅ Descriptor results saved: {csv_path}")
        print(f"   Total records: {len(df_descriptors)}")

    if baseline_results:
        df_baseline = pd.DataFrame(baseline_results)
        csv_path = os.path.join(TARGET_PATH, 'module3_baseline_results.csv')
        df_baseline.to_csv(csv_path, index=False)
        print(f"✅ Baseline results saved: {csv_path}")
        print(f"   Total records: {len(df_baseline)}")

    # Summary statistics
    if all_results:
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Total descriptor evaluations: {len(df_descriptors)}")
        print(f"   Unique combinations: {df_descriptors['combination_id'].nunique()}")
        print(f"   Unique descriptors: {df_descriptors['descriptor'].nunique()}")
        print(f"   Failed descriptors (R² ≤ 0): {len(df_descriptors[df_descriptors['cv_mean_r2'] <= 0])}")

        # Top performers
        top_descriptors = df_descriptors.groupby('descriptor')['cv_mean_r2'].mean().sort_values(ascending=False).head(10)
        print(f"\n🏆 TOP 10 DESCRIPTORS (by average CV R²):")
        for desc, avg_r2 in top_descriptors.items():
            print(f"   {desc}: {avg_r2:.4f}")

def find_failed_descriptors():
    """Find all descriptors with R² ≤ 0 across all result files"""
    failed_descriptors = {}

    print("\n🔍 SCANNING FOR FAILED DESCRIPTORS (R² ≤ 0)...")
    print("="*60)

    for dataset in os.listdir(TARGET_PATH):
        dataset_path = os.path.join(TARGET_PATH, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split_type in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split_type)
            if not os.path.isdir(split_path):
                continue

            for fingerprint in os.listdir(split_path):
                fp_path = os.path.join(split_path, fingerprint)
                if not os.path.isdir(fp_path):
                    continue

                # Find JSON file
                json_files = [f for f in os.listdir(fp_path) if f.endswith('_results.json')]
                if not json_files:
                    continue

                json_path = os.path.join(fp_path, json_files[0])
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Handle both direct and nested JSON structure
                    if 'result' in data:
                        result_data = data['result']
                    else:
                        result_data = data

                    if 'descriptors' not in result_data:
                        continue

                    descriptors = result_data['descriptors']
                    combination_key = f"{dataset}/{split_type}/{fingerprint}"

                    # Check each descriptor for R² ≤ 0
                    for desc_name, desc_data in descriptors.items():
                        if desc_name in ['descriptor_array', '3d_conformers']:
                            continue

                        cv_mean_r2 = desc_data.get('cv_mean_r2', 0.0)
                        if cv_mean_r2 <= 0.0:
                            if combination_key not in failed_descriptors:
                                failed_descriptors[combination_key] = []
                            failed_descriptors[combination_key].append({
                                'name': desc_name,
                                'r2': cv_mean_r2,
                                'json_path': json_path
                            })

                except Exception as e:
                    print(f"❌ Error reading {json_path}: {e}")

    return failed_descriptors

def recompute_failed_descriptors(failed_descriptors):
    """Recompute only the failed descriptors"""
    print(f"\n🔄 RECOMPUTING {sum(len(v) for v in failed_descriptors.values())} FAILED DESCRIPTORS...")
    print("="*60)

    recomputed_count = 0
    for combination_key, failed_list in failed_descriptors.items():
        dataset, split_type, fingerprint = combination_key.split('/')

        print(f"\n📊 Processing: {combination_key}")
        print(f"   Failed descriptors: {len(failed_list)}")

        # Load data for this combination
        data = load_data(dataset, split_type)
        if not data:
            print(f"❌ Failed to load data for {combination_key}")
            continue

        # Load original JSON to update
        json_path = failed_list[0]['json_path']
        try:
            with open(json_path, 'r') as f:
                result_data = json.load(f)

            # Handle nested structure
            if 'result' in result_data:
                descriptors_dict = result_data['result']['descriptors']
            else:
                descriptors_dict = result_data['descriptors']

            # Recompute each failed descriptor
            for failed_desc in failed_list:
                desc_name = failed_desc['name']
                old_r2 = failed_desc['r2']

                print(f"   🔄 Recomputing: {desc_name} (was R² = {old_r2:.4f})")

                try:
                    # Directly recompute single descriptor using evaluate_descriptor_cv5
                    from extra_code.mol_fps_maker import get_fingerprints_combined
                    from rdkit import Chem

                    # Convert SMILES to mols
                    train_mols = [Chem.MolFromSmiles(smiles) for smiles in data['train_smiles']]
                    test_mols = [Chem.MolFromSmiles(smiles) for smiles in data['test_smiles']]

                    # Filter valid molecules
                    train_valid = [(mol, i) for i, mol in enumerate(train_mols) if mol is not None]
                    test_valid = [(mol, i) for i, mol in enumerate(test_mols) if mol is not None]

                    train_mols_valid = [m for m, _ in train_valid]
                    test_mols_valid = [m for m, _ in test_valid]

                    # Get fingerprints
                    train_fps = get_fingerprints_combined(
                        train_mols_valid, dataset, split_type, 'train',
                        fingerprint_type=fingerprint, module_name='3_solubility_feature_deeplearning'
                    )
                    test_fps = get_fingerprints_combined(
                        test_mols_valid, dataset, split_type, 'test',
                        fingerprint_type=fingerprint, module_name='3_solubility_feature_deeplearning'
                    )

                    # Load precomputed single descriptor
                    descriptor_dir = f"result/chemical_descriptors/{dataset}/{split_type}"
                    train_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_train_descriptors.npz"
                    test_desc_file = f"{descriptor_dir}/{dataset}_{split_type}_test_descriptors.npz"

                    if os.path.exists(train_desc_file) and os.path.exists(test_desc_file):
                        train_desc_data = np.load(train_desc_file, allow_pickle=True)
                        test_desc_data = np.load(test_desc_file, allow_pickle=True)

                        if desc_name in train_desc_data and desc_name in test_desc_data:
                            train_descriptor = train_desc_data[desc_name]
                            test_descriptor = test_desc_data[desc_name]

                            # Evaluate single descriptor using CV-5
                            try:
                                cv_mean_r2, cv_std_r2, test_r2, desc_time, mem_used, avg_cpu_usage = evaluate_descriptor_cv5(
                                    train_fps, train_descriptor, desc_name, data['train_y'],
                                    test_fps, test_descriptor, data['test_y'],
                                    dataset=dataset, split_type=split_type, fingerprint=fingerprint
                                )

                                # Update descriptor data
                                new_desc_data = {
                                    'cv_mean_r2': cv_mean_r2,
                                    'cv_std_r2': cv_std_r2,
                                    'test_r2': test_r2,
                                    'time': desc_time,
                                    'memory_usage': mem_used,
                                    'cpu_usage': avg_cpu_usage
                                }
                            except Exception as eval_error:
                                print(f"     ❌ Error evaluating {desc_name}: {eval_error}")
                                continue

                            descriptors_dict[desc_name] = new_desc_data

                            print(f"     ✅ Updated: {desc_name}: {old_r2:.4f} → {cv_mean_r2:.4f}")
                            recomputed_count += 1
                        else:
                            print(f"     ❌ Descriptor {desc_name} not found in precomputed data")
                    else:
                        print(f"     ❌ Precomputed descriptor files not found")

                except Exception as e:
                    print(f"     ❌ Error recomputing {desc_name}: {e}")

            # Save updated results
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"   💾 Updated: {json_path}")

        except Exception as e:
            print(f"❌ Error updating {json_path}: {e}")

    print(f"\n✅ RECOMPUTATION COMPLETE: {recomputed_count} descriptors updated")
    return recomputed_count

def generate_consolidated_plots():
    """Generate consolidated plots with 3 different sorting methods"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from config import CHEMICAL_DESCRIPTORS

    print("\n📊 GENERATING CONSOLIDATED PLOTS WITH 3 SORTING METHODS...")
    print("="*60)

    # Find the best performing combination for representative plots
    best_combinations = []

    for dataset in os.listdir(TARGET_PATH):
        dataset_path = os.path.join(TARGET_PATH, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split_type in os.listdir(dataset_path):
            split_path = os.path.join(dataset_path, split_type)
            if not os.path.isdir(split_path):
                continue

            for fingerprint in os.listdir(split_path):
                fp_path = os.path.join(split_path, fingerprint)
                if not os.path.isdir(fp_path):
                    continue

                json_files = [f for f in os.listdir(fp_path) if f.endswith('_results.json')]
                if not json_files:
                    continue

                json_path = os.path.join(fp_path, json_files[0])
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    if 'result' in data:
                        result_data = data['result']
                    else:
                        result_data = data

                    if 'descriptors' not in result_data:
                        continue

                    descriptors = result_data['descriptors']

                    # Calculate success rate
                    valid_descriptors = {k: v for k, v in descriptors.items()
                                       if k not in ['descriptor_array', '3d_conformers']}

                    if len(valid_descriptors) > 0:
                        successful = sum(1 for v in valid_descriptors.values()
                                       if v.get('cv_mean_r2', 0) > 0)
                        success_rate = successful / len(valid_descriptors)

                        best_combinations.append({
                            'dataset': dataset,
                            'split_type': split_type,
                            'fingerprint': fingerprint,
                            'success_rate': success_rate,
                            'successful_count': successful,
                            'total_count': len(valid_descriptors),
                            'descriptors': valid_descriptors,
                            'baseline': result_data.get('dnn_baseline', {})
                        })

                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    # Select best combination (highest success rate)
    if not best_combinations:
        print("❌ No valid combinations found")
        return

    best_combo = max(best_combinations, key=lambda x: x['success_rate'])
    print(f"📈 Using best combination: {best_combo['dataset']}/{best_combo['split_type']}/{best_combo['fingerprint']}")
    print(f"   Success rate: {best_combo['success_rate']:.1%} ({best_combo['successful_count']}/{best_combo['total_count']})")

    descriptors = best_combo['descriptors']

    # Prepare data for all 49 descriptors
    all_descriptor_data = {}
    for desc_name in CHEMICAL_DESCRIPTORS:
        if desc_name in descriptors:
            all_descriptor_data[desc_name] = descriptors[desc_name]
        else:
            # Missing descriptor - use zeros
            all_descriptor_data[desc_name] = {
                'cv_mean_r2': 0.0,
                'cv_std_r2': 0.0,
                'test_r2': 0.0
            }

    # Create plots directory
    plots_dir = os.path.join(TARGET_PATH, 'consolidated_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Generate 3 different plots
    plot_configs = [
        {
            'sort_key': lambda x: x[1]['cv_mean_r2'],
            'reverse': True,
            'title_suffix': 'Descending Order (Best to Worst)',
            'filename_suffix': 'descending'
        },
        {
            'sort_key': lambda x: x[1]['cv_mean_r2'],
            'reverse': False,
            'title_suffix': 'Ascending Order (Worst to Best)',
            'filename_suffix': 'ascending'
        },
        {
            'sort_key': lambda x: CHEMICAL_DESCRIPTORS.index(x[0]) if x[0] in CHEMICAL_DESCRIPTORS else 999,
            'reverse': False,
            'title_suffix': 'Original Order (MolWeight to GETAWAY)',
            'filename_suffix': 'original_order'
        }
    ]

    for config in plot_configs:
        # Sort descriptors
        sorted_desc = sorted(all_descriptor_data.items(),
                           key=config['sort_key'],
                           reverse=config['reverse'])

        names = [d[0] for d in sorted_desc]
        cv_scores = [d[1]['cv_mean_r2'] for d in sorted_desc]
        cv_stds = [d[1]['cv_std_r2'] for d in sorted_desc]
        test_scores = [d[1]['test_r2'] for d in sorted_desc]

        # Create plot
        fig_width = max(40, len(names) * 1.0)
        fig, ax = plt.subplots(figsize=(fig_width, 14))

        x = np.arange(len(names))
        bar_width = 0.45

        bars1 = ax.bar(x - bar_width/2, cv_scores, bar_width, label='CV R²', color='skyblue',
                      yerr=cv_stds, capsize=3, error_kw={'linewidth': 1, 'alpha': 0.8})
        bars2 = ax.bar(x + bar_width/2, test_scores, bar_width, label='Test R²', color='orange')

        # Add R² values inside bars
        for i, (bar1, bar2, cv_val, cv_std, test_val) in enumerate(zip(bars1, bars2, cv_scores, cv_stds, test_scores)):
            # CV R² value
            height1 = bar1.get_height()
            if height1 > 0.05:
                y_pos1 = height1 * 0.1
                ax.text(bar1.get_x() + bar1.get_width()/2., y_pos1,
                       f'{cv_val:.3f}±{cv_std:.3f}',
                       ha='center', va='bottom', fontsize=16, rotation=90,
                       color='black', fontweight='bold')

            # Test R² value
            height2 = bar2.get_height()
            if height2 > 0.05:
                y_pos2 = height2 * 0.1
                ax.text(bar2.get_x() + bar2.get_width()/2., y_pos2,
                       f'{test_val:.3f}',
                       ha='center', va='bottom', fontsize=16, rotation=90,
                       color='black', fontweight='bold')

        # Add baseline if available
        if best_combo['baseline']:
            baseline_cv = best_combo['baseline'].get('cv_r2', 0)
            baseline_test = best_combo['baseline'].get('test_r2', 0)
            if baseline_cv > 0:
                ax.axhline(baseline_cv, color='red', linestyle='--', linewidth=2,
                          label=f'Baseline CV: {baseline_cv:.3f}')
            if baseline_test > 0:
                ax.axhline(baseline_test, color='darkred', linestyle='--', linewidth=2,
                          label=f'Baseline Test: {baseline_test:.3f}')

        # Styling
        ax.set_xlabel('Chemical Descriptors', fontsize=18, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=18, fontweight='bold')
        ax.set_title(f'DNN Performance Enhancement - {config["title_suffix"]}\n'
                    f'{best_combo["dataset"].upper()} Dataset - {best_combo["split_type"].upper()} Split - '
                    f'{best_combo["fingerprint"].capitalize()} Fingerprint',
                    fontsize=20, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=14,
                          fontweight='bold', color='black')
        ax.legend(fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save plot
        plot_path = os.path.join(plots_dir, f'consolidated_plot_{config["filename_suffix"]}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"✅ Generated: {plot_path}")

    print(f"🎯 All 3 consolidated plots saved in: {plots_dir}")

def main():
    """Main function for solubility feature deep learning - Module 3"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Module 3: Solubility Feature Deep Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 3_solubility_feature_deeplearning.py                    # Normal execution
    python 3_solubility_feature_deeplearning.py --retry-failed     # Retry failed descriptors only
    python 3_solubility_feature_deeplearning.py --scan-only        # Only scan for failed descriptors
        """
    )
    parser.add_argument('--retry-failed', action='store_true',
                        help='Recompute descriptors with R² ≤ 0 only')
    parser.add_argument('--scan-only', action='store_true',
                        help='Only scan and report failed descriptors without recomputing')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to process (default: from CODE_SPECIFIC_DATASETS). Options: ws, de, lo, hu')

    args = parser.parse_args()

    # Set EPOCHS with proper priority
    global EPOCHS
    EPOCHS = get_epochs_for_module('3', args)

    # Set datasets with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    global CODE_DATASETS
    from config import CODE_SPECIFIC_DATASETS, ACTIVE_SPLIT_TYPES, DATA_PATH
    from pathlib import Path

    if args.datasets:
        CODE_DATASETS = args.datasets
        print(f"Datasets from argparse: {CODE_DATASETS}")
    elif '3' in CODE_SPECIFIC_DATASETS:
        CODE_DATASETS = CODE_SPECIFIC_DATASETS['3']
        print(f"Datasets from CODE_SPECIFIC_DATASETS: {CODE_DATASETS}")
    else:
        # Fallback: scan data directory
        CODE_DATASETS = []
        for split_type in ACTIVE_SPLIT_TYPES:
            split_dir = Path(DATA_PATH) / 'train' / split_type
            if split_dir.exists():
                for csv_file in split_dir.glob('*_train.csv'):
                    dataset = csv_file.stem.split('_')[1]  # Extract dataset from filename
                    if dataset not in CODE_DATASETS:
                        CODE_DATASETS.append(dataset)
        print(f"Datasets from data directory scan: {CODE_DATASETS}")

    # Print training method configuration
    print(f"\n{'='*80}")
    print(f"🧪 MODULE 3: SOLUBILITY FEATURE DEEP LEARNING")
    print(f"{'='*80}")
    from datetime import datetime as dt
    print(f"📅 Started: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Module Focus: Solubility Prediction with Deep Learning")

    if args.retry_failed:
        print(f"🔄 Mode: RETRY FAILED DESCRIPTORS ONLY")
    elif args.scan_only:
        print(f"🔍 Mode: SCAN FAILED DESCRIPTORS ONLY")
    else:
        print(f"▶️ Mode: NORMAL EXECUTION")

    print(f"{'='*80}")
    print(f"Training Method: {TRAINING_METHOD}")

    # Handle special modes
    if args.scan_only or args.retry_failed:
        failed_descriptors = find_failed_descriptors()

        if not failed_descriptors:
            print("\n✅ NO FAILED DESCRIPTORS FOUND!")
            print("All descriptors have R² > 0. No action needed.")
            return

        # Report findings
        total_failed = sum(len(v) for v in failed_descriptors.values())
        print(f"\n📊 FOUND {total_failed} FAILED DESCRIPTORS across {len(failed_descriptors)} combinations:")

        for combination_key, failed_list in failed_descriptors.items():
            print(f"\n   {combination_key}: {len(failed_list)} failed descriptors")
            for failed_desc in failed_list[:3]:  # Show first 3
                print(f"     - {failed_desc['name']}: R² = {failed_desc['r2']:.4f}")
            if len(failed_list) > 3:
                print(f"     ... and {len(failed_list) - 3} more")

        if args.scan_only:
            print(f"\n🔍 SCAN COMPLETE. Use --retry-failed to recompute these descriptors.")
            return

        # Confirm retry
        print(f"\n❓ Do you want to recompute these {total_failed} failed descriptors? (y/N): ", end='')
        try:
            confirm = input().strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Retry cancelled by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Retry cancelled by user.")
            return

        # Recompute failed descriptors
        recomputed_count = recompute_failed_descriptors(failed_descriptors)

        print(f"\n🎯 RETRY SUMMARY:")
        print(f"   Failed descriptors found: {total_failed}")
        print(f"   Successfully recomputed: {recomputed_count}")
        print(f"   Still failed: {total_failed - recomputed_count}")

        if recomputed_count > 0:
            print(f"\n🔄 Regenerating plots with updated results...")
            generate_comparison_plots()
            print("✅ Plots updated!")

        return
    if TRAINING_METHOD == 'subprocess':
        print(f"  - Using learning_process_pytorch_torchscript.py")
        print(f"  - Features: Memory isolation, TorchScript saving, model persistence")
    else:
        print(f"  - Using direct SimpleDNN training")
        print(f"  - Features: Fast execution, easy debugging, no file I/O overhead")
    print(f"{'='*80}\n")

    # Check renew setting from config
    from config import MODEL_CONFIG
    renew = MODEL_CONFIG.get('renew', False)
    print(f"⚙️  Renew setting: {renew} ({'Fresh start' if renew else 'Resume mode'})")

    # Set up logging to file
    from pathlib import Path
    from config import MODULE_NAMES
    from datetime import datetime
    import sys
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get module name from config
    module_name = MODULE_NAMES.get('3', '3_solubility_feature_deeplearning')

    # Create logs directory
    log_dir = Path(f"logs/{module_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module_name}_{timestamp}.log"

    # Create a custom logger class
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            self.log.write(f"{'='*60}\n")
            self.log.write(f"Module 3 (Solubility Feature DL) Execution Started: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n\n")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    # Redirect stdout to both terminal and file
    logger = Logger(log_file)
    sys.stdout = logger

    start_time = time.time()
    print("Starting Solubility Feature Deep Learning - Module 3...")

    # Memory tracking (with explicit import)
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")

    print(f"Active Datasets (Code 3): {CODE_DATASETS}")
    print(f"Active Fingerprints (Code 3): {CODE_FINGERPRINTS}")
    print(f"Split types: {SPLIT_TYPES_TO_USE}")

    # Use code-specific datasets
    datasets = CODE_DATASETS

    # Calculate total combinations
    total_combinations = len(SPLIT_TYPES_TO_USE) * len(datasets) * len(CODE_FINGERPRINTS)
    current_combination = 0
    start_time = time.time()
    combination_times = []

    # Check for existing completed work and load their times
    completed_combinations = []
    completed_times = []
    for split_type in SPLIT_TYPES_TO_USE:
        for dataset in datasets:
            for fingerprint in CODE_FINGERPRINTS:
                combination_id = f"{dataset}_{split_type}_{fingerprint}"
                result_file = os.path.join(TARGET_PATH, dataset, split_type, fingerprint, f"{combination_id}_results.json")
                if os.path.exists(result_file):
                    completed_combinations.append(combination_id)
                    # Load previous timing data if available
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            if 'combination_time' in data:
                                completed_times.append(data['combination_time'])
                            elif 'total_time' in data:
                                completed_times.append(data['total_time'])
                    except:
                        pass

    # Use historical data for initial ETA estimate
    if completed_times:
        combination_times = completed_times.copy()

    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    remaining_combinations = total_combinations - len(completed_combinations)

    print(f"\n🚀 STARTING PROCESSING")
    print(f"   Total combinations: {total_combinations}")
    print(f"   Already completed: {len(completed_combinations)}")
    print(f"   Remaining: {remaining_combinations}")
    print(f"   Datasets: {len(datasets)} ({datasets})")
    print(f"   Split types: {len(SPLIT_TYPES_TO_USE)} ({SPLIT_TYPES_TO_USE})")
    print(f"   Fingerprints: {len(CODE_FINGERPRINTS)} ({CODE_FINGERPRINTS})")

    # Show descriptor count per combination
    # Check if we have any existing descriptor files to get count
    total_descriptors = 49  # Default based on your previous logs
    try:
        # Try to get descriptor count from existing files
        descriptor_dir = f"result/chemical_descriptors/ws/rm"
        train_desc_file = f"{descriptor_dir}/ws_rm_train_descriptors.npz"
        if os.path.exists(train_desc_file):
            train_data = np.load(train_desc_file, allow_pickle=True)
            descriptor_names = [key for key in train_data.keys() if key not in ['train_mols', 'test_mols', 'descriptor_array', '3d_conformers']]
            total_descriptors = len(descriptor_names)
    except:
        pass
    print(f"   Descriptors per combination: {total_descriptors}")
    print(f"   Total evaluations: {total_combinations * total_descriptors:,}")

    # Show initial ETA estimate if we have historical data
    if combination_times and remaining_combinations > 0:
        avg_time = sum(combination_times) / len(combination_times)
        estimated_total_time = avg_time * remaining_combinations
        print(f"   Estimated time remaining: {format_time(estimated_total_time)} (avg: {format_time(avg_time)}/combo)")

    # Show system resource info
    import psutil
    mem_total = psutil.virtual_memory().total / (1024**3)  # GB
    mem_available = psutil.virtual_memory().available / (1024**3)  # GB
    cpu_count = psutil.cpu_count()
    print(f"   System resources: {cpu_count} CPUs, {mem_total:.1f}GB RAM ({mem_available:.1f}GB available)")

    if completed_combinations:
        print(f"\n📋 PREVIOUSLY COMPLETED:")
        for combo in completed_combinations[:5]:  # Show first 5
            print(f"   ✅ {combo}")
        if len(completed_combinations) > 5:
            print(f"   ... and {len(completed_combinations) - 5} more")

    print("="*80)

    for split_idx, split_type in enumerate(SPLIT_TYPES_TO_USE, 1):
        print(f"\n=== Processing split type: {split_type} ===")

        for dataset_idx, dataset in enumerate(datasets, 1):
            print(f"\nProcessing {get_dataset_display_name(dataset)} with {split_type} split...")

            for fingerprint_idx, fingerprint in enumerate(CODE_FINGERPRINTS, 1):
                current_combination += 1
                combination_id = f"{dataset}_{split_type}_{fingerprint}"

                # Check if combination already completed (check in subdirectory structure)
                result_file = os.path.join(TARGET_PATH, dataset, split_type, fingerprint, f"{combination_id}_results.json")
                if os.path.exists(result_file):
                    print(f"\n[{current_combination}/{total_combinations}] ✅ SKIPPING (already completed): {combination_id}")
                    continue

                print(f"\n[{current_combination}/{total_combinations}] 🔄 PROCESSING: {combination_id}")
                progress_percent = (current_combination / total_combinations) * 100

                # Calculate estimated time remaining
                if combination_times and current_combination > 1:
                    elapsed_time = time.time() - start_time
                    avg_time_per_combo = sum(combination_times) / len(combination_times)
                    remaining_combos = total_combinations - current_combination
                    estimated_remaining = avg_time_per_combo * remaining_combos
                    print(f"   Progress: {progress_percent:.1f}% | ETA: {format_time(estimated_remaining)} | Avg: {format_time(avg_time_per_combo)}/combo")
                else:
                    print(f"   Progress: {progress_percent:.1f}% | Calculating ETA...")

                combo_start_time = time.time()
                try:
                    # Load data first
                    data = load_data(dataset, split_type)
                    if not data:
                        print(f"✗ Failed to load data for {combination_id}")
                        continue
                    
                    # Prepare x_map and y_map for process_combination
                    x_map = {split_type: {
                        f"{dataset}_train": data['train_smiles'],
                        f"{dataset}_test": data['test_smiles']
                    }}
                    y_map = {split_type: {
                        f"{dataset}_train": data['train_y'],
                        f"{dataset}_test": data['test_y']
                    }}
                    
                    # Create combo_info for progress tracking
                    combo_info = {
                        'total_combinations': total_combinations,
                        'current_combination': current_combination,
                        'total_datasets': len(datasets),
                        'total_splits': len(SPLIT_TYPES_TO_USE),
                        'total_fingerprints': len(CODE_FINGERPRINTS),
                        'current_dataset_idx': dataset_idx,
                        'current_split_idx': split_idx,
                        'current_fingerprint_idx': fingerprint_idx
                    }

                    result = process_combination(dataset, split_type, fingerprint,
                                               x_map, y_map, None, False, combo_info=combo_info)

                    combo_elapsed = time.time() - combo_start_time
                    combination_times.append(combo_elapsed)

                    if result:
                        # Save result to file for resume capability
                        result_data = {
                            'combination_id': combination_id,
                            'dataset': dataset,
                            'split_type': split_type,
                            'fingerprint': fingerprint,
                            'completed_time': datetime.now().isoformat(),
                            'best_cv_r2': result.get('best_cv_r2', 0.0),
                            'total_time': result.get('total_time', 0.0),
                            'combination_time': combo_elapsed,
                            'result': result
                        }

                        # Ensure directory exists
                        os.makedirs(os.path.dirname(result_file), exist_ok=True)
                        with open(result_file, 'w') as f:
                            json.dump(result_data, f, indent=2)

                        # Create visualization plot after each fingerprint combination
                        try:
                            plot_path = create_fingerprint_combination_plot(
                                dataset=dataset,
                                split_type=split_type,
                                fingerprint=fingerprint,
                                results=result_data,
                                output_dir=TARGET_PATH
                            )
                        except Exception as plot_error:
                            print(f"    ⚠️  Failed to create plot: {plot_error}")

                        # Update ETA with latest completion
                        if len(combination_times) > 1:
                            avg_time = sum(combination_times) / len(combination_times)
                            remaining_combos = total_combinations - current_combination
                            eta = avg_time * remaining_combos

                            print(f"✅ [{current_combination}/{total_combinations}] COMPLETED: {combination_id}")
                            best_r2 = result.get('best_cv_r2', 0.0)
                            best_std = result.get('best_cv_std', 0.0)
                            print(f"   Best CV R²: {best_r2:.4f} ± {best_std:.4f} | Time: {format_time(combo_elapsed)}")
                            print(f"   Progress: {progress_percent:.1f}% | Updated ETA: {format_time(eta)}")
                        else:
                            print(f"✅ [{current_combination}/{total_combinations}] COMPLETED: {combination_id}")
                            best_r2 = result.get('best_cv_r2', 0.0)
                            best_std = result.get('best_cv_std', 0.0)
                            print(f"   Best CV R²: {best_r2:.4f} ± {best_std:.4f} | Time: {format_time(combo_elapsed)}")
                    else:
                        print(f"❌ [{current_combination}/{total_combinations}] FAILED: {combination_id}")

                except Exception as e:
                    combo_elapsed = time.time() - combo_start_time
                    combination_times.append(combo_elapsed)
                    print(f"❌ [{current_combination}/{total_combinations}] ERROR: {combination_id} - {e}")
                    continue
    
    # Final summary - count result files in subdirectories
    final_completed = 0
    for root, dirs, files in os.walk(TARGET_PATH):
        final_completed += len([f for f in files if f.endswith('_results.json')])

    print(f"\n🎉 MODULE 3 COMPLETED!")
    print(f"   Total combinations processed: {final_completed}/{total_combinations}")
    print(f"   Success rate: {(final_completed/total_combinations)*100:.1f}%")
    print("="*80)

    # Generate comparison plots for all results
    print("\nGenerating comparison plots...")
    generate_comparison_plots()
    print("Plots generation completed!")

    # Generate consolidated plots with 3 different sorting options
    print("\nGenerating consolidated plots with 3 sorting options...")
    generate_consolidated_plots()
    print("Consolidated plots generation completed!")

    # Generate CSV summary
    generate_csv_summary()

    # Close log file
    logger.log.write(f"\n{'='*60}\n")
    logger.log.write(f"Module 3 (Solubility Feature DL) Execution Finished: {datetime.now()}\n")
    logger.log.write(f"{'='*60}\n")
    sys.stdout = logger.terminal
    logger.close()


def get_split_types_display():
    """Get display string for split types"""
    return f"ACTIVE_SPLIT_TYPES: {SPLIT_TYPES_TO_USE}"

if __name__ == "__main__":
    print("Starting main() function...")
    main_start_time = time.time()
    try:
        main()
    finally:
        # Calculate and display total execution time
        total_execution_time = time.time() - main_start_time
        hours = int(total_execution_time // 3600)
        minutes = int((total_execution_time % 3600) // 60)
        seconds = total_execution_time % 60

        print("\n" + "="*80)
        print("🎯 MODULE 3 EXECUTION SUMMARY")
        print("="*80)
        print(f"⏱️  Total execution time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        print(f"📊 Total execution time: {total_execution_time:.2f} seconds")

        # Final memory usage
        try:
            import psutil
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"🧠 Final memory usage: {final_memory:.1f} MB")
        except:
            pass

        print("="*80)
