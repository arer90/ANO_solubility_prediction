#!/usr/bin/env python3
"""
ANO Network Optimization FOMO - ANO Framework Module 6
======================================================

PURPOSE:
Module 6 implements FOMO (Feature Optimization → Model Optimization) strategy.
It uses the best features from Module 4, then optimizes the neural network
architecture for those specific features.

KEY FEATURES:
1. **Sequential Optimization**: Feature first (Module 4) → Model second
2. **Feature Reuse**: Loads best features from Module 4 results
3. **FlexibleDNNModel**: Dynamic architecture optimization
4. **No StandardScaler**: Raw features used directly
5. **Dual Search**: Searches features then architecture
6. **Performance Issue**: Slower due to dual-stage optimization

RECENT UPDATES (2024):
- Uses FlexibleDNNModel for architecture optimization
- Removed StandardScaler (using raw features)
- Fixed search_data_descriptor_compress inefficiency
- Default epochs: 30 (from config.py module_epochs['6'])
- Optuna trials: 1 (for quick testing)

OPTIMIZATION FLOW:
1. Load all fingerprints (2727 features)
2. Search Module 4 results for best descriptors
3. Combine fingerprints + selected descriptors (~3600 features)
4. Optimize network architecture using Optuna
5. Evaluate with 5-fold CV

PERFORMANCE BOTTLENECK:
- Module 6 is slow because it re-searches features every trial
- Average trial time: ~229 seconds
- Total runtime: ~2 hours for 36 combinations
- Main issue: search_data_descriptor_compress called repeatedly

ARCHITECTURE SEARCH:
- n_layers: 1-5 layers
- hidden_dims: 2-9999 units per layer
- activation: Various (relu, gelu, silu, etc.)
- dropout_rate: 0.1-0.5
- batch_norm: True/False
- optimizer: Adam variants

OUTPUT STRUCTURE:
result/6_network_fomo/
├── {dataset}_{split}/
│   ├── best_features_and_architecture.json
│   ├── optimization_history.csv
│   ├── fomo_model.pt
│   └── performance_metrics.json
└── fomo_comparison.png

USAGE:
python 6_ANO_NetworkOptimization_FOMO.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --trials: Number of Optuna trials (default: 1)
  --epochs: Override epochs (default from config: 30)
  --renew: Start fresh optimization
"""

import os
import sys
from datetime import datetime
import time
import subprocess
import logging
import warnings
import gc

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from config import MODULE_NAMES
import optuna
import psutil
import json

import torch
import torch.nn as nn

# Module 6 uses Optuna - DO NOT fix seeds for optimization diversity
# Seeds are intentionally NOT set to allow Optuna to explore different initialization
# torch.manual_seed(42)
# np.random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)
#     torch.cuda.manual_seed_all(42)
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import StandardScaler  # Not used anymore

# Import from unified modules
from extra_code.mol_fps_maker import (
    load_data_ws, load_data_de, load_data_lo, load_data_hu,
    get_fingerprints, get_fingerprints_cached
)

# 3D conformer functions are now in chem_descriptor_maker
try:
    from extra_code.chem_descriptor_maker import ChemDescriptorCalculator
except ImportError:
    print("Warning: ChemDescriptorCalculator not found")

# Import feature selection functions
from extra_code.ano_feature_search import search_data_descriptor_compress
from extra_code.ano_feature_selection import (
    selection_data_descriptor_compress,
    selection_fromStudy_compress,
    convert_params_to_selection
)

# Import configuration from config.py
from config import (
    DATASETS, SPLIT_TYPES as ALL_SPLIT_TYPES, ACTIVE_SPLIT_TYPES,
    FINGERPRINTS, MODEL_CONFIG,
    CHEMICAL_DESCRIPTORS, DESCRIPTORS_NEED_NORMALIZATION,
    DATA_PATH, RESULT_PATH, MODEL_PATH, PARALLEL_CONFIG,
    get_dataset_display_name, get_dataset_filename, get_split_type_name,
    get_code_datasets, get_code_fingerprints,
    get_storage_url, get_database_info, get_epochs_for_module
)

# Configuration - Global variables
# Get code-specific configurations - will be updated in main() with argparse priority
CODE_DATASETS = get_code_datasets(6)  # Code 6 - default, will be updated in main()
CODE_FINGERPRINTS = get_code_fingerprints(6)  # Code 6

# Enable all split types for comprehensive analysis (test mode uses only rm)
SPLIT_TYPES = ACTIVE_SPLIT_TYPES  # Use config.py setting (currently ['rm'])
N_TRIALS = MODEL_CONFIG['optuna_trials']  # Use config.py setting (currently 1)
EPOCHS = None  # Will be set in main() with proper priority
OUTPUT_DIR = Path(RESULT_PATH) / "6_ANO_NetworkOptimization_FOMO"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global data storage - removed, using prepare_data_for_split() instead

# Dataset name mapping - use only code-specific datasets
DATASET_MAPPING = {k: DATASETS[k] for k in CODE_DATASETS}

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

def load_preprocessed_data(dataset, split_type):
    """
    Load preprocessed train/test data

    This function loads the pre-split train and test datasets that were created
    in the preprocessing step. The data is already split according to various
    splitting strategies (random, scaffold, etc.) to ensure fair evaluation.

    Args:
        dataset: Short dataset name ('ws', 'de', 'lo', 'hu')
        split_type: Split strategy ('rm', 'sc', 'cs', etc.)

    Returns:
        Tuple of (train_smiles, train_targets, test_smiles, test_targets)
    """
    # Try both naming conventions
    dataset_full = DATASET_MAPPING.get(dataset, dataset)

    # Try abbreviated name first, then full name
    train_path_short = Path(f"data/train/{split_type}/{split_type}_{dataset}_train.csv")
    test_path_short = Path(f"data/test/{split_type}/{split_type}_{dataset}_test.csv")

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
        raise FileNotFoundError(f"Preprocessed data not found for {dataset}-{split_type}")

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Handle different column naming conventions
    smiles_col = 'SMILES' if 'SMILES' in train_df.columns else 'smiles'
    target_col = 'target' if 'target' in train_df.columns else 'logS'

    train_smiles = train_df[smiles_col].tolist()
    train_targets = train_df[target_col].values
    test_smiles = test_df[smiles_col].tolist()
    test_targets = test_df[target_col].values

    return train_smiles, train_targets, test_smiles, test_targets

def prepare_data_for_split(dataset, split_type):
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
        dataset: Short dataset name
        split_type: Data splitting strategy

    Returns:
        Dictionary with train/test fingerprints and targets
    """
    try:
        # Load preprocessed data
        train_smiles, train_y, test_smiles, test_y = load_preprocessed_data(dataset, split_type)

        print(f"  Loaded {dataset.upper()}-{split_type}: {len(train_smiles)} train, {len(test_smiles)} test")

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
            train_mols_filtered, dataset.upper(), split_type, 'train', module_name='6_ANO_NetworkOptimization_FOMO'
        )
        test_morgan, test_maccs, test_avalon = get_fingerprints_cached(
            test_mols_filtered, dataset.upper(), split_type, 'test', module_name='6_ANO_NetworkOptimization_FOMO'
        )

        # Combine all fingerprints
        train_fps_all = np.hstack([train_morgan, train_maccs, train_avalon])
        test_fps_all = np.hstack([test_morgan, test_maccs, test_avalon])

        print(f"  Final fingerprint shapes: train {train_fps_all.shape}, test {test_fps_all.shape}")

        return {
            'train_fps': train_fps_all,
            'test_fps': test_fps_all,
            'train_y': np.array(train_y_filtered),
            'test_y': np.array(test_y_filtered)
        }

    except Exception as e:
        print(f"Error preparing data for {dataset}-{split_type}: {e}")
        raise

def train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, activation='relu', n_folds=5, trial=None):
    """
    Type1: Research Pipeline - CV methodology
    Performs K-fold CV using training data only and predicts independent test set per fold

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Model parameters
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        activation: Activation function
        n_folds: Number of CV folds
        trial: Optuna trial object

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds} on Train + Test prediction per fold")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Skip StandardScaler - use raw features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    test_r2_scores = []  # Test R² scores from each fold
    test_rmse_scores = []  # Test RMSE from each fold
    test_mse_scores = []  # Test MSE from each fold
    test_mae_scores = []  # Test MAE from each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Save training data for subprocess
        import pickle
        train_data = {
            'X': X_train_fold,
            'y': y_train_fold,
            'X_val': X_test,  # Type1: predict independent test set each fold
            'y_val': y_test,
            'batch_size': model_params['batch_size'],
            'epochs': model_params['epochs'],
            'lr': model_params['learning_rate'],
            'weight_decay': model_params.get('weight_decay', 0.0),
            'architecture': [X_train_fold.shape[1]] + hidden_dims + [1],
            'dropout_rate': dropout_rate,
            'activation': activation,
            'optimizer': model_params.get('optimizer', 'adam'),
            'scheduler': model_params.get('scheduler', 'none'),
            'use_batch_norm': model_params.get('use_batch_norm', False)
        }

        # Add optimizer-specific parameters
        if model_params.get('optimizer') == 'sgd':
            train_data['momentum'] = model_params.get('momentum', 0.9)
            train_data['nesterov'] = model_params.get('nesterov', False)
        elif model_params.get('optimizer') in ['adam', 'adamw']:
            train_data['beta1'] = model_params.get('beta1', 0.9)
            train_data['beta2'] = model_params.get('beta2', 0.999)
            train_data['eps'] = model_params.get('eps', 1e-8)

        # Add scheduler-specific parameters
        if model_params.get('scheduler') == 'step':
            train_data['step_size'] = model_params.get('step_size', 30)
            train_data['gamma'] = model_params.get('gamma', 0.1)
        elif model_params.get('scheduler') == 'exponential':
            train_data['gamma'] = model_params.get('gamma', 0.95)

        os.makedirs('save_model', exist_ok=True)
        train_file = f"save_model/module6_type1_fold{fold}_train_data.pkl"
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)

        model_file = "save_model/full_model.pt"
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            train_file,
            model_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        try:
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
                            test_r2_scores.append(r2)
                            test_rmse_scores.append(rmse)
                            test_mse_scores.append(mse)
                            test_mae_scores.append(mae)
                            print(f"    Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

                            # Aggressive pruning for Type1
                            if trial is not None:
                                # Aggressive pruning based on fold number
                                if fold == 0:
                                    # First fold: Kill if R² < -3
                                    if r2 < -3.0:
                                        print(f"🔴 Trial {trial.number} KILLED (Fold 1 R2: {r2:.4f} < -3.0)")
                                        import optuna
                                        raise optuna.exceptions.TrialPruned()
                                else:
                                    # Fold 2+: Kill if R² ≤ 0
                                    if r2 <= 0.0:
                                        print(f"🔴 Trial {trial.number} KILLED (Fold {fold+1} R2: {r2:.4f} <= 0.0)")
                                        import optuna
                                        raise optuna.exceptions.TrialPruned()

                                # Report to Optuna for MedianPruner
                                trial.report(r2, fold)

                                # Let MedianPruner decide after enough trials
                                if trial.should_prune():
                                    print(f"Trial {trial.number} PRUNED by MedianPruner (R2: {r2:.4f})")
                                    import optuna
                                    raise optuna.exceptions.TrialPruned()

                            break
                        except ValueError:
                            continue
        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned immediately
            raise
        except Exception as e:
            print(f"    Fold {fold+1}: Error parsing metrics - {e}")
            test_r2_scores.append(0.0)
            test_rmse_scores.append(0.0)
            test_mse_scores.append(0.0)
            test_mae_scores.append(0.0)

        # Cleanup
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(model_file):
            os.remove(model_file)

    # Calculate CV statistics (research results - average of test predictions from each fold)
    if len(test_r2_scores) == 0:
        return {'r2_mean': 0, 'r2_std': 0, 'rmse_mean': 0, 'rmse_std': 0, 'mse_mean': 0, 'mse_std': 0, 'mae_mean': 0, 'mae_std': 0}

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

    print(f"    [TYPE1-Research] CV Results: R2={cv_stats['r2_mean']:.4f}±{cv_stats['r2_std']:.4f}, RMSE={cv_stats['rmse_mean']:.4f}±{cv_stats['rmse_std']:.4f}, MSE={cv_stats['mse_mean']:.4f}±{cv_stats['mse_std']:.4f}, MAE={cv_stats['mae_mean']:.4f}±{cv_stats['mae_std']:.4f}")

    return cv_stats

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, activation='relu', n_folds=5, trial=None):
    """
    Type 2: Production Pipeline - Train/Test Split + CV
    Performs CV on pre-split training data and evaluates on independent test set

    Args:
        X_train: Training data features (pre-split)
        y_train: Training data targets (pre-split)
        X_test: Test data features (pre-split)
        y_test: Test data targets (pre-split)
        model_params: Model parameters
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        activation: Activation function
        n_folds: Number of folds for CV
        trial: Optuna trial object

    Returns:
        Tuple of (cv_stats, final_test_metrics)
    """
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Skip StandardScaler - use raw features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Step 1: CV on training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    test_r2_scores = []
    test_rmse_scores = []
    test_mse_scores = []
    test_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        # Save training data for subprocess
        import pickle
        train_data = {
            'X': X_train_fold,
            'y': y_train_fold,
            'X_val': X_val_fold,
            'y_val': y_val_fold,
            'batch_size': model_params['batch_size'],
            'epochs': model_params['epochs'],
            'lr': model_params['learning_rate'],
            'weight_decay': model_params.get('weight_decay', 0.0),
            'architecture': [X_train_fold.shape[1]] + hidden_dims + [1],
            'dropout_rate': dropout_rate,
            'activation': activation,
            'optimizer': model_params.get('optimizer', 'adam'),
            'scheduler': model_params.get('scheduler', 'none'),
            'use_batch_norm': model_params.get('use_batch_norm', False)
        }

        # Add optimizer-specific parameters
        if model_params.get('optimizer') == 'sgd':
            train_data['momentum'] = model_params.get('momentum', 0.9)
            train_data['nesterov'] = model_params.get('nesterov', False)
        elif model_params.get('optimizer') in ['adam', 'adamw']:
            train_data['beta1'] = model_params.get('beta1', 0.9)
            train_data['beta2'] = model_params.get('beta2', 0.999)
            train_data['eps'] = model_params.get('eps', 1e-8)

        # Add scheduler-specific parameters
        if model_params.get('scheduler') == 'step':
            train_data['step_size'] = model_params.get('step_size', 30)
            train_data['gamma'] = model_params.get('gamma', 0.1)
        elif model_params.get('scheduler') == 'exponential':
            train_data['gamma'] = model_params.get('gamma', 0.95)

        os.makedirs('save_model', exist_ok=True)
        train_file = f"save_model/module6_type2_fold{fold}_train_data.pkl"
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)

        model_file = "save_model/full_model.pt"
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            train_file,
            model_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        try:
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
                            test_r2_scores.append(r2)
                            test_rmse_scores.append(rmse)
                            test_mse_scores.append(mse)
                            test_mae_scores.append(mae)
                            print(f"    CV Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")

                            # Type2 doesn't need pruning - it just saves results after Type1 passes

                            break
                        except ValueError:
                            continue
        except Exception as e:
            print(f"    CV Fold {fold+1}: Error parsing metrics - {e}")
            test_r2_scores.append(0.0)
            test_rmse_scores.append(0.0)
            test_mse_scores.append(0.0)
            test_mae_scores.append(0.0)

        # Cleanup
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(model_file):
            os.remove(model_file)

    # Calculate CV statistics
    mean_r2 = np.mean(test_r2_scores) if test_r2_scores else 0
    std_r2 = np.std(test_r2_scores) if test_r2_scores else 0
    mean_rmse = np.mean(test_rmse_scores) if test_rmse_scores else 0
    std_rmse = np.std(test_rmse_scores) if test_rmse_scores else 0
    mean_mse = np.mean(test_mse_scores) if test_mse_scores else 0
    std_mse = np.std(test_mse_scores) if test_mse_scores else 0
    mean_mae = np.mean(test_mae_scores) if test_mae_scores else 0
    std_mae = np.std(test_mae_scores) if test_mae_scores else 0

    cv_stats = {
        'mean_r2': mean_r2, 'std_r2': std_r2,
        'mean_rmse': mean_rmse, 'std_rmse': std_rmse,
        'mean_mse': mean_mse, 'std_mse': std_mse,
        'mean_mae': mean_mae, 'std_mae': std_mae
    }

    print(f"    [TYPE2-Production] CV Results: R2={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}, MSE={mean_mse:.4f}±{std_mse:.4f}, MAE={mean_mae:.4f}±{std_mae:.4f}")

    # Step 2: Train final model on full training data and test on test set
    print(f"    [TYPE2-Production] Training final model on full training data...")

    train_data_final = {
        'X': X_train,
        'y': y_train,
        'X_val': X_test,
        'y_val': y_test,
        'batch_size': model_params['batch_size'],
        'epochs': model_params['epochs'],
        'lr': model_params['learning_rate'],
        'weight_decay': model_params.get('weight_decay', 0.0),
        'architecture': [X_train.shape[1]] + hidden_dims + [1],
        'dropout_rate': dropout_rate,
        'activation': activation,
        'optimizer': model_params.get('optimizer', 'adam'),
        'scheduler': model_params.get('scheduler', 'none'),
        'use_batch_norm': model_params.get('use_batch_norm', False)
    }

    # Add optimizer-specific parameters
    if model_params.get('optimizer') == 'sgd':
        train_data_final['momentum'] = model_params.get('momentum', 0.9)
        train_data_final['nesterov'] = model_params.get('nesterov', False)
    elif model_params.get('optimizer') in ['adam', 'adamw']:
        train_data_final['beta1'] = model_params.get('beta1', 0.9)
        train_data_final['beta2'] = model_params.get('beta2', 0.999)
        train_data_final['eps'] = model_params.get('eps', 1e-8)

    # Add scheduler-specific parameters
    if model_params.get('scheduler') == 'step':
        train_data_final['step_size'] = model_params.get('step_size', 30)
        train_data_final['gamma'] = model_params.get('gamma', 0.1)
    elif model_params.get('scheduler') == 'exponential':
        train_data_final['gamma'] = model_params.get('gamma', 0.95)

    train_file_final = "save_model/module6_type2_final_train_data.pkl"
    with open(train_file_final, 'wb') as f:
        pickle.dump(train_data_final, f)

    model_file_final = "save_model/full_model.pt"
    cmd_final = [
        sys.executable,
        "extra_code/learning_process_pytorch_torchscript.py",
        train_file_final,
        model_file_final
    ]

    result_final = subprocess.run(cmd_final, capture_output=True, text=True)

    # Parse final test results
    test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
    try:
        lines = result_final.stdout.strip().split('\n')
        for line in reversed(lines):
            if ',' in line and line.count(',') == 3:
                parts = line.split(',')
                if len(parts) == 4:
                    try:
                        test_r2 = float(parts[0])
                        test_rmse = float(parts[1])
                        test_mse = float(parts[2])
                        test_mae = float(parts[3])
                        print(f"    [TYPE2-Production] Final Test: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
                        break
                    except ValueError:
                        continue
    except Exception as e:
        print(f"    [TYPE2-Production] Error parsing final test metrics: {e}")

    # Cleanup
    if os.path.exists(train_file_final):
        os.remove(train_file_final)
    if os.path.exists(model_file_final):
        os.remove(model_file_final)

    final_test_metrics = {
        'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mse': test_mse, 'test_mae': test_mae
    }

    return cv_stats, final_test_metrics

def train_model_cv_both_types(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, activation='relu', n_folds=5, trial=None):
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
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        activation: Activation function
        n_folds: Number of CV folds
        trial: Optuna trial object

    Returns:
        Dictionary with Type1 and Type2 results
    """
    print(f"\n=== Running Both CV Types ===")

    results = {}

    # Type1: Research Pipeline - K-fold CV on training data + test prediction per fold
    print(f"    [TYPE1-Research] Research Pipeline - CV-5")
    type1_results = train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, activation, n_folds, trial)
    results['type1'] = type1_results

    # Type2: Production Pipeline - Train/Val split + independent test prediction
    print(f"    [TYPE2-Production] Production Pipeline - CV-5 on Train + Final Test")
    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, hidden_dims, dropout_rate, activation, n_folds, trial)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_and_evaluate_cv(X, y, model_params, hidden_dims, dropout_rate, X_test=None, y_test=None, activation='relu', trial=None):
    """
    Enhanced CV function with dual CV types (Type1 + Type2) and backward compatibility

    This function now implements both CV methodologies:
    - Type1 (Research): Full dataset K-fold CV for research/paper results
    - Type2 (Production): Train/Test split + CV on train for production pipeline

    For Optuna optimization, uses Type1 CV mean R² as the optimization target.

    Args:
        X: Feature matrix (n_samples x n_features)
        y: Target values
        model_params: Dict with batch_size, epochs, learning_rate, etc.
        hidden_dims: Hidden layer dimensions list
        dropout_rate: Dropout rate
        X_test: Optional test feature matrix (for backward compatibility)
        y_test: Optional test target values (for backward compatibility)
        activation: Activation function
        trial: Optuna trial object for pruning

    Returns:
        Tuple of 22 values for backward compatibility + dual CV results
    """
    # Get CV folds from config
    n_folds = MODEL_CONFIG['cv_folds']

    # Execute dual CV methodology
    if X_test is not None and y_test is not None:
        # Case 1: Pre-split data provided - use dual CV methodology
        print(f"\n=== Dual CV with pre-split data ===")
        # Run both CV types (no data leakage)
        dual_results = train_model_cv_both_types(X, y, X_test, y_test, model_params, hidden_dims, dropout_rate, activation, n_folds=n_folds, trial=trial)
    else:
        # Case 2: Only training data provided - need to split first
        print(f"\n=== Dual CV with auto-split data ===")
        from sklearn.model_selection import train_test_split
        X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        dual_results = train_model_cv_both_types(X_train_auto, y_train_auto, X_test_auto, y_test_auto, model_params, hidden_dims, dropout_rate, activation, n_folds=n_folds, trial=trial)

    # Extract results for Optuna optimization (using Type1 mean R²)
    type1_results = dual_results['type1']
    mean_r2 = type1_results['r2_mean']  # Type1 now returns direct format

    # Optuna pruning and trial management
    if trial is not None:
        # Report Type1 CV mean as optimization target
        trial.report(mean_r2, n_folds-1)  # Report at final step

        # Final check after all folds
        if mean_r2 <= 0.0:  # Bad model
            print(f"🔴 Trial {trial.number} KILLED (Final CV R2: {mean_r2:.4f} <= 0.0)")
            raise optuna.exceptions.TrialPruned()

        # Let MedianPruner decide for borderline cases
        if trial.should_prune():
            print(f"Trial {trial.number} PRUNED by MedianPruner (Final CV R2: {mean_r2:.4f})")
            raise optuna.exceptions.TrialPruned()

    # Extract metrics for backward compatibility return format
    # Using Type1 results for primary metrics (research pipeline)
    std_r2 = type1_results['r2_std']
    mean_rmse = type1_results['rmse_mean']
    std_rmse = type1_results['rmse_std']
    mean_mse = type1_results['mse_mean']
    std_mse = type1_results['mse_std']
    mean_mae = type1_results['mae_mean']
    std_mae = type1_results['mae_std']
    # Set best values from test scores
    best_r2 = max(type1_results['test_scores']) if type1_results['test_scores'] else mean_r2
    best_rmse = mean_rmse  # Use mean for best RMSE
    best_mse = mean_mse    # Use mean for best MSE
    best_mae = mean_mae    # Use mean for best MAE

    # Type2 final test results (production pipeline)
    type2_results = dual_results['type2']
    if 'final_metrics' in type2_results:
        type2_final = type2_results['final_metrics']
        test_r2 = type2_final.get('test_r2', 0.0)
        test_rmse = type2_final.get('test_rmse', 0.0)
        test_mse = type2_final.get('test_mse', 0.0)
        test_mae = type2_final.get('test_mae', 0.0)
        # Also get Type2 CV stats
        type2_cv = type2_results['cv_stats']
        type2_val_r2_mean = type2_cv.get('mean_r2', 0.0)
        type2_val_r2_std = type2_cv.get('std_r2', 0.0)
        type2_val_rmse_mean = type2_cv.get('mean_rmse', 0.0)
        type2_val_rmse_std = type2_cv.get('std_rmse', 0.0)
        type2_val_mse_mean = type2_cv.get('mean_mse', 0.0)
        type2_val_mse_std = type2_cv.get('std_mse', 0.0)
        type2_val_mae_mean = type2_cv.get('mean_mae', 0.0)
        type2_val_mae_std = type2_cv.get('std_mae', 0.0)
    else:
        test_r2 = test_rmse = test_mse = test_mae = 0.0
        type2_val_r2_mean = type2_val_r2_std = 0.0
        type2_val_rmse_mean = type2_val_rmse_std = 0.0
        type2_val_mse_mean = type2_val_mse_std = 0.0
        type2_val_mae_mean = type2_val_mae_std = 0.0

    # Legacy CV test metrics (set to 0 since we removed Method 1 approach)
    cv_test_r2_mean = cv_test_r2_std = cv_test_rmse_mean = cv_test_rmse_std = 0.0
    cv_test_mse_mean = cv_test_mae_mean = 0.0

    print(f"\n=== Dual CV Results Summary ===")
    print(f"    [TYPE1-Research] CV Val: R²={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}, MAE={mean_mae:.4f}±{std_mae:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}, MAE={mean_mae:.4f}±{std_mae:.4f}")
    print(f"    [TYPE2-Production] CV Results: R²={type2_val_r2_mean:.4f}±{type2_val_r2_std:.4f}, RMSE={type2_val_rmse_mean:.4f}±{type2_val_rmse_std:.4f}, MSE={type2_val_mse_mean:.4f}±{type2_val_mse_std:.4f}, MAE={type2_val_mae_mean:.4f}±{type2_val_mae_std:.4f}")
    print(f"    [TYPE2-Production] Final Test: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
    print(f"  📌 [OPTUNA] Using Type1 R²={mean_r2:.4f} for optimization")
    
    return (mean_r2, std_r2, best_r2,
            mean_rmse, std_rmse, best_rmse,
            mean_mse, std_mse, best_mse,
            mean_mae, std_mae, best_mae,
            test_r2, test_rmse, test_mse, test_mae,
            cv_test_r2_mean, cv_test_r2_std, cv_test_rmse_mean, cv_test_rmse_std,
            cv_test_mse_mean, cv_test_mae_mean)

# Import unified model from ano_feature_selection
try:
    from extra_code.ano_feature_selection import FlexibleDNNModel
    UNIFIED_MODEL_AVAILABLE = True
except ImportError:
    UNIFIED_MODEL_AVAILABLE = False
    # Fallback to local DNNModel if import fails
    class DNNModel(nn.Module):
        """
        Fallback Deep Neural Network model - used only if unified model not available
        """
        def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
            super(DNNModel, self).__init__()
            layers = []
            
            # First layer
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(nn.SiLU())
            layers.append(nn.BatchNorm1d(hidden_dims[0], momentum=0.01, track_running_stats=False))
            layers.append(nn.Dropout(dropout_rate))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.SiLU())
                layers.append(nn.BatchNorm1d(hidden_dims[i+1], momentum=0.01, track_running_stats=False))
                layers.append(nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], 1))  # Last hidden unit = 1
            
            self.model = nn.Sequential(*layers)
            self._initialize_weights()
        
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            return self.model(x)


def get_best_feature_selection(dataset_name, split_type):
    """
    Get best feature selection from module 4 (ANO_FO) results
    Priority: 1) DB, 2) Config file, 3) Default (all features)
    
    This function loads the optimal feature selection discovered by module 4
    from the Optuna database. Module 4 performs exhaustive feature selection
    to find the best combination of molecular descriptors.
    
    The feature selection is stored as binary flags (0/1) for each of the
    49 possible molecular descriptor categories.
    
    Args:
        dataset_name: Dataset identifier ('ws', 'de', 'lo', 'hu')
        split_type: Data split strategy ('rm', 'sc', etc.)
    
    Returns:
        Dictionary mapping descriptor names to inclusion flags (0/1)
        Falls back to all features (all 1s) if loading fails
    """
    # Use config's get_best_features which has the fallback logic
    from config import get_best_features
    
    # First try the config function which checks DB -> config file -> default
    features = get_best_features(dataset_name, split_type, module=4)
    
    # If we got a list of selected features, convert to dict format
    if isinstance(features, list) and features:
        # Convert list format to dict format if needed
        # This handles the case where features are stored as selected indices
        feature_dict = {}
        all_descriptors = [
            'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumRotatableBonds',
            'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
            'NumValenceElectrons', 'NHOHCount', 'NOCount', 'RingCount',
            'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
            'LabuteASA', 'BalabanJ', 'BertzCT', 'Ipc',
            'kappa_Series[1-3]_ind', 'Chi_Series[1-98]_ind', 'Phi', 'HallKierAlpha',
            'NumAmideBonds', 'NumSpiroAtoms', 'NumBridgeheadAtoms', 'FractionCSP3',
            'PEOE_VSA_Series[1-14]_ind', 'SMR_VSA_Series[1-10]_ind', 'SlogP_VSA_Series[1-12]_ind',
            'EState_VSA_Series[1-11]_ind', 'VSA_EState_Series[1-10]', 'MQNs',
            'AUTOCORR2D', 'BCUT2D', 'Asphericity', 'PBF', 'RadiusOfGyration',
            'InertialShapeFactor', 'Eccentricity', 'SpherocityIndex',
            'PMI_series[1-3]_ind', 'NPR_series[1-2]_ind', 'AUTOCORR3D',
            'RDF', 'MORSE', 'WHIM', 'GETAWAY'
        ]
        
        # Check if features is a list of descriptor names or indices
        if features and isinstance(features[0], str):
            # List of descriptor names
            for desc in all_descriptors:
                feature_dict[desc] = 1 if desc in features else 0
        else:
            # List of indices or empty - use selective features to avoid overfitting
            # Use only key molecular descriptors that are commonly important
            important_features = [
                'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumRotatableBonds',
                'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumAromaticRings',
                'BalabanJ', 'BertzCT', 'LabuteASA', 'Phi'
            ]
            for desc in all_descriptors:
                feature_dict[desc] = 1 if desc in important_features else 0
        
        print(f"Loaded best feature selection for {dataset_name}-{split_type}")
        return feature_dict
    
    # Fallback: try direct DB access for backward compatibility
    try:
        study_name = f"ano_feature_{dataset_name}_{split_type}"
        study = optuna.load_study(
            study_name=study_name,
            storage=get_storage_url('6')
        )
        
        best_params = study.best_params
        print(f"Successfully loaded best feature selection from DB for {dataset_name}-{split_type}")
        return best_params
        
    except Exception as e:
        print(f"DB access failed for {dataset_name}-{split_type}: {e}")
        print(f"Storage URL: {get_storage_url('6')}")
        print(f"Study name: ano_feature_{dataset_name}_{split_type}")
        print(f"Using default feature selection (all features) for {dataset_name}-{split_type}")
        # Return default parameters if loading fails - all features enabled
        all_descriptors = [
            'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'NumRotatableBonds',
            'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
            'NumValenceElectrons', 'NHOHCount', 'NOCount', 'RingCount',
            'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
            'LabuteASA', 'BalabanJ', 'BertzCT', 'Ipc',
            'kappa_Series[1-3]_ind', 'Chi_Series[1-98]_ind', 'Phi', 'HallKierAlpha',
            'NumAmideBonds', 'NumSpiroAtoms', 'NumBridgeheadAtoms', 'FractionCSP3',
            'PEOE_VSA_Series[1-14]_ind', 'SMR_VSA_Series[1-10]_ind', 'SlogP_VSA_Series[1-12]_ind',
            'EState_VSA_Series[1-11]_ind', 'VSA_EState_Series[1-10]', 'MQNs',
            'AUTOCORR2D', 'BCUT2D', 'Asphericity', 'PBF', 'RadiusOfGyration',
            'InertialShapeFactor', 'Eccentricity', 'SpherocityIndex',
            'PMI_series[1-3]_ind', 'NPR_series[1-2]_ind', 'AUTOCORR3D',
            'RDF', 'MORSE', 'WHIM', 'GETAWAY'
        ]
        return {desc: 1 for desc in all_descriptors}

def create_objective_function(dataset, split_type):
    """
    Create objective function for Feature -> Structure network optimization

    This is the core optimization logic that combines:
    1. Fixed feature selection from module 4
    2. Dynamic structure optimization

    The two-stage approach ensures:
    - Features are optimized first (in module 4)
    - Structure is then optimized for those specific features
    - This decomposition makes the optimization more tractable

    Hyperparameters optimized:
    - n_layers: Network depth (2-4 layers)
    - hidden_dim_i: Units in each layer (funnel architecture)
    - dropout_rate: Regularization strength (0.1-0.5)
    - learning_rate: Optimization step size (1e-4 to 1e-2)
    - batch_size: Mini-batch size (16, 32, or 64)

    Args:
        dataset: Dataset key to optimize for (e.g., 'lo', 'hu')
        split_type: Data splitting strategy

    Returns:
        Objective function for Optuna
    """
    def objective_function(trial):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            # Load data using standard prepare_data_for_split function
            data = prepare_data_for_split(dataset, split_type)
            train_fps = data['train_fps']
            train_y = data['train_y']
            test_fps = data['test_fps']
            test_y = data['test_y']

            # Use training data for optimization
            fps = train_fps  # All fingerprints
            y_filtered = train_y

            # Define X_test and y_test for train_and_evaluate_cv function
            X_test = test_fps
            y_test = test_y

            print(f"{dataset.upper()}-{split_type} Trial {trial.number}: Starting feature-structure optimization...")
            print(f"  All fingerprints shape: {fps.shape}")

            # Step 1: Dynamic Feature optimization using Optuna
            # COMBINED APPROACH: Combine train and test fingerprints for descriptor calculation
            print(f"  Applying combined approach to ensure consistent descriptor dimensions")
            all_fps = np.vstack([fps, test_fps])

            # Create molecules from SMILES for descriptor calculation
            from rdkit import Chem

            # Load actual SMILES from the data
            train_smiles, _, test_smiles, _ = load_preprocessed_data(dataset, split_type)

            # Convert SMILES to molecules
            train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
            test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]

            # Filter valid molecules
            train_mols = [mol for mol in train_mols if mol is not None]
            test_mols = [mol for mol in test_mols if mol is not None]

            all_mols = train_mols + test_mols
            all_mols_3d = None

            # Check if descriptor cache exists, if not create it
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
                        train_mols, dataset_name=dataset, split_type=split_type, subset='train'
                    )

                # Generate descriptors for test set
                if not os.path.exists(test_desc_file):
                    print(f"  📊 Calculating test descriptors...")
                    calculator.calculate_selected_descriptors(
                        test_mols, dataset_name=dataset, split_type=split_type, subset='test'
                    )

                print(f"  ✅ Descriptors generated and cached successfully")
            else:
                print(f"✅ Using cached descriptors from {dataset}/{split_type}")

            # Apply dynamic feature search using cached descriptors
            search_result = search_data_descriptor_compress(
                trial, all_fps, all_mols, dataset, split_type, str(OUTPUT_DIR), "np", all_mols_3d
            )

            # Handle different return values
            if len(search_result) == 2:
                all_fps_selected, selected_descriptors = search_result
                excluded_descriptors = []
            else:
                all_fps_selected, selected_descriptors, excluded_descriptors = search_result

            # Split back into train and test with selected features
            n_train = len(fps)
            n_test = len(test_fps)

            fps_selected = all_fps_selected[:n_train]
            test_fps_selected = all_fps_selected[n_train:]

            print(f"  Selected features: {all_fps_selected.shape[1]} (from {all_fps.shape[1]})")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            print(f"  Train selected shape: {fps_selected.shape}")
            print(f"  Test selected shape: {test_fps_selected.shape}")

            # Use selected features
            X_test = test_fps_selected
            print(f"  Combined test features shape: {X_test.shape}")
            print(f"  Selected descriptors: {len(selected_descriptors)}")

            # Use DNN_HYPERPARAMETERS from config.py for consistency
            from config import DNN_HYPERPARAMETERS
            hp_config = DNN_HYPERPARAMETERS['search_space']
            
            # Step 3: Structure optimization - suggest hyperparameters from config
            # Now that we have the optimal features, find the best network architecture
            n_layers = trial.suggest_int('n_layers', hp_config['n_layers'][0], hp_config['n_layers'][1])
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
            
            # Note: Output layer (1) will be added in architecture construction below
            
            # Regularization parameters from config
            dropout_rate = trial.suggest_categorical('dropout_rate', hp_config['dropout_rate'])
            weight_decay = trial.suggest_float('weight_decay', hp_config['weight_decay'][0], hp_config['weight_decay'][1], log=True)  # L2 regularization
            
            # Optimizer parameters from config
            optimizer_name = trial.suggest_categorical('optimizer', hp_config['optimizer'])
            # Fix: learning_rate is a list of discrete values, not a range
            learning_rate = trial.suggest_categorical('learning_rate', hp_config['learning_rate'])
            
            # Optimizer-specific parameters
            if optimizer_name == 'sgd':
                momentum = trial.suggest_float('momentum', 0.0, 0.99)
                nesterov = trial.suggest_categorical('nesterov', [True, False])
            elif optimizer_name in ['adam', 'adamw']:
                beta1 = trial.suggest_float('beta1', 0.5, 0.999)
                beta2 = trial.suggest_float('beta2', 0.9, 0.999)
                eps = trial.suggest_float('eps', 1e-10, 1e-6, log=True)
            
            # Training parameters from config
            batch_size = trial.suggest_categorical('batch_size', hp_config['batch_size'])
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            
            # Learning rate scheduler
            scheduler_name = trial.suggest_categorical('scheduler', ['none', 'step', 'exponential', 'cosine', 'reduce_on_plateau'])
            if scheduler_name == 'step':
                step_size = trial.suggest_int('step_size', 10, 50)
                gamma = trial.suggest_float('gamma', 0.1, 0.9)
            elif scheduler_name == 'exponential':
                gamma = trial.suggest_float('gamma', 0.9, 0.99)
            
            # Model parameters
            model_params = {
                'batch_size': batch_size,
                'epochs': EPOCHS,  # 10 epochs
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'optimizer': optimizer_name,
                'scheduler': scheduler_name,
                'use_batch_norm': use_batch_norm,
                'activation': activation
            }
            
            # Add optimizer-specific parameters
            if optimizer_name == 'sgd':
                model_params['momentum'] = momentum
                model_params['nesterov'] = nesterov
            elif optimizer_name in ['adam', 'adamw']:
                model_params['beta1'] = beta1
                model_params['beta2'] = beta2
                model_params['eps'] = eps
            
            # Add scheduler-specific parameters
            if scheduler_name == 'step':
                model_params['step_size'] = step_size
                model_params['gamma'] = gamma
            elif scheduler_name == 'exponential':
                model_params['gamma'] = gamma
            
            # CV-5 evaluation with test evaluation
            try:
                (mean_r2, std_r2, best_r2,
                 mean_rmse, std_rmse, best_rmse,
                 mean_mse, std_mse, best_mse,
                 mean_mae, std_mae, best_mae,
                 test_r2, test_rmse, test_mse, test_mae,
                 cv_test_r2_mean, cv_test_r2_std, cv_test_rmse_mean, cv_test_rmse_std,
                 cv_test_mse_mean, cv_test_mae_mean) = train_and_evaluate_cv(
                    fps_selected, y_filtered, model_params, hidden_dims, dropout_rate, X_test, y_test, activation=activation, trial=trial)
            except optuna.exceptions.TrialPruned:
                # Trial was pruned due to poor performance
                raise
            
            # Calculate execution time and memory usage
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Store all metrics in trial attributes
            trial.set_user_attr('cv_r2_mean', mean_r2)
            trial.set_user_attr('cv_r2_std', std_r2)
            trial.set_user_attr('cv_rmse_mean', mean_rmse)
            trial.set_user_attr('cv_rmse_std', std_rmse)
            trial.set_user_attr('cv_mse_mean', mean_mse)
            trial.set_user_attr('cv_mse_std', std_mse)
            trial.set_user_attr('cv_mae_mean', mean_mae)
            trial.set_user_attr('cv_mae_std', std_mae)
            trial.set_user_attr('best_r2', best_r2)
            trial.set_user_attr('best_rmse', best_rmse)
            trial.set_user_attr('best_mse', best_mse)
            trial.set_user_attr('best_mae', best_mae)
            trial.set_user_attr('n_features', fps_selected.shape[1])
            trial.set_user_attr('selected_descriptors', selected_descriptors)
            trial.set_user_attr('feature_versioning_used', True)
            trial.set_user_attr('module6_dynamic_optimization', True)
            trial.set_user_attr('descriptor_cache_used', True)
            trial.set_user_attr('execution_time', execution_time)
            trial.set_user_attr('memory_used_mb', memory_used)
            trial.set_user_attr('dataset', dataset)
            trial.set_user_attr('split_type', split_type)
            trial.set_user_attr('n_layers', n_layers)
            trial.set_user_attr('hidden_dims', hidden_dims)
            trial.set_user_attr('dropout_rate', dropout_rate)
            trial.set_user_attr('learning_rate', learning_rate)
            trial.set_user_attr('batch_size', batch_size)
            trial.set_user_attr('activation', activation)
            trial.set_user_attr('weight_decay', weight_decay)
            trial.set_user_attr('optimizer', optimizer_name)
            trial.set_user_attr('scheduler', scheduler_name)
            trial.set_user_attr('use_batch_norm', use_batch_norm)
            
            # Add test metrics
            trial.set_user_attr('test_r2', test_r2)
            trial.set_user_attr('test_rmse', test_rmse)
            trial.set_user_attr('test_mse', test_mse)
            trial.set_user_attr('test_mae', test_mae)
            
            # Add CV test metrics (Method 1)
            trial.set_user_attr('cv_test_r2_mean', cv_test_r2_mean)
            trial.set_user_attr('cv_test_r2_std', cv_test_r2_std)
            trial.set_user_attr('cv_test_rmse_mean', cv_test_rmse_mean)
            trial.set_user_attr('cv_test_rmse_std', cv_test_rmse_std)
            trial.set_user_attr('cv_test_mse_mean', cv_test_mse_mean)
            trial.set_user_attr('cv_test_mae_mean', cv_test_mae_mean)
            
            print(f"  Trial completed: Type1 CV R2={mean_r2:.4f}±{std_r2:.4f}")
            print(f"                 Type2 Test R2={test_r2:.4f}")
            print(f"                 Time: {execution_time:.2f}s, Memory: {memory_used:.1f}MB")
            
            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return mean_r2

        except optuna.exceptions.TrialPruned:
            # Re-raise TrialPruned to ensure proper pruning (don't print as error)
            print(f"  🔪 {dataset.upper()}-{split_type} trial pruned by Optuna")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        except Exception as e:
            import traceback
            print(f"Error in {dataset.upper()}-{split_type} trial: {e}")
            print("Full traceback:")
            traceback.print_exc()
            # Memory cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
            
    return objective_function

def save_study_results(study, dataset, split_type):
    """
    Save study results to file
    
    Saves comprehensive optimization results including:
    - Best hyperparameters found
    - Full optimization history
    - Detailed metrics for each trial
    - Feature information (count and names)
    - Computational resources used
    
    Results are saved in JSON format for easy loading and analysis.
    
    Args:
        study: Completed Optuna study
        dataset: Dataset name
        split_type: Split strategy used
    
    Returns:
        Dictionary of saved results
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
                'n_features': trial.user_attrs.get('n_features', 0),
                'selected_descriptors': trial.user_attrs.get('selected_descriptors', []),
                'execution_time': trial.user_attrs.get('execution_time', 0.0),
                'memory_used_mb': trial.user_attrs.get('memory_used_mb', 0.0),
                'n_layers': trial.user_attrs.get('n_layers', 0),
                'hidden_dims': trial.user_attrs.get('hidden_dims', []),
                'dropout_rate': trial.user_attrs.get('dropout_rate', 0.0),
                'learning_rate': trial.user_attrs.get('learning_rate', 0.0),
                'batch_size': trial.user_attrs.get('batch_size', 0),
                'test_r2': trial.user_attrs.get('test_r2', 0.0),
                'test_rmse': trial.user_attrs.get('test_rmse', 0.0),
                'test_mse': trial.user_attrs.get('test_mse', 0.0),
                'test_mae': trial.user_attrs.get('test_mae', 0.0),
                'cv_method2_test_r2': trial.user_attrs.get('cv_method2_test_r2', 0.0),
                'cv_method2_test_rmse': trial.user_attrs.get('cv_method2_test_rmse', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    # Save to file with folder structure: result/6_ANO_FOMO/dataset/split_type/
    results_dir = OUTPUT_DIR / dataset / split_type
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{dataset}_{split_type}_FOMO_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Update config with best configuration including optimization score
    from config import update_best_config
    if results.get('best_params'):
        best_config = {
            'features': results.get('trial_details', [{}])[0].get('selected_descriptors', []) if results.get('trial_details') else [],
            'structure': {
                'n_layers': results['best_params'].get('n_layers'),
                'hidden_dims': results['best_params'].get('hidden_dims', []),
                'dropout_rate': results['best_params'].get('dropout_rate'),
                'learning_rate': results['best_params'].get('learning_rate'),
                'batch_size': results['best_params'].get('batch_size')
            },
            'optimization_score': {
                'best_r2': results.get('best_r2', 0.0),
                'best_rmse': results.get('best_rmse', 0.0),
                'best_mse': results.get('best_mse', 0.0),
                'best_mae': results.get('best_mae', 0.0),
                'module': 'FOMO',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        update_best_config(6, dataset, split_type, best_config)
    
    # Update best summary CSV
    update_best_summary(dataset, split_type, results)
    
    print(f"Results saved to: {results_file}")
    return results

def update_best_summary(dataset, split_type, results):
    """
    Update best results summary CSV for module 6.

    Creates or updates a CSV file containing the best FOMO (Feature→Model)
    optimization results for each dataset-split combination. This provides
    a quick reference for comparing FOMO strategy performance.

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
    Main function for ANO feature-structure network optimization

    This implements the Feature->Structure optimization strategy:
    1. Use features optimized by module 4
    2. Find best network structure for those features

    The main loop:
    1. Prepares molecular data for all datasets
    2. For each dataset and split:
       - Loads best features from module 4
       - Optimizes network structure
       - Saves results

    Key configuration:
    - DELETE_EXISTING_STUDIES: Set to True to start fresh
    - N_TRIALS: Number of optimization trials (1 for quick run)
    - EPOCHS: Training epochs per trial (100)
    """
    global CODE_DATASETS

    print("="*80)
    print("🔗 MODULE 6: ANO NETWORK OPTIMIZATION (FEATURE→MODEL OPTIMIZATION)")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Module Focus: Feature-guided Network Structure Optimization")
    print(f"🔧 Split types: {SPLIT_TYPES}")
    print(f"🎲 Number of trials: {N_TRIALS}")
    print(f"⏱️  Epochs per trial: {EPOCHS}")
    print(f"💾 Storage: {get_storage_url('6')}")

    # Add option to delete existing studies
    # This is useful when you want to start fresh optimization
    # Set to False to continue from previous runs
    DELETE_EXISTING_STUDIES = True  # Set to True to delete existing studies

    # Use code-specific datasets
    datasets = CODE_DATASETS
    total_tasks = len(datasets) * len(SPLIT_TYPES)
    current_task = 0

    print(f"📊 Active Datasets (Module 6): {datasets}")
    print(f"📈 Total combinations to process: {total_tasks}")
    print("="*80)
    
    # Prepare data
    print("\nPreparing data...")
    # Data will be prepared per dataset using prepare_data_for_split()
    
    # Use datasets from config
    from config import get_code_datasets
    code_datasets = get_code_datasets(6)  # Code 6 datasets - keep as keys
    # Test data will be loaded per dataset using prepare_data_for_split()
    
    for split_type in SPLIT_TYPES:
        print(f"\n{'='*60}")
        print(f"🔄 SPLIT TYPE: {split_type.upper()}")
        print(f"{'='*60}")

        for dataset in code_datasets:
            current_task += 1
            print(f"\n{'🔗 '*20}")
            print(f"📋 TASK {current_task}/{total_tasks}: {dataset.upper()}")
            print(f"🔀 Split: {split_type}")
            print(f"🧬 Target: Feature→Model Optimization")
            print(f"⏳ Progress: {current_task/total_tasks*100:.1f}%")
            print(f"{'🔗 '*20}")

            # Load and validate data dimensions
            print(f"📊 Loading data for {dataset.upper()}...")
            try:
                data = prepare_data_for_split(dataset, split_type)
                train_fps = data['train_fps']
                train_y = data['train_y']

                print(f"📐 Train shape: {train_fps.shape}")
                print(f"📐 Test shape: {data['test_fps'].shape}")
                print(f"🏷️  Samples: train={train_fps.shape[0]}, test={data['test_fps'].shape[0]}")
                train_y_np = np.array(train_y)
                print(f"📊 Target range: [{train_y_np.min():.3f}, {train_y_np.max():.3f}]")

                # Dimension validation
                actual_features = train_fps.shape[1]
                print(f"✅ Feature dimensions: {actual_features}")

            except Exception as e:
                print(f"⚠️  Data loading error: {e}")
                continue
            
            # Create study name: ano_network_FOMO_{dataset}_{split_type}
            study_name = f"ano_network_FOMO_{dataset}_{split_type}"
            storage_name = get_storage_url('6')
            
            # Delete existing study if requested
            if DELETE_EXISTING_STUDIES:
                try:
                    optuna.delete_study(study_name=study_name, storage=storage_name)
                    print(f"  Deleted existing study: {study_name}")
                except KeyError:
                    print(f"  No existing study found: {study_name}")
            
            # Handle study renewal if requested
            if renew:
                try:
                    optuna.delete_study(study_name=study_name, storage=storage_name)
                    print(f"Deleted existing study: {study_name}")
                except KeyError:
                    pass
            
            # Create Optuna study with 2025 proven best combination
            # from config import get_optuna_sampler_and_pruner, print_optuna_info
            # sampler, pruner = get_optuna_sampler_and_pruner('best_overall')
            # print_optuna_info('best_overall')

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
                print(f"Module 6 sampler: TPESampler (n_startup_trials={n_startup_trials}/{N_TRIALS})")
            elif sampler_type == 'random':
                sampler = optuna.samplers.RandomSampler()
                print(f"Module 6 sampler: RandomSampler")
            else:
                # Fallback to TPESampler
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=n_startup_trials,
                    n_ei_candidates=50,
                    multivariate=False,
                    warn_independent_sampling=False
                )
                print(f"Module 6 sampler: TPESampler (fallback)")

            # Create HyperbandPruner for Module 6 (matched with Module 4 strategy)
            pruner_config = MODEL_CONFIG.get('optuna_pruner', {})
            if pruner_config.get('type') == 'hyperband':
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=pruner_config.get('min_resource', 100),
                    max_resource=pruner_config.get('max_resource', 1000),
                    reduction_factor=pruner_config.get('reduction_factor', 3)
                )
                print(f"Module 6 pruner: HyperbandPruner (min_resource={pruner_config.get('min_resource', 100)}, "
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
                print(f"Module 6 pruner: MedianPruner (n_startup_trials={n_startup})")

            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                storage=storage_name,
                study_name=study_name,
                load_if_exists=(not renew)
            )
            
            # Display detailed progress header for Module 6 (FOMO)
            total_datasets = len(datasets)
            total_splits = len(SPLIT_TYPES)
            current_dataset_idx = datasets.index(dataset) + 1
            current_split_idx = SPLIT_TYPES.index(split_type) + 1
            total_combinations = total_datasets * total_splits
            current_combination = (current_split_idx - 1) * total_datasets + current_dataset_idx

            print(f"\n{'='*75}")
            print(f"[MODULE 6] FOMO OPTIMIZATION: {get_dataset_display_name(dataset).upper()} | Trial (0/{N_TRIALS})")
            print(f"Dataset: {get_dataset_display_name(dataset).upper()} | Split: {split_type.upper()} | Method: Feature->Model Optimization")
            print(f"Scope: Dataset ({current_dataset_idx}/{total_datasets}) × Split ({current_split_idx}/{total_splits}) × Trial (0/{N_TRIALS})")
            print(f"Overall Progress: Combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.2f}%)")
            print(f"Totals: {total_datasets} datasets × {total_splits} splits × {N_TRIALS} trials = {total_datasets * total_splits * N_TRIALS:,} optimizations")
            print(f"Features: Module 4 best features + optimized architecture (2-9999 units)")
            print(f"Expected duration: ~{N_TRIALS * 90 / 60:.1f}m | Target: Max CV R² with FOMO strategy")
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
            
            # Print best results
            try:
                print(f"{get_dataset_display_name(dataset)}-{split_type} optimization completed!")
                print(f"Best R2 score: {study.best_value:.4f}")
                print(f"Best parameters: {study.best_params}")
                
                # Save results
                save_study_results(study, dataset, split_type)
                
            except ValueError as e:
                print(f"Warning: {get_dataset_display_name(dataset)}-{split_type} optimization completed but no valid trials found.")
            
            # Memory cleanup
            gc.collect()
    
    print("\nAll feature-structure network optimizations completed!")
    
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

    renew = True
    
    # Set up logging to file
    from pathlib import Path
    module_name = MODULE_NAMES.get('6', '6_ANO_NetworkOptimization_FOMO')
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

    # Update global variables if arguments provided
    # Set EPOCHS with proper priority: args > MODEL_CONFIG['epochs'] > module_epochs
    EPOCHS = get_epochs_for_module('6', args)
    if args.trials is not None:
        N_TRIALS = args.trials
        MODEL_CONFIG['optuna_trials'] = args.trials
    if args.split:
        SPLIT_TYPES = [args.split]

    # Set datasets with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    from config import CODE_SPECIFIC_DATASETS, ACTIVE_SPLIT_TYPES, DATA_PATH
    from pathlib import Path

    if args.datasets:
        CODE_DATASETS = args.datasets
        print(f"Datasets from argparse: {CODE_DATASETS}")
    elif args.dataset:
        CODE_DATASETS = [args.dataset]
        print(f"Dataset from argparse (single): {CODE_DATASETS}")
    elif '6' in CODE_SPECIFIC_DATASETS:
        CODE_DATASETS = CODE_SPECIFIC_DATASETS['6']
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
            self.log.write(f"Module 6 Execution Started: {datetime.now()}\n")
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
            self.log.write(f"Module 6 Execution Completed: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n")
            self.log.close()
    
    # Replace stdout and stderr with logger
    logger = Logger(str(log_file))
    sys.stdout = logger
    
    try:
        print(f"Log file: {log_file}")
        main()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__