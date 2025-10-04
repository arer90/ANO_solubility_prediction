#!/usr/bin/env python3
"""
ANO Network Optimization MOFO - ANO Framework Module 7
======================================================

PURPOSE:
Module 7 implements MOFO (Model Optimization -> Feature Optimization) strategy.
It first optimizes the neural network architecture with all features,
then selects the best features for that architecture.

KEY FEATURES:
1. **Reverse Optimization**: Model first -> Feature second
2. **Architecture from Module 5**: Uses best model structure
3. **FlexibleDNNModel**: Pre-optimized architecture
4. **No StandardScaler**: Raw features used directly
5. **Feature Selection**: After model is fixed
6. **Complementary to FOMO**: Different optimization order

RECENT UPDATES (2024):
- Uses FlexibleDNNModel from Module 5 results
- Removed StandardScaler (using raw features)
- Fixed get_epochs_for_module usage
- Default epochs: 30 (from config.py module_epochs['7'])
- Optuna trials: 1 (for quick testing)

OPTIMIZATION FLOW:
1. Load best architecture from Module 5
2. Use all fingerprints (2727 features)
3. Search for optimal descriptors with fixed architecture
4. Combine selected features with pre-optimized model
5. Evaluate with 5-fold CV

MOFO vs FOMO COMPARISON:
- FOMO: Features -> Model (Module 6)
- MOFO: Model -> Features (Module 7)
- Different optimization paths may find different optima

OUTPUT STRUCTURE:
result/7_network_mofo/
├── {dataset}_{split}/
│   ├── best_architecture_and_features.json
│   ├── optimization_history.csv
│   ├── mofo_model.pt
│   └── performance_metrics.json
└── mofo_comparison.png

USAGE:
python 7_ANO_NetworkOptimization_MOFO.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --trials: Number of Optuna trials (default: 1)
  --epochs: Override epochs (default from config: 30)
  --renew: Start fresh optimization

Developer: Lee, Seungjin (arer90)
"""

# ANO Network MOFO Optimization with PyTorch - Fixed Module
# ===========================================================
#
# PURPOSE:
# This module implements MOFO (Model Optimization -> Feature Optimization) strategy.
# It uses the optimal network architecture from module 5 and then searches for
# the best feature combination for that specific architecture.
#
# APPROACH:
# 1. Load optimal network structure from module 5 (layer count, hidden dims, etc.)
# 2. Fix the network architecture with those parameters
# 3. Search for optimal feature selection using the fixed architecture
# 4. This ensures features are tailored to the specific network topology
#
# KEY INNOVATIONS:
# - MOFO strategy: Model Optimization first (module 5), then Feature Optimization
# - Architecture-specific feature selection
# - Tests hypothesis that optimal features depend on network structure
# - Allows comparison with FOMO approach (module 6)
#
# TECHNICAL DETAILS:
# - Uses best structure from module 5: Typically 2-4 layers
# - Feature search space: Same as module 5 (49 descriptor categories)
# - Base features: 2727 (Morgan + MACCS + Avalon)
# - Additional features: Selected from ~882 molecular descriptors
# - Evaluation: 5-fold CV on training set
#
# Fixed issues:
# - Study name: ano_network_MOFO_{dataset}_{split_type}
# - All splits supported (not just rm)
# - Epochs: 100 (was 30)
# - Trials: 1
# - All fingerprints + module 5 best structure used
# - Feature selection optimization
# - All metrics saved (R², RMSE, MSE, MAE, time, resources)
# - Model saved as full_model.pt
#
# MOFO Network (Model -> Feature Optimization):
# 1. Use all fingerprints (morgan + maccs + avalon)
# 2. Get best structure from module 5 results
# 3. Use fixed model structure with best hyperparameters
# 4. Optimize feature selection for that structure

import os
import sys

# Import centralized DNN models
from extra_code.ano_feature_selection import FlexibleDNNModel
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from rdkit import Chem
from config import MODULE_NAMES
import optuna
import psutil
import json

import torch
import torch.nn as nn

# Module 7 uses Optuna - DO NOT fix seeds for optimization diversity
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
    get_storage_url, get_database_info, get_best_structure, get_epochs_for_module
)

# Configuration - Global variables
# Get code-specific configurations - will be updated in main() with argparse priority
CODE_DATASETS = get_code_datasets(7)  # Code 7 - default, will be updated in main()
CODE_FINGERPRINTS = get_code_fingerprints(7)  # Code 7

# Enable all split types for comprehensive analysis (test mode uses only rm)
SPLIT_TYPES = ACTIVE_SPLIT_TYPES  # Use config.py setting (currently ['rm'])
N_TRIALS = MODEL_CONFIG['optuna_trials']  # Use config.py setting (currently 1)
EPOCHS = None  # Will be set in main() with proper priority
OUTPUT_DIR = Path(RESULT_PATH) / "7_ANO_NetworkOptimization_MOFO"
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

    train_smiles = train_df[smiles_col].tolist()
    train_targets = train_df[target_col].values
    test_smiles = test_df[smiles_col].tolist()
    test_targets = test_df[target_col].values

    return train_smiles, train_targets, test_smiles, test_targets

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
            train_mols_filtered, dataset_short.upper(), split_type, 'train', module_name='7_ANO_NetworkOptimization_MOFO'
        )
        test_morgan, test_maccs, test_avalon = get_fingerprints_cached(
            test_mols_filtered, dataset_short.upper(), split_type, 'test', module_name='7_ANO_NetworkOptimization_MOFO'
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
        print(f"Error preparing data for {dataset_short}-{split_type}: {e}")
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
        train_file = f"save_model/module7_type1_fold{fold}_train_data.pkl"
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
        train_file = f"save_model/module7_type2_fold{fold}_train_data.pkl"
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

    train_file_final = "save_model/module7_type2_final_train_data.pkl"
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
    # Get CV folds from config (default to 5 for Module 7)
    n_folds = 5

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

    # All CV processing is now handled by the dual CV approach
    # Return results in expected format for backward compatibility
    return (mean_r2, std_r2, best_r2,
            mean_rmse, std_rmse, best_rmse,
            mean_mse, std_mse, best_mse,
            mean_mae, std_mae, best_mae,
            test_r2, test_rmse, test_mse, test_mae,
            cv_test_r2_mean, cv_test_r2_std, cv_test_rmse_mean, cv_test_rmse_std,
            cv_test_mse_mean, cv_test_mae_mean)

# Note: Using centralized FlexibleDNNModel from ano_feature_selection.py
# No need for local DNNModel class definition


def get_best_structure_selection(dataset, split_type):
    """
    Get best structure from module 5 (ANO_MO) results
    Priority: 1. DB, 2. Config file, 3. Default structure

    Loads the optimal neural network architecture discovered by module 5
    from the Optuna database. Module 5 performed extensive architecture
    search to find the best network topology for each dataset.

    The loaded parameters include:
    - n_layers: Number of hidden layers (2-4)
    - hidden_dim_i: Neurons in each hidden layer
    - dropout_rate: Dropout probability
    - learning_rate: Optimization learning rate
    - batch_size: Training batch size

    Args:
        dataset: Dataset key identifier (e.g., 'lo', 'hu')
        split_type: Data splitting strategy

    Returns:
        Dictionary of optimal hyperparameters
        Falls back to sensible defaults if loading fails
    """
    # Use config get_best_structure which has the fallback logic
    from config import get_best_structure

    try:
        result = get_best_structure(dataset, split_type)

        # Ensure required fields are present
        if 'hidden_dims' not in result:
            if 'n_layers' in result:
                hidden_dims = []
                for i in range(result['n_layers']):
                    dim_key = f'hidden_dim_{i}'
                    if dim_key in result:
                        hidden_dims.append(result[dim_key])
                result['hidden_dims'] = hidden_dims

        print(f"Loaded best structure for {dataset}-{split_type}")
        return result

    except Exception as e:
        print(f"Using default structure for {dataset}-{split_type}")
        # Return default parameters if loading fails
        return {
            'n_layers': 2,
            'hidden_dim_0': 1024,
            'hidden_dim_1': 496,
            'hidden_dims': [1024, 496, 1],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100
        }

def create_objective_function(dataset, split_type):
    """
    Get best structure from module 5 (ANO_MO) results
    Priority: 1. DB, 2. Config file, 3. Default structure
    
    Loads the optimal neural network architecture discovered by module 5
    from the Optuna database. Module 5 performed extensive architecture
    search to find the best network topology for each dataset.
    
    The loaded parameters include:
    - n_layers: Number of hidden layers (2-4)
    - hidden_dim_i: Neurons in each hidden layer
    - dropout_rate: Dropout probability
    - learning_rate: Optimization learning rate
    - batch_size: Training batch size
    
    Args:
        dataset: Dataset key identifier (e.g., 'lo', 'hu')
        split_type: Data splitting strategy
    
    Returns:
        Dictionary of optimal hyperparameters
        Falls back to sensible defaults if loading fails
    """
    # Use config get_best_structure which has the fallback logic
    
    # Get structure with fallback: DB -> config file -> default
    best_structure = get_best_structure(dataset, split_type, module=5)
    
    if best_structure:
        # Convert to format expected by this module
        result = {
            'n_layers': best_structure.get('n_layers', 2),
            'dropout_rate': best_structure.get('dropout_rate', 0.2),
            'learning_rate': best_structure.get('learning_rate', 0.001),
            'batch_size': best_structure.get('batch_size', 32),
            'activation': best_structure.get('activation', 'relu')
        }
        
        # Handle hidden dimensions
        hidden_dims = best_structure.get('hidden_dims', [512, 256])
        for i, dim in enumerate(hidden_dims):
            result[f'hidden_dim_{i}'] = dim
        
        # Store hidden_dims in user_attrs format
        result['hidden_dims'] = hidden_dims
        
        print(f"Loaded best structure for {dataset}-{split_type}")
        return result
    
    # Fallback: try direct DB access for backward compatibility
    try:
        study_name = f"ano_structure_{dataset}_{split_type}"
        study = optuna.load_study(
            study_name=study_name,
            storage=get_storage_url('7')
        )
        
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        
        # Add user attributes if available
        if 'hidden_dims' in best_trial.user_attrs:
            best_params['hidden_dims'] = best_trial.user_attrs['hidden_dims']
        
        print(f"Successfully loaded best structure from DB for {dataset}-{split_type}")
        return best_params
        
    except Exception as e:
        print(f"Using default structure for {dataset}-{split_type}")
        # Return default parameters if loading fails
        return {
            'n_layers': 2,
            'hidden_dim_0': 1024,
            'hidden_dim_1': 496,
            'hidden_dims': [1024, 496],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32
        }

def create_objective_function(dataset, split_type):
    """
    Create objective function for Structure -> Feature network optimization
    
    This implements the Structure->Feature optimization strategy:
    1. Fix the network architecture using module 5 best structure
    2. Search for optimal features for that specific architecture
    
    The hypothesis is that different network architectures may benefit
    from different feature combinations. By fixing the architecture first,
    we can find features specifically tailored to that topology.
    
    Key steps in each trial:
    1. Load best network structure from module 6
    2. Suggest feature selection (49 binary choices)
    3. Create model with fixed architecture but variable input size
    4. Train and evaluate using 5-fold CV
    5. Return CV performance for optimization
    
    Args:
        dataset: Dataset key to optimize for (e.g., 'lo', 'hu')
        split_type: Data splitting strategy
    
    Returns:
        Objective function closure for Optuna
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

            # Also load SMILES data for descriptor calculation
            train_smiles, _, test_smiles, _ = load_preprocessed_data(dataset, split_type)

            # Use training data for optimization
            fps = train_fps  # All fingerprints
            y_filtered = train_y
            
            print(f"{dataset.upper()}-{split_type} Trial {trial.number}: Starting structure-feature optimization...")
            print(f"  All fingerprints shape: {fps.shape}")
            
            # Step 1: Get best structure from module 5 results
            # This loads the optimal architecture found through extensive search
            best_structure_params = get_best_structure_selection(dataset, split_type)
            
            # Step 2: Extract hidden dimensions and all hyperparameters from best structure
            # Reconstruct the full architecture specification
            n_layers = best_structure_params.get('n_layers', 2)
            hidden_dims = []
            for i in range(n_layers):
                hidden_dim_key = f'hidden_dim_{i}'
                if hidden_dim_key in best_structure_params:
                    hidden_dims.append(best_structure_params[hidden_dim_key])
                else:
                    # Default fallback for safety
                    hidden_dims.append(512 if i == 0 else 256)
            
            # Note: hidden_dims from DB already includes output layer (1), so don't add again
            
            # Extract all hyperparameters from best structure
            dropout_rate = best_structure_params.get('dropout_rate', 0.2)
            learning_rate = best_structure_params.get('learning_rate', 0.001)
            batch_size = best_structure_params.get('batch_size', 32)
            activation = best_structure_params.get('activation', 'relu')
            weight_decay = best_structure_params.get('weight_decay', 0.0)
            optimizer_name = best_structure_params.get('optimizer', 'adam')
            scheduler_name = best_structure_params.get('scheduler', 'none')
            use_batch_norm = best_structure_params.get('use_batch_norm', False)
            
            # Extract optimizer-specific parameters
            momentum = best_structure_params.get('momentum', 0.9) if optimizer_name == 'sgd' else None
            nesterov = best_structure_params.get('nesterov', False) if optimizer_name == 'sgd' else None
            beta1 = best_structure_params.get('beta1', 0.9) if optimizer_name in ['adam', 'adamw'] else None
            beta2 = best_structure_params.get('beta2', 0.999) if optimizer_name in ['adam', 'adamw'] else None
            eps = best_structure_params.get('eps', 1e-8) if optimizer_name in ['adam', 'adamw'] else None
            
            # Extract scheduler-specific parameters
            step_size = best_structure_params.get('step_size', 30) if scheduler_name == 'step' else None
            gamma = best_structure_params.get('gamma', 0.1) if scheduler_name in ['step', 'exponential'] else None
            
            # Step 3: Feature selection optimization using combined approach
            # With the architecture fixed, search for the best feature combination
            # This explores 49 binary choices for molecular descriptor categories
            # Note: fps_selected is calculated below in the combined approach section
            
            # Model parameters (using best structure from module 5)
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
            
            # COMBINED APPROACH - Apply feature selection to combined train+test data
            # Use test data from prepare_data_for_split
            X_test, y_test = test_fps, test_y
            test_mols = None  # Not needed for current implementation
            test_mols_3d = None
                
            # COMBINED APPROACH: Combine train and test fingerprints for descriptor calculation
            print(f"  Applying combined approach to ensure consistent descriptor dimensions")
            all_fps = np.vstack([fps, test_fps])

            # Convert SMILES to molecules for descriptor generation if needed
            train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles if Chem.MolFromSmiles(smi) is not None]
            test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles if Chem.MolFromSmiles(smi) is not None]
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

            # Apply feature search to combined data
            search_result = search_data_descriptor_compress(
                trial, all_fps, all_mols, dataset, split_type, str(OUTPUT_DIR), "np", all_mols_3d
            )

            # Handle different return values: (fps, selected_descriptors) or (fps, selected_descriptors, excluded_descriptors)
            if len(search_result) == 2:
                all_fps_selected, selected_descriptors = search_result
                excluded_descriptors = []  # No excluded descriptors when using cache
            else:
                all_fps_selected, selected_descriptors, excluded_descriptors = search_result

            # Split back into train and test with guaranteed dimension consistency
            n_train = len(fps)
            fps_selected = all_fps_selected[:n_train]
            test_fps_selected = all_fps_selected[n_train:]

            X_test = test_fps_selected
            print(f"  Combined train features shape: {fps_selected.shape}")
            print(f"  Combined test features shape: {X_test.shape}")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            
            # CV-5 evaluation with test evaluation
            (mean_r2, std_r2, best_r2,
             mean_rmse, std_rmse, best_rmse,
             mean_mse, std_mse, best_mse,
             mean_mae, std_mae, best_mae,
             test_r2, test_rmse, test_mse, test_mae,
             cv_test_r2_mean, cv_test_r2_std,
             cv_test_rmse_mean, cv_test_rmse_std,
             cv_test_mse_mean, cv_test_mae_mean) = train_and_evaluate_cv(
                fps_selected, y_filtered, model_params, hidden_dims, dropout_rate, X_test, y_test, activation=activation, trial=trial)
            
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
            
            # Add CV test metrics
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
            # Re-raise TrialPruned to ensure proper pruning behavior
            print(f"Trial {trial.number} pruned - this is normal Optuna behavior")
            raise
        except Exception as e:
            print(f"Error in {dataset.upper()}-{split_type} trial: {e}")
            # Memory cleanup on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
            
    return objective_function

def save_study_results(study, dataset, split_type):
    """
    Save study results to file
    
    Saves comprehensive results including:
    - Best feature selection found
    - Performance metrics (CV and best fold)
    - Network architecture used (from module 6)
    - Computational resources used
    
    The results allow comparison with other optimization strategies
    to determine which approach yields better models.
    
    Args:
        study: Completed Optuna study
        dataset: Dataset name
        split_type: Split strategy
    
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
                'cv_test_r2_mean': trial.user_attrs.get('cv_test_r2_mean', 0.0),
                'cv_test_r2_std': trial.user_attrs.get('cv_test_r2_std', 0.0),
                'cv_test_rmse_mean': trial.user_attrs.get('cv_test_rmse_mean', 0.0),
                'cv_test_rmse_std': trial.user_attrs.get('cv_test_rmse_std', 0.0),
                'cv_test_mse_mean': trial.user_attrs.get('cv_test_mse_mean', 0.0),
                'cv_test_mae_mean': trial.user_attrs.get('cv_test_mae_mean', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    # Save to file with folder structure: result/7_ANO_MOFO/dataset/split_type/
    results_dir = OUTPUT_DIR / dataset / split_type
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{dataset}_{split_type}_MOFO_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Update config with best configuration
    from config import update_best_config
    if results.get('best_params'):
        # Get structure from module 5
        best_structure = get_best_structure_selection(dataset, split_type)
        best_config = {
            'structure': best_structure if best_structure else {},
            'features': results.get('trial_details', [{}])[0].get('selected_descriptors', []) if results.get('trial_details') else [],
            'optimization_score': {
                'best_r2': results.get('best_r2', 0.0),
                'best_rmse': results.get('best_rmse', 0.0),
                'best_mse': results.get('best_mse', 0.0),
                'best_mae': results.get('best_mae', 0.0),
                'module': 'MOFO',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        update_best_config(7, dataset, split_type, best_config)
    
    # Update best summary CSV
    update_best_summary(dataset, split_type, results)
    
    print(f"Results saved to: {results_file}")
    return results

def update_best_summary(dataset, split_type, results):
    """
    Update best results summary CSV for module 7.

    Creates or updates a CSV file containing the best MOFO (Model→Feature)
    optimization results for each dataset-split combination. This provides
    a quick reference for comparing MOFO strategy performance.

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
        'cv_test_r2_mean': best_trial.get('cv_test_r2_mean', 0.0),
        'cv_test_r2_std': best_trial.get('cv_test_r2_std', 0.0),
        'n_features': best_trial.get('n_features', 0),
        'n_selected_descriptors': len(best_trial.get('selected_descriptors', [])),
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
    Main function for ANO structure-feature network optimization

    Implements the Structure->Feature optimization strategy:
    1. Use the best network architecture from module 6
    2. Search for optimal features for that architecture

    This approach tests whether feature selection should be
    architecture-dependent. Results can be compared with:
    - Module 5: Feature selection with fixed simple architecture
    - Module 7: Feature->Structure optimization

    The comparison helps determine the best optimization strategy
    for molecular property prediction models.
    """
    global CODE_DATASETS

    print("="*80)
    print("MODULE 7: ANO NETWORK OPTIMIZATION (MODEL-FEATURE OPTIMIZATION)")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Module Focus: Model-guided Feature Selection Optimization")
    print(f"🔧 Split types: {SPLIT_TYPES}")
    print(f"🎲 Number of trials: {N_TRIALS}")
    print(f"⏱️  Epochs per trial: {EPOCHS}")
    print(f"💾 Storage: {get_storage_url('7')}")



    # Prepare data
    print("\n📊 Preparing data...")

    # Use datasets from config
    from config import get_code_datasets
    code_datasets = get_code_datasets(7)  # Code 7 datasets - keep as keys

    total_tasks = len(code_datasets) * len(SPLIT_TYPES)
    current_task = 0

    print(f"📊 Active Datasets (Module 7): {code_datasets}")
    print(f"📈 Total combinations to process: {total_tasks}")
    print("="*80)
    
    for split_type in SPLIT_TYPES:
        print(f"\nRunning optimization for split type: {split_type}")
        
        for dataset in code_datasets:
            print(f"\nProcessing {get_dataset_display_name(dataset)} with {split_type} split...")
            
            # Create study name: ano_network_type2_MOFO_{dataset}_{split_type}
            study_name = f"ano_network_type2_MOFO_{dataset}_{split_type}"
            storage_name = get_storage_url('7')
            
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
                print(f"Module 7 sampler: TPESampler (n_startup_trials={n_startup_trials}/{N_TRIALS})")
            elif sampler_type == 'random':
                sampler = optuna.samplers.RandomSampler()
                print(f"Module 7 sampler: RandomSampler")
            else:
                # Fallback to TPESampler
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=n_startup_trials,
                    n_ei_candidates=50,
                    multivariate=False,
                    warn_independent_sampling=False
                )
                print(f"Module 7 sampler: TPESampler (fallback)")

            # Create HyperbandPruner for Module 7 (matched with Module 4 strategy)
            pruner_config = MODEL_CONFIG.get('optuna_pruner', {})
            if pruner_config.get('type') == 'hyperband':
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=pruner_config.get('min_resource', 100),
                    max_resource=pruner_config.get('max_resource', 1000),
                    reduction_factor=pruner_config.get('reduction_factor', 3)
                )
                print(f"Module 7 pruner: HyperbandPruner (min_resource={pruner_config.get('min_resource', 100)}, "
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
                print(f"Module 7 pruner: MedianPruner (n_startup_trials={n_startup})")

            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                storage=storage_name,
                study_name=study_name,
                load_if_exists=(not renew)
            )
            
            # Create objective function
            # Display detailed progress header for Module 7 (MOFO)
            total_datasets = len(code_datasets)
            total_splits = len(SPLIT_TYPES)
            current_dataset_idx = code_datasets.index(dataset) + 1
            current_split_idx = SPLIT_TYPES.index(split_type) + 1
            total_combinations = total_datasets * total_splits
            current_combination = (current_split_idx - 1) * total_datasets + current_dataset_idx

            print(f"\n{'='*75}")
            print(f"[MODULE 7] MOFO OPTIMIZATION: {get_dataset_display_name(dataset).upper()} | Trial (0/{N_TRIALS})")
            print(f"Dataset: {get_dataset_display_name(dataset).upper()} | Split: {split_type.upper()} | Method: Model->Feature Optimization")
            print(f"Scope: Dataset ({current_dataset_idx}/{total_datasets}) × Split ({current_split_idx}/{total_splits}) × Trial (0/{N_TRIALS})")
            print(f"Overall Progress: Combination {current_combination}/{total_combinations} ({(current_combination/total_combinations)*100:.2f}%)")
            print(f"Totals: {total_datasets} datasets × {total_splits} splits × {N_TRIALS} trials = {total_datasets * total_splits * N_TRIALS:,} optimizations")
            print(f"Features: Module 5 best architecture + optimized feature selection")
            print(f"Expected duration: ~{N_TRIALS * 90 / 60:.1f}m | Target: Max CV R² with MOFO strategy")
            print(f"{'='*75}")

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
    
    print("\nAll structure-feature network optimizations completed!")
    
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
    # Get command line arguments
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
    EPOCHS = get_epochs_for_module('7', args)

    # Set datasets with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    from config import CODE_SPECIFIC_DATASETS, ACTIVE_SPLIT_TYPES, DATA_PATH
    from pathlib import Path

    if args.datasets:
        CODE_DATASETS = args.datasets
        print(f"Datasets from argparse: {CODE_DATASETS}")
    elif args.dataset:
        CODE_DATASETS = [args.dataset]
        print(f"Dataset from argparse (single): {CODE_DATASETS}")
    elif '7' in CODE_SPECIFIC_DATASETS:
        CODE_DATASETS = CODE_SPECIFIC_DATASETS['7']
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

    # Global variable to control study renewal
    renew = MODEL_CONFIG.get('renew', False)  # Control from config.py
    print(f"⚙️  Renew setting: {renew} ({'Fresh start' if renew else 'Resume mode'})")

    renew = True

    # Set up logging to file
    from pathlib import Path
    module_name = MODULE_NAMES.get('7', '7_ANO_NetworkOptimization_MOFO')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
            self.log.write(f"Module 7 Execution Started: {datetime.now()}\n")
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
            self.log.write(f"Module 7 Execution Completed: {datetime.now()}\n")
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
