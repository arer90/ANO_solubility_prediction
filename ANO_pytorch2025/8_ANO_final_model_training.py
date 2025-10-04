#!/usr/bin/env python3
"""
ANO Final Model Training - ANO Framework Module 8
=================================================

PURPOSE:
Module 8 trains final production models using the best configurations
from Modules 4-7, then compares their performance to select the best approach.

KEY FEATURES:
1. **Best Configuration Loading**: Retrieves optimal settings from each module
2. **Final Training**: Trains with more epochs (1000) for production
3. **Model Comparison**: Evaluates FO vs MO vs FOMO vs MOFO strategies
4. **Model Selection**: Automatically selects best performing approach
5. **Production Models**: Saves final models for Module 9 testing
6. **Comprehensive Metrics**: R², RMSE, MAE, Pearson, Spearman correlations

RECENT UPDATES (2024):
- Removed StandardScaler (using raw features)
- Fixed model loading for both SimpleDNN (FO) and FlexibleDNNModel (MO/FOMO/MOFO)
- Saves models in both result/ and save_model/ directories
- Default epochs: 1000 (from config.py module_epochs['8'])
- Added model info JSON for proper loading

ANO STRATEGY COMPARISON:
- FO (Module 4): Optimized features + SimpleDNN
- MO (Module 5): All fingerprints + optimized architecture
- FOMO (Module 6): Optimized features → optimized architecture
- MOFO (Module 7): Optimized architecture → optimized features

MODEL ARCHITECTURE:
- FO: SimpleDNN with fixed [1024, 496] layers
- MO/FOMO/MOFO: FlexibleDNNModel with Optuna-optimized architecture

OUTPUT STRUCTURE:
result/8_final_model/
├── {dataset}_{split}/
│   ├── FO_model.pt
│   ├── MO_model.pt
│   ├── FOMO_model.pt
│   ├── MOFO_model.pt
│   ├── model_info.json
│   └── performance_comparison.json
├── plots/
│   ├── module_performance_comparison.png
│   └── performance_heatmap.png
└── best_model_selection.json

USAGE:
python 8_ANO_final_model_training.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --module: Specific module (FO/MO/FOMO/MOFO)
  --epochs: Override epochs (default from config: 1000)
"""

import os
from datetime import datetime
import json
import random
import argparse
import subprocess
import pickle
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
from config import MODULE_NAMES

# Performance profiling
# from extra_code.performance_profiler import profiler  # Commented out - module not available
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler  # Used for AD analysis
from config import get_epochs_for_module
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from rdkit import Chem
import sys
import warnings

warnings.filterwarnings('ignore')

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")

# Import centralized DNN models from ano_feature_selection
from extra_code.ano_feature_selection import SimpleDNN, FlexibleDNNModel

# Add both DNN models to safe globals for PyTorch 2.6 compatibility
torch.serialization.add_safe_globals([SimpleDNN, FlexibleDNNModel])

# Also make both models available as __main__ for subprocess compatibility
import __main__
__main__.SimpleDNN = SimpleDNN
__main__.FlexibleDNNModel = FlexibleDNNModel

# Import descriptor selection functions and fingerprint loader
from extra_code.ano_feature_selection import (
    convert_params_to_selection,
    selection_data_descriptor_compress
)
from extra_code.mol_fps_maker import get_fingerprints_cached

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Import configuration (like modules 4-7)
from config import (
    MODEL_CONFIG,
    CODE_SPECIFIC_DATASETS,
    CODE_SPECIFIC_FINGERPRINTS,
    ACTIVE_SPLIT_TYPES,
    DATASETS,
    DATASET_DISPLAY_NAMES,
    DATA_PATH,
    RESULT_PATH,
    load_best_configurations,  # Global configuration functions
    get_storage_url, get_database_info,
    get_code_datasets, get_code_fingerprints
)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Global epoch setting (highest priority)
# Set this to override both command-line and config.py settings
# EPOCHS will be set in main() with proper priority
EPOCHS = None

# Directories (use config.py paths)
OUTPUT_DIR = Path(RESULT_PATH) / '8_ANO_final_model_training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot directory
PLOT_DIR = OUTPUT_DIR / 'plots'
PLOT_DIR.mkdir(exist_ok=True)

# Data directory
DATA_DIR = Path(DATA_PATH)

# Activation functions mapping
ACTIVATION_FUNCTIONS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,  # Swish
    'Mish': nn.Mish,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'Softplus': nn.Softplus,
    'PReLU': nn.PReLU,
    'SELU': nn.SELU,
    'ReLU6': nn.ReLU6,
    'Hardswish': nn.Hardswish,
    'Hardsigmoid': nn.Hardsigmoid
}

# Optimizers mapping
OPTIMIZERS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'AdamW': optim.AdamW,
    'Adamax': optim.Adamax,
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'ASGD': optim.ASGD,
    'NAdam': optim.NAdam,
    'RAdam': optim.RAdam,
    'LBFGS': optim.LBFGS
}

# Learning rate schedulers mapping
SCHEDULERS = {
    'StepLR': lr_scheduler.StepLR,
    'MultiStepLR': lr_scheduler.MultiStepLR,
    'ExponentialLR': lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
    'CyclicLR': lr_scheduler.CyclicLR,
    'OneCycleLR': lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
    'LinearLR': lr_scheduler.LinearLR,
    'PolynomialLR': lr_scheduler.PolynomialLR
}

# Note: Using centralized DNN models from ano_feature_selection.py
# FO (Module 4): SimpleDNN
# MO, FOMO, MOFO (Modules 5,6,7): FlexibleDNNModel

# Dataset name mapping
CODE_DATASETS = get_code_datasets(8)  # Code 8
CODE_FINGERPRINTS = get_code_fingerprints(8)  # Code 8
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
    from extra_code.mol_fps_maker import get_fingerprints_cached

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
            train_mols_filtered, dataset_short.upper(), split_type, 'train', module_name='8_ANO_final_model_training'
        )
        test_morgan, test_maccs, test_avalon = get_fingerprints_cached(
            test_mols_filtered, dataset_short.upper(), split_type, 'test', module_name='8_ANO_final_model_training'
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

def load_data_and_features(dataset, split_type='rm', module='FO'):
    """Load data and prepare features based on module requirements"""
    
    print(f"  Loading {dataset.upper()}-{split_type} for {module}...")
    
    # Load split data from train/test folders (like modules 4,5,6,7)
    # Map dataset names to their corresponding file names
    dataset_file_map = {
        'ws': 'ws496_logS',
        'de': 'delaney-processed',
        'lo': 'Lovric2020_logS0',
        'hu': 'huusk'
    }
    
    dataset_filename = dataset_file_map.get(dataset, dataset)
    train_path = DATA_DIR / 'train' / split_type / f'{split_type}_{dataset_filename}_train.csv'
    test_path = DATA_DIR / 'test' / split_type / f'{split_type}_{dataset_filename}_test.csv'
    
    if not train_path.exists() or not test_path.exists():
        print(f"    Dataset files not found: {train_path} or {test_path}")
        return None, None, None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Get SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'isomeric_smiles', 'target', 'logS', 'LogS', 'logs', 'logS0', 'measured log(solubility:mol/L)', 'y', 'Solubility']:
        if col in train_df.columns:
            smiles_col = col
            break
    
    if not smiles_col:
        print(f"    No SMILES column found")
        return None, None, None, None
    
    # Dataset-specific target column mapping
    target_mapping = {
        'ws': 'exp',
        'de': 'measured log solubility in mols per litre',
        'lo': 'logS0',
        'hu': 'Solubility'
    }
    
    target_col = target_mapping.get(dataset)
    
    # If mapping doesn't work, try common column names
    if not target_col or target_col not in train_df.columns:
        for col in ['exp', 'logS', 'LogS', 'logs', 'logS0', 'measured log(solubility:mol/L)', 'target', 'Solubility', 'y']:
            if col in train_df.columns:
                target_col = col
                break
    
    if not target_col or target_col not in train_df.columns:
        print(f"    No target column found. Available columns: {train_df.columns.tolist()[:10]}")
        return None, None, None, None
    
    # Get SMILES and targets
    train_smiles = train_df[smiles_col].values
    test_smiles = test_df[smiles_col].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    # Convert SMILES to molecules (like Module 4)
    train_mols = [Chem.MolFromSmiles(str(smi)) for smi in train_smiles]
    test_mols = [Chem.MolFromSmiles(str(smi)) for smi in test_smiles]
    
    # Filter valid molecules  
    train_valid = [(mol, y) for mol, y in zip(train_mols, y_train) if mol is not None]
    test_valid = [(mol, y) for mol, y in zip(test_mols, y_test) if mol is not None]
    
    train_mols = [mol for mol, y in train_valid]
    y_train = np.array([y for mol, y in train_valid])
    test_mols = [mol for mol, y in test_valid]
    y_test = np.array([y for mol, y in test_valid])
    
    print(f"    Data split: {len(train_mols)} train, {len(test_mols)} test")
    
    # Use the existing fingerprint cache system from mol_fps_maker
    print(f"    Loading fingerprints from cache...")
    
    # Get fingerprints for train data (returns morgan, maccs, avalon)
    morgan_train, maccs_train, avalon_train = get_fingerprints_cached(
        train_mols,
        dataset_name=dataset,
        split_type=split_type,
        data_type='train',
        module_name='8_ANO_final_model_training'
    )

    # Get fingerprints for test data
    morgan_test, maccs_test, avalon_test = get_fingerprints_cached(
        test_mols,
        dataset_name=dataset,
        split_type=split_type,
        data_type='test',
        module_name='8_ANO_final_model_training'
    )
    
    # Combine ALL fingerprints (morgan + maccs + avalon)
    base_train = np.hstack([morgan_train, maccs_train, avalon_train])
    base_test = np.hstack([morgan_test, maccs_test, avalon_test])
    
    print(f"    Loaded ALL fingerprints: morgan(2048) + maccs(167) + avalon(512) = {base_train.shape[1]} features")
    
    # Prepare features based on module
    if module == 'FO':
        # Feature Optimization - use ALL fingerprints + selected descriptors from Module 4
        result_file = Path(RESULT_PATH) / f'4_ANO_FeatureOptimization_FO/{dataset}/{split_type}/{dataset}_{split_type}_FO_results.json'
        if not result_file.exists():
            print(f"Warning: FO results not found for {dataset}-{split_type}, skipping...")
            return None, None, None, None
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract selected descriptors from best_params
        if 'best_params' not in results:
            raise ValueError(f"No best_params found in FO results for {dataset}-{split_type}")
        
        best_params = results['best_params']
        
        # Convert params to selection format and get descriptors
        selection = convert_params_to_selection(best_params)

        # Step 1: Check if descriptor cache exists, if not create it
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
                    mols_3d=None
                )

            # Generate descriptors for test set
            if not os.path.exists(test_desc_file):
                print(f"  📊 Calculating test descriptors...")
                calculator.calculate_selected_descriptors(
                    test_mols, dataset_name=dataset, split_type=split_type, subset='test',
                    mols_3d=None
                )

            print(f"  ✅ Descriptors generated and cached successfully")

        # Step 2: Apply feature selection using cached descriptors
        # Combine train and test to ensure consistent descriptor dimensions
        all_mols = train_mols + test_mols
        all_base = np.vstack([base_train, base_test])

        # Use selection_data_descriptor_compress with cached descriptors
        X_all, selected_desc_names = selection_data_descriptor_compress(
            selection, all_base, all_mols, f"{dataset}-{split_type}",
            target_path=str(OUTPUT_DIR), save_res="np",
            mols_3d=None
        )

        # Split back into train and test
        n_train = len(train_mols)
        X_train = X_all[:n_train]
        X_test = X_all[n_train:]
        
        print(f"    FO: Using ALL fingerprints ({base_train.shape[1]}) + selected descriptors from Module 4")
        print(f"        Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            print(f"        WARNING: Train and test have different feature dimensions!")
        print(f"        Selected descriptors: {', '.join(selected_desc_names[:5]) if selected_desc_names else 'None'}{'...' if len(selected_desc_names) > 5 else ''}")
            
    elif module == 'MO':
        # Model Optimization - uses base fingerprints with optimized architecture
        X_train = base_train
        X_test = base_test
        print(f"    MO: Using base fingerprints with optimized architecture")
        
    elif module == 'FOMO':
        # Network Optimization Type 1 - Use FO results from Module 4 (all fingerprints + selected descriptors) with optimized structure
        # FOMO uses Module 4's FO feature results and Module 6's optimized network structure
        result_file = Path(RESULT_PATH) / f'4_ANO_FeatureOptimization_FO/{dataset}/{split_type}/{dataset}_{split_type}_FO_results.json'
        if not result_file.exists():
            print(f"Warning: FO results not found for FOMO {dataset}-{split_type}, skipping...")
            return None, None, None, None
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Extract selected descriptors from Module 4 FO results
        if 'best_params' not in results:
            raise ValueError(f"No best_params found in FO results for FOMO {dataset}-{split_type}")
        
        best_params = results['best_params']
        
        # Convert params to selection format and get descriptors
        selection = convert_params_to_selection(best_params)

        # Step 1: Check if descriptor cache exists, if not create it
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
                    mols_3d=None
                )

            # Generate descriptors for test set
            if not os.path.exists(test_desc_file):
                print(f"  📊 Calculating test descriptors...")
                calculator.calculate_selected_descriptors(
                    test_mols, dataset_name=dataset, split_type=split_type, subset='test',
                    mols_3d=None
                )

            print(f"  ✅ Descriptors generated and cached successfully")
        else:
            print(f"✅ Using cached descriptors from {dataset}/{split_type}")

        # Step 2: Apply feature selection using cached descriptors
        # Combine train and test to ensure consistent descriptor dimensions
        all_mols = train_mols + test_mols
        all_base = np.vstack([base_train, base_test])

        # Use selection_data_descriptor_compress with cached descriptors
        X_all, selected_desc_names = selection_data_descriptor_compress(
            selection, all_base, all_mols, f"{dataset}-{split_type}",
            target_path=str(OUTPUT_DIR), save_res="np",
            mols_3d=None
        )

        # Split back into train and test
        n_train = len(train_mols)
        X_train = X_all[:n_train]
        X_test = X_all[n_train:]
        
        print(f"    FOMO: Using ALL fingerprints ({base_train.shape[1]}) + descriptors from Module 4 FO")
        print(f"        Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        if X_train.shape[1] != X_test.shape[1]:
            print(f"        WARNING: Train and test have different feature dimensions!")
        print(f"        With optimized network structure from Module 6")
        
    elif module == 'MOFO':
        # Network Optimization - Fixed structure + optimized features
        # MOFO uses features optimized in Module 7
        result_file = Path(RESULT_PATH) / f'7_ANO_NetworkOptimization_MOFO/{dataset}/{split_type}/{dataset}_{split_type}_MOFO_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                no2_results = json.load(f)
            
            # MOFO stores selected_descriptors as a list
            selected_descriptors = no2_results.get('selected_descriptors', [])
            n_features_expected = no2_results.get('n_features', 2727)
            
            if selected_descriptors:
                print(f"    MOFO: Found {len(selected_descriptors)} selected descriptors")
                print(f"        Expected total features: {n_features_expected}")
                
                # Import the feature generation function used by MOFO
                from extra_code.ano_feature_search import search_data_descriptor_compress
                
                # Create a mock trial that returns the saved selections
                class MockTrial:
                    def __init__(self, selections):
                        self.selections = selections
                    
                    def suggest_categorical(self, name, choices):
                        # Return 1 if descriptor was selected, 0 otherwise
                        if name in self.selections:
                            return 1
                        return 0
                
                mock_trial = MockTrial(selected_descriptors)
                
                # Combine train and test for consistent processing
                all_mols = train_mols + test_mols
                all_base = np.vstack([base_train, base_test])

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

                # Generate features exactly as MOFO did during training
                X_all, actual_selected, _ = search_data_descriptor_compress(
                    mock_trial, all_base, all_mols, dataset, split_type,
                    target_path=str(OUTPUT_DIR), save_res="np", mols_3d=None
                )
                
                # Split back into train and test
                n_train = len(train_mols)
                X_train = X_all[:n_train]
                X_test = X_all[n_train:]
                
                print(f"        Generated features: Train {X_train.shape}, Test {X_test.shape}")
                
            else:
                # No descriptors selected, use base fingerprints
                X_train = base_train
                X_test = base_test
                print(f"    MOFO: No descriptors selected, using base fingerprints")
        else:
            print(f"    MOFO results not found, using base fingerprints")
            X_train = base_train
            X_test = base_test
    else:
        X_train = base_train
        X_test = base_test
    
    print(f"    Final features shape: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# @profiler.profile_function(module_name="8_ANO_final_model_training", save_args=True)
def get_model_params(dataset, split_type, module):
    """3-Stage Fallback System: result files -> database -> default DNN + chemical descriptors"""

    print(f"  🔍 Stage 1: Loading optimized parameters from result files...")

    # Stage 1: Try to load from result files
    result_params = _load_from_result_files(dataset, split_type, module)
    if result_params:
        print(f"    ✅ Stage 1 Success: Loaded from result files")
        return result_params

    print(f"  🔍 Stage 2: Loading optimized parameters from database...")

    # Stage 2: Try to load from Optuna database
    db_params = _load_from_database(dataset, split_type, module)
    if db_params:
        print(f"    ✅ Stage 2 Success: Loaded from database")
        return db_params

    # Check if we should skip this combination
    print(f"    ⚠️  No parameters found for {dataset}-{split_type} {module}")
    return None  # Return None instead of default params to skip


def _load_from_result_files(dataset, split_type, module):
    """Stage 1: Load parameters from result files"""
    try:
        if module == 'FO':
            # Module 4: Feature Optimization results
            result_file = Path(RESULT_PATH) / f'4_ANO_FeatureOptimization_FO/{dataset}/{split_type}/{dataset}_{split_type}_FO_results.json'
        elif module == 'MO':
            # Module 5: Model Optimization results
            result_file = Path(RESULT_PATH) / f'5_ANO_ModelOptimization_MO/{dataset}/{split_type}/{dataset}_{split_type}_MO_results.json'
        elif module == 'FOMO':
            # Module 6: Network Optimization Type1 results
            result_file = Path(RESULT_PATH) / f'6_ANO_NetworkOptimization_FOMO/{dataset}/{split_type}/{dataset}_{split_type}_FOMO_results.json'
        elif module == 'MOFO':
            # Module 7: Network Optimization results
            result_file = Path(RESULT_PATH) / f'7_ANO_NetworkOptimization_MOFO/{dataset}/{split_type}/{dataset}_{split_type}_MOFO_results.json'
        else:
            return None

        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)

            if 'best_params' in results:
                params = results['best_params']
                return _convert_to_model_params(params)
            elif 'best_config' in results and 'structure' in results['best_config']:
                # New format with optimization scores
                structure = results['best_config']['structure']
                return _convert_to_model_params(structure)

    except Exception as e:
        print(f"      ❌ Stage 1 failed: {e}")

    return None


def _load_from_database(dataset, split_type, module):
    """Stage 2: Load parameters from Optuna database"""
    try:
        import optuna

        db_path = Path(RESULT_PATH) / f'optuna_studies/{module.lower()}_study.db'
        if not db_path.exists():
            return None

        # Use Optuna to access the study
        study_name = f"{module}_{dataset}_{split_type}"
        storage = get_storage_url('8')

        study = optuna.load_study(study_name=study_name, storage=storage)

        if study.best_trial:
            best_params = study.best_trial.params
            return _convert_to_model_params(best_params)

    except Exception as e:
        print(f"      ❌ Stage 2 failed: {e}")

    return None


def _get_default_dnn_params():
    """Stage 3: Default DNN configuration with chemical descriptors support"""
    return {
        'n_layers': 3,
        'hidden_dims': [1024, 512, 256, 1],  # Suitable for large chemical descriptor input
        'activation': 'ReLU',
        'dropout_rate': 0.3,  # Higher dropout for complex descriptors
        'optimizer': 'Adam',
        'lr': 0.001,
        'batch_size': 32,
        'use_batch_norm': True,  # Batch norm helps with descriptor normalization
        'weight_init': 'xavier_uniform',
        'scheduler': 'ReduceLROnPlateau',  # Adaptive learning rate
        'weight_decay': 1e-5  # Light regularization
    }


def _convert_to_model_params(params):
    """Convert optimization parameters to model parameters format"""
    # For Module 4 (FO) - feature selection, use default DNN
    if all(isinstance(v, (int, float)) and k not in ['n_layers', 'hidden_dim_0', 'activation', 'dropout_rate', 'optimizer'] for k, v in params.items()):
        # This is feature selection params from Module 4
        return {
            'n_layers': 2,
            'hidden_dims': [1024, 496, 1],
            'activation': 'ReLU',
            'dropout_rate': 0.2,
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 32,
            'use_batch_norm': False,
            'weight_init': 'xavier_uniform',
            'scheduler': None,
            'weight_decay': 0.0
        }

    # Extract hidden dimensions for Module 5, 6, 7
    hidden_dims = []
    n_layers = params.get('n_layers', 3)

    # Check for hidden_dim_X format (Module 5, 6, 7)
    for i in range(n_layers):
        if f'hidden_dim_{i}' in params:
            hidden_dims.append(params[f'hidden_dim_{i}'])

    # Add final output layer if not present
    if hidden_dims and hidden_dims[-1] != 1:
        hidden_dims.append(1)

    # If no hidden_dims found, use default based on n_layers
    if not hidden_dims:
        hidden_dims = [256, 128, 64, 1] if n_layers >= 3 else [512, 256, 1]

    # Default values for missing parameters
    return {
        'n_layers': n_layers,
        'hidden_dims': hidden_dims,
        'activation': params.get('activation', 'ReLU'),
        'dropout_rate': params.get('dropout_rate', 0.2),
        'optimizer': params.get('optimizer', 'Adam'),
        'lr': params.get('learning_rate', params.get('lr', 0.001)),
        'batch_size': params.get('batch_size', 32),
        'use_batch_norm': params.get('use_batch_norm', False),
        'weight_init': params.get('weight_init', 'xavier_uniform'),
        'scheduler': params.get('scheduler', None),
        'weight_decay': params.get('weight_decay', 0.0)
    }

def create_scatter_plot(y_true, y_pred, title, save_path):
    """
    Create enhanced scatter plot with detailed R² analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Calculate comprehensive metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate additional statistics
    residuals = y_true - y_pred
    std_residuals = residuals / (np.std(residuals) + 1e-8)  # Avoid division by zero
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    spearman_r = scipy.stats.spearmanr(y_true, y_pred)[0]
    
    # Calculate explained variance
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    explained_var = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 1. Main scatter plot (top left)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=15, c=np.abs(residuals), 
                         cmap='coolwarm', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction', alpha=0.8)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(y_true), p(sorted(y_true)), "g-", alpha=0.8, lw=2,
             label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
    
    # Add 95% prediction interval
    predict_mean_se = scipy.stats.sem(residuals) if len(residuals) > 1 else 0
    margin = 1.96 * predict_mean_se
    ax1.fill_between(sorted(y_true), 
                     p(sorted(y_true)) - margin, 
                     p(sorted(y_true)) + margin, 
                     alpha=0.2, color='green', label='95% CI')
    
    # Colorbar for residuals
    plt.colorbar(scatter, ax=ax1, label='|Residuals|')
    
    # Enhanced metrics text box
    n = len(y_true)
    adj_r2 = 1 - (1-r2)*(n-1)/(n-2) if n > 2 else r2
    metrics_text = (f'R² Score: {r2:.5f}\n'
                   f'Adj. R²: {adj_r2:.5f}\n'
                   f'Pearson r: {pearson_r:.5f}\n'
                   f'Spearman ρ: {spearman_r:.5f}\n'
                   f'RMSE: {rmse:.4f}\n'
                   f'MAE: {mae:.4f}\n'
                   f'MSE: {mse:.4f}\n'
                   f'Explained Var: {explained_var:.5f}\n'
                   f'N samples: {n}')
    
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction vs Actual with R² Analysis', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Residual plot (top right)
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.6, s=10, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    std_res = np.std(residuals)
    ax2.axhline(y=std_res, color='orange', linestyle=':', lw=1, label='±1 STD')
    ax2.axhline(y=-std_res, color='orange', linestyle=':', lw=1)
    ax2.axhline(y=2*std_res, color='red', linestyle=':', lw=1, alpha=0.5, label='±2 STD')
    ax2.axhline(y=-2*std_res, color='red', linestyle=':', lw=1, alpha=0.5)
    
    # Add residual statistics
    res_stats = (f'Mean: {np.mean(residuals):.4f}\n'
                f'Std: {np.std(residuals):.4f}\n'
                f'Skew: {scipy.stats.skew(residuals):.4f}\n'
                f'Kurt: {scipy.stats.kurtosis(residuals):.4f}')
    ax2.text(0.05, 0.95, res_stats, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Q-Q plot (bottom left)
    ax3 = axes[1, 0]
    scipy.stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax3.set_ylabel('Sample Quantiles', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Histogram of residuals with KDE (bottom right)
    ax4 = axes[1, 1]
    n_bins = min(30, max(10, len(residuals)//10))
    n_hist, bins, patches = ax4.hist(residuals, bins=n_bins, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black')
    
    # Add KDE if enough samples
    if len(residuals) > 10:
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            ax4.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        except:
            pass  # Skip KDE if it fails
    
    # Add normal distribution for comparison
    mu, std = np.mean(residuals), np.std(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax4.plot(x_range, scipy.stats.norm.pdf(x_range, mu, std), 'g--', lw=2, 
             label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
    
    # Shapiro-Wilk test for normality
    if 3 <= len(residuals) < 5000:  # Shapiro-Wilk has sample size limits
        try:
            shapiro_stat, shapiro_p = scipy.stats.shapiro(residuals)
            normality_text = f'Shapiro-Wilk test:\nStatistic: {shapiro_stat:.5f}\np-value: {shapiro_p:.5f}\n'
            normality_text += 'Normally distributed' if shapiro_p > 0.05 else 'Not normally distributed'
        except:
            normality_text = 'Normality test failed'
    elif len(residuals) >= 5000:
        # Use Kolmogorov-Smirnov test for large samples
        ks_stat, ks_p = scipy.stats.kstest(residuals, 'norm', args=(mu, std))
        normality_text = f'K-S test:\nStatistic: {ks_stat:.5f}\np-value: {ks_p:.5f}\n'
        normality_text += 'Normally distributed' if ks_p > 0.05 else 'Not normally distributed'
    else:
        normality_text = 'Too few samples for\nnormality test'
    
    ax4.text(0.65, 0.95, normality_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax4.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    # Save plot with high quality
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return r2, rmse, mae

# @profiler.profile_function(module_name="8_ANO_final_model_training", save_args=False)
def train_and_evaluate(X_train, X_test, y_train, y_test, model_params, epochs=MODEL_CONFIG['epochs'], module='MO'):
    """
    Train and evaluate model without CV - just train on full training data using subprocess
    """
    temp_path = Path('temp_models')
    temp_path.mkdir(exist_ok=True)

    # Save training data
    train_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_test,  # Use test as validation for this simple version
        'y_val': y_test,
        'X_test': X_test,
        'y_test': y_test,
        'architecture': {
            'n_layers': model_params['n_layers'],
            'hidden_dims': model_params['hidden_dims']
        },
        'dropout_rate': model_params.get('dropout_rate', 0.2),
        'epochs': epochs,
        'batch_size': 32,
        'lr': model_params.get('lr', 0.001)
    }

    data_path = temp_path / "module8_full_train_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(train_data, f)

    # Run subprocess
    model_path = Path("save_model") / "full_model.pt"
    cmd = [
        sys.executable,
        "extra_code/learning_process_pytorch_torchscript.py",
        str(data_path),
        str(model_path)
    ]

    # Use environment with visible GPUs
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for subprocess
    OS_TYPE = platform.system()

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
        print(f"        Training subprocess timeout (600s)")
        return None, {'test_r2': -999, 'test_rmse': 999}

    if result.returncode != 0:
        print(f"        Training failed with return code {result.returncode}")
        print(f"        STDERR: {result.stderr}")
        print(f"        STDOUT: {result.stdout}")
        return None, {'test_r2': -999, 'test_rmse': 999}

    # Parse metrics from subprocess output (last line contains r2,rmse,mse,mae)
    try:
        output_lines = result.stdout.strip().split('\n')
        # Find the metrics line (format: r2,rmse,mse,mae)
        metrics_line = None
        predictions_line = None
        for line in reversed(output_lines):
            if line.startswith('PREDICTIONS:'):
                predictions_line = line
            elif ',' in line and not line.startswith('[') and not 'Epoch' in line and not line.startswith('PREDICTIONS:'):
                try:
                    # Try to parse as metrics
                    parts = line.split(',')
                    if len(parts) == 4:
                        float(parts[0])  # Test if it's a number
                        metrics_line = line
                        break
                except:
                    continue

        if metrics_line:
            r2, rmse, mse, mae = map(float, metrics_line.split(','))
            print(f"        Subprocess metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            # Parse predictions if available
            y_pred = np.zeros(len(y_test))  # Default placeholder
            if predictions_line:
                try:
                    pred_str = predictions_line.split('PREDICTIONS:')[1]
                    y_pred = np.array([float(p) for p in pred_str.split(',')])
                    if len(y_pred) != len(y_test):
                        print(f"        Warning: Prediction count mismatch: {len(y_pred)} vs {len(y_test)}")
                        y_pred = np.zeros(len(y_test))
                    else:
                        print(f"        Successfully parsed {len(y_pred)} predictions")
                except Exception as e:
                    print(f"        Warning: Failed to parse predictions: {e}")
                    y_pred = np.zeros(len(y_test))

            # Load the trained model before cleaning up
            model = None
            try:
                # Try to load the model
                if model_path.exists():
                    model = torch.jit.load(str(model_path), map_location=DEVICE)
                else:
                    model_path_pth = model_path.with_suffix('.pth')
                    if model_path_pth.exists():
                        checkpoint = torch.load(str(model_path_pth), map_location=DEVICE, weights_only=False)
                        if 'model' in checkpoint:
                            model = checkpoint['model']
                        else:
                            model = checkpoint

                if model is not None:
                    model.eval()
                    print(f"        Successfully loaded model from subprocess")
                else:
                    print(f"        Warning: Could not load model from subprocess")

            except Exception as e:
                print(f"        Error loading model from subprocess: {e}")
                model = None

            # Clean up temp files
            try:
                data_path.unlink()
                if model_path.exists():
                    model_path.unlink()
                model_path_pth = model_path.with_suffix('.pth')
                if model_path_pth.exists():
                    model_path_pth.unlink()
            except:
                pass

            # Return model and metrics
            return model, {'test_r2': r2, 'test_rmse': rmse, 'test_mae': mae, 'train_r2': r2, 'y_pred': y_pred}
        else:
            print(f"        Could not parse metrics from subprocess output")
            print(f"        Last 5 lines of output:")
            for line in output_lines[-5:]:
                print(f"          {line}")
            return None, {'test_r2': -999, 'test_rmse': 999, 'test_mae': 999, 'train_r2': -999, 'y_pred': np.zeros(len(y_test))}

    except Exception as e:
        print(f"        Error parsing subprocess output: {e}")
        import traceback
        print(f"        Full traceback: {traceback.format_exc()}")
        return None, {'test_r2': -999, 'test_rmse': 999, 'test_mae': 999, 'train_r2': -999, 'y_pred': np.zeros(len(y_test))}

# @profiler.profile_function(module_name="8_ANO_final_model_training", save_args=False)
def train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, epochs=100, n_folds=5):
    """
    Type1: Research Pipeline - CV methodology
    Performs K-fold CV using training data only and predicts independent test set per fold

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_params: Model parameters
        epochs: Number of training epochs
        n_folds: Number of CV folds

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds} on Train + Test prediction per fold")

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    test_r2_scores = []  # Test R² scores from each fold
    test_rmse_scores = []  # Test RMSE from each fold
    test_mse_scores = []  # Test MSE from each fold
    test_mae_scores = []  # Test MAE from each fold

    temp_path = Path("save_model")
    temp_path.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Save training data for subprocess
        fold_data = {
            'X_train': X_train_fold,
            'y_train': y_train_fold,
            'X_val': X_test,  # Type1: predict independent test set each fold
            'y_val': y_test,
            'X_test': X_test,  # Also for compatibility
            'y_test': y_test,
            'architecture': {
                'n_layers': model_params['n_layers'],
                'hidden_dims': model_params['hidden_dims']
            },
            'dropout_rate': model_params.get('dropout_rate', 0.2),
            'epochs': epochs,
            'batch_size': 32,
            'lr': model_params.get('lr', 0.001)
        }

        fold_data_path = temp_path / f"module8_type1_fold{fold}_data.pkl"
        with open(fold_data_path, 'wb') as f:
            pickle.dump(fold_data, f)

        # Run subprocess training
        cmd = [
            sys.executable,
            "extra_code/training_subprocess_final.py",
            str(fold_data_path),
            "save_model/full_model.pt"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results - learning_process_pytorch_torchscript.py outputs: r2,rmse,mse,mae
        metrics_found = False
        try:
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                # Look for comma-separated metrics line (r2,rmse,mse,mae)
                if ',' in line and not ':' in line:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            r2 = float(parts[0])
                            rmse = float(parts[1])
                            mse = float(parts[2])
                            mae = float(parts[3])
                            test_r2_scores.append(r2)
                            test_rmse_scores.append(rmse)
                            test_mse_scores.append(mse)
                            test_mae_scores.append(mae)
                            print(f"    Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
                            metrics_found = True
                            break
                    except (ValueError, IndexError):
                        continue

            if not metrics_found:
                # If no valid metrics found, report error but don't append zeros
                print(f"    Fold {fold+1}: No valid metrics found in output")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"    Fold {fold+1}: Error parsing metrics - {e}")

        # Cleanup
        if fold_data_path.exists():
            fold_data_path.unlink()

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

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, epochs=100, n_folds=5):
    """
    Type 2: Production Pipeline - Train/Test Split + CV
    Performs CV on pre-split training data and evaluates on independent test set

    Args:
        X_train: Training data features (pre-split)
        y_train: Training data targets (pre-split)
        X_test: Test data features (pre-split)
        y_test: Test data targets (pre-split)
        model_params: Model parameters
        epochs: Number of training epochs
        n_folds: Number of folds for CV

    Returns:
        Tuple of (cv_stats, final_test_metrics)
    """
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()

    # Step 1: CV on training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    test_r2_scores = []
    test_rmse_scores = []
    test_mse_scores = []
    test_mae_scores = []

    temp_path = Path("save_model")
    temp_path.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        # Save training data for subprocess
        fold_data = {
            'X_train': X_train_fold,
            'y_train': y_train_fold,
            'X_val': X_val_fold,
            'y_val': y_val_fold,
            'X_test': X_val_fold,  # For compatibility
            'y_test': y_val_fold,
            'architecture': {
                'n_layers': model_params['n_layers'],
                'hidden_dims': model_params['hidden_dims']
            },
            'dropout_rate': model_params.get('dropout_rate', 0.2),
            'epochs': epochs,
            'batch_size': 32,
            'lr': model_params.get('lr', 0.001)
        }

        fold_data_path = temp_path / f"module8_type2_fold{fold}_data.pkl"
        with open(fold_data_path, 'wb') as f:
            pickle.dump(fold_data, f)

        # Run subprocess training
        cmd = [
            sys.executable,
            "extra_code/training_subprocess_final.py",
            str(fold_data_path),
            "save_model/full_model.pt"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        try:
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                if 'Final test metrics:' in line:
                    # Extract metrics from: "Final test metrics: R2=0.1234, RMSE=0.5678, MSE=0.3456, MAE=0.2345"
                    parts = line.split(':')[1].strip().split(', ')
                    r2 = float(parts[0].split('=')[1])
                    rmse = float(parts[1].split('=')[1])
                    mse = float(parts[2].split('=')[1])
                    mae = float(parts[3].split('=')[1])
                    test_r2_scores.append(r2)
                    test_rmse_scores.append(rmse)
                    test_mse_scores.append(mse)
                    test_mae_scores.append(mae)
                    print(f"    CV Fold {fold+1}: R2={r2:.4f}, RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")
                    break
        except Exception as e:
            print(f"    CV Fold {fold+1}: Error parsing metrics - {e}")
            test_r2_scores.append(0.0)
            test_rmse_scores.append(0.0)
            test_mse_scores.append(0.0)
            test_mae_scores.append(0.0)

        # Cleanup
        if fold_data_path.exists():
            fold_data_path.unlink()

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

    fold_data_final = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_test,
        'y_val': y_test,
        'X_test': X_test,
        'y_test': y_test,
        'architecture': {
            'n_layers': model_params['n_layers'],
            'hidden_dims': model_params['hidden_dims']
        },
        'dropout_rate': model_params.get('dropout_rate', 0.2),
        'epochs': epochs,
        'batch_size': 32,
        'lr': model_params.get('lr', 0.001)
    }

    fold_data_path_final = temp_path / "module8_type2_final_data.pkl"
    with open(fold_data_path_final, 'wb') as f:
        pickle.dump(fold_data_final, f)

    # Run subprocess training
    cmd_final = [
        sys.executable,
        "extra_code/training_subprocess_final.py",
        str(fold_data_path_final),
        "save_model/full_model.pt"
    ]

    result_final = subprocess.run(cmd_final, capture_output=True, text=True)

    # Parse final test results
    test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
    try:
        output_lines = result_final.stdout.strip().split('\n')
        for line in reversed(output_lines):
            if 'Final test metrics:' in line:
                # Extract metrics from: "Final test metrics: R2=0.1234, RMSE=0.5678, MSE=0.3456, MAE=0.2345"
                parts = line.split(':')[1].strip().split(', ')
                test_r2 = float(parts[0].split('=')[1])
                test_rmse = float(parts[1].split('=')[1])
                test_mse = float(parts[2].split('=')[1])
                test_mae = float(parts[3].split('=')[1])
                print(f"    [TYPE2-Production] Final Test: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MSE={test_mse:.4f}, MAE={test_mae:.4f}")
                break
    except Exception as e:
        print(f"    [TYPE2-Production] Error parsing final test metrics: {e}")

    # Cleanup
    if fold_data_path_final.exists():
        fold_data_path_final.unlink()

    final_test_metrics = {
        'test_r2': test_r2, 'test_rmse': test_rmse, 'test_mse': test_mse, 'test_mae': test_mae
    }

    return cv_stats, final_test_metrics

def train_model_cv_both_types(X_train, y_train, X_test, y_test, model_params, epochs=100, n_folds=5):
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
        epochs: Number of training epochs
        n_folds: Number of CV folds

    Returns:
        Dictionary with Type1 and Type2 results
    """
    print(f"\n=== Running Both CV Types ===")

    results = {}

    # Type1: Research Pipeline - K-fold CV on training data + test prediction per fold
    print(f"    [TYPE1-Research] Research Pipeline - CV-5")
    type1_results = train_model_cv_type1(X_train, y_train, X_test, y_test, model_params, epochs, n_folds)
    results['type1'] = type1_results

    # Type2: Production Pipeline - Train/Val split + independent test prediction
    print(f"    [TYPE2-Production] Production Pipeline - CV-5 on Train + Final Test")
    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, epochs, n_folds)

    type2_cv_stats, type2_final_metrics = train_model_cv_type2(X_train, y_train, X_test, y_test, model_params, epochs, n_folds)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_and_evaluate_with_cv_subprocess(X_train, X_test, y_train, y_test, model_params, epochs=MODEL_CONFIG['epochs'],
                                         dataset_name="", module_name="", output_dir=None):
    """
    Enhanced CV function with dual CV types (Type1 + Type2) using subprocess for isolation and speed

    This function now implements both CV methodologies:
    - Type1 (Research): Full dataset K-fold CV for research/paper results
    - Type2 (Production): Train/Test split + CV on train for production pipeline

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target values
        y_test: Test target values
        model_params: Model parameters dict
        epochs: Training epochs
        dataset_name: Dataset name for logging
        module_name: Module name for logging
        output_dir: Output directory

    Returns:
        Dict with comprehensive results from both CV types
    """
    print(f"    Running Dual CV for {module_name} (subprocess mode for speed)...")

    # Get CV folds from config
    n_folds = MODEL_CONFIG['cv_folds']

    # Combine train and test for Type1 CV
    # Execute dual CV methodology with corrected approach (no data leakage)
    dual_results = train_model_cv_both_types(
        X_train, y_train, X_test, y_test, model_params, epochs, n_folds
    )

    # Extract results for compatibility with existing code
    type1_results = dual_results['type1']  # Type1 now returns direct format
    type2_results = dual_results['type2']

    print(f"\n=== Final Module 8 Dual CV Results ===")
    print(f"    [TYPE1-Research] CV Val: R²={type1_results['r2_mean']:.4f}±{type1_results['r2_std']:.4f}, RMSE={type1_results['rmse_mean']:.4f}±{type1_results['rmse_std']:.4f}, MAE={type1_results['mae_mean']:.4f}±{type1_results['mae_std']:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={type1_results['r2_mean']:.4f}±{type1_results['r2_std']:.4f}, RMSE={type1_results['rmse_mean']:.4f}±{type1_results['rmse_std']:.4f}, MAE={type1_results['mae_mean']:.4f}±{type1_results['mae_std']:.4f}")
    if 'final_metrics' in type2_results:
        type2_final = type2_results['final_metrics']
        type2_cv = type2_results['cv_stats']
        print(f"    [TYPE2-Production] CV Results: R²={type2_cv.get('mean_r2', 0):.4f}±{type2_cv.get('std_r2', 0):.4f}, RMSE={type2_cv.get('mean_rmse', 0):.4f}±{type2_cv.get('std_rmse', 0):.4f}, MSE={type2_cv.get('mean_mse', 0):.4f}±{type2_cv.get('std_mse', 0):.4f}, MAE={type2_cv.get('mean_mae', 0):.4f}±{type2_cv.get('std_mae', 0):.4f}")
        print(f"    [TYPE2-Production] Final Test: R²={type2_final.get('test_r2', 0):.4f}, RMSE={type2_final.get('test_rmse', 0):.4f}, MSE={type2_final.get('test_mse', 0):.4f}, MAE={type2_final.get('test_mae', 0):.4f}")

    # Return results in the format expected by existing code
    # For backward compatibility, return Type1 results as primary CV metrics
    results = {
        'method1_test_pred_mean': 0.0,  # Legacy compatibility
        'method1_test_pred_std': 0.0,   # Legacy compatibility
        'method1_val_r2_mean': type1_results['r2_mean'],
        'method1_val_r2_std': type1_results['r2_std'],
        'method1_val_rmse_mean': type1_results['rmse_mean'],
        'method1_val_rmse_std': type1_results['rmse_std'],
        'dual_cv_results': dual_results,  # Include full dual CV results
        'type1_results': type1_results,   # Type1 research results
        'type2_results': type2_results    # Type2 production results
    }

    return results
# @profiler.profile_function(module_name="8_ANO_final_model_training", save_args=False)
def train_and_evaluate_with_cv(X_train, X_test, y_train, y_test, model_params, epochs=MODEL_CONFIG['epochs'],
                               dataset_name="", module_name="", output_dir=None):
    """Train and evaluate a model with given parameters using direct training (no subprocess)"""

    # Use the dual CV approach for consistency with modules 4-7
    print(f"    Running training for {module_name}...")

    input_dim = X_train.shape[1]

    # Create model based on module type
    # FO (Module 4): SimpleDNN with fixed architecture
    # MO/FOMO/MOFO (Modules 5,6,7): FlexibleDNNModel with Optuna-optimized params

    if module_name == 'FO':
        # Module 4 uses SimpleDNN with fixed/default architecture
        hidden_dims = model_params.get('hidden_dims', [1024, 496])
        if isinstance(hidden_dims, str):
            import ast
            try:
                hidden_dims = ast.literal_eval(hidden_dims)
            except:
                hidden_dims = [1024, 496]

        # Remove output dim if present
        if hidden_dims and hidden_dims[-1] == 1:
            hidden_dims = hidden_dims[:-1]

        # Fix invalid dims
        if not hidden_dims or (len(hidden_dims) == 1 and hidden_dims[0] == 1):
            hidden_dims = [1024, 496]

        model = SimpleDNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=model_params.get('dropout_rate', 0.5),
            use_batch_norm=model_params.get('use_batch_norm', False),
            l2_reg=model_params.get('l2_reg', 1e-5),
            activation=model_params.get('activation', 'relu')
        ).to(DEVICE)
    else:
        # Modules 5,6,7 use FlexibleDNNModel with all Optuna params
        n_layers = model_params.get('n_layers', 3)
        hidden_dims = model_params.get('hidden_dims', [])

        if isinstance(hidden_dims, str):
            import ast
            try:
                hidden_dims = ast.literal_eval(hidden_dims)
            except:
                hidden_dims = []

        # Remove output dim if present
        if hidden_dims and hidden_dims[-1] == 1:
            hidden_dims = hidden_dims[:-1]

        # Build dims if empty
        if not hidden_dims:
            hidden_dims = []
            dim = max(512, input_dim // 2)
            for i in range(n_layers):
                hidden_dims.append(dim)
                dim = max(32, dim // 2)

        # Use FlexibleDNNModel with all hyperparameters from Optuna
        model = FlexibleDNNModel(
            input_dim=input_dim,
            n_layers=n_layers,
            hidden_dims=hidden_dims,
            activation=model_params.get('activation', 'relu'),
            dropout_rate=model_params.get('dropout_rate', 0.2),
            use_batch_norm=model_params.get('use_batch_norm', False),
            weight_decay=model_params.get('weight_decay', 0.0),
            optimizer_name=model_params.get('optimizer', 'Adam'),
            lr=model_params.get('lr', 0.001),
            batch_size=model_params.get('batch_size', 32),
            scheduler=model_params.get('scheduler', None)
        ).to(DEVICE)

    # Skip StandardScaler - use raw features
    # if module_name != 'MO':  # MO uses only fingerprints, no scaling needed
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer_name = model_params.get('optimizer', 'Adam')
    # For SimpleDNN (FO): use 1e-5 weight_decay to match l2_reg
    # For FlexibleDNN (MO/FOMO/MOFO): use params or 0.0
    default_weight_decay = 1e-5 if module_name == 'FO' else 0.0
    weight_decay = model_params.get('weight_decay', default_weight_decay)
    
    if optimizer_name in OPTIMIZERS:
        optimizer_class = OPTIMIZERS[optimizer_name]
        if optimizer_name == 'SGD':
            optimizer = optimizer_class(
                model.parameters(), 
                lr=model_params['lr'],
                momentum=model_params.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=model_params.get('nesterov', False)
            )
        elif optimizer_name in ['Adam', 'AdamW', 'NAdam', 'RAdam']:
            optimizer = optimizer_class(
                model.parameters(),
                lr=model_params['lr'],
                betas=model_params.get('betas', (0.9, 0.999)),
                eps=model_params.get('eps', 1e-8),
                weight_decay=weight_decay
            )
        else:
            optimizer = optimizer_class(model.parameters(), lr=model_params['lr'], weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    scheduler_name = model_params.get('scheduler', None)
    if scheduler_name and scheduler_name in SCHEDULERS:
        scheduler_class = SCHEDULERS[scheduler_name]
        if scheduler_name == 'StepLR':
            scheduler = scheduler_class(optimizer, step_size=model_params.get('step_size', 30))
        elif scheduler_name == 'ExponentialLR':
            scheduler = scheduler_class(optimizer, gamma=model_params.get('gamma', 0.95))
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = scheduler_class(optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = scheduler_class(optimizer, T_max=epochs)
        elif scheduler_name == 'OneCycleLR':
            scheduler = scheduler_class(optimizer, max_lr=model_params['lr'], total_steps=epochs)
        else:
            scheduler = scheduler_class(optimizer)
    
    # Split off validation set for early stopping (20% of training data)
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create DataLoaders for batch training
    batch_size = model_params.get('batch_size', 32)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_split),
        torch.FloatTensor(y_train_split)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_split),
        torch.FloatTensor(y_val_split)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = MODEL_CONFIG.get('early_stopping_patience', 30)

    for epoch in range(epochs):
        # Training step with batches
        model.train()
        train_losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            if model_params.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model_params['gradient_clip'])

            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validation step with batches
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_losses.append(loss.item())
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())

        val_loss = np.mean(val_losses)
        val_r2 = r2_score(val_targets, val_predictions)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"      Early stopping at epoch {epoch+1} (val_loss: {val_loss:.4f}, val_r2: {val_r2:.4f})")
                break

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"      Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}, LR: {current_lr:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"      Restored best model with validation loss: {best_val_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.inference_mode():
        # Training metrics
        train_pred = model(X_train_tensor).cpu().numpy().flatten()
        train_r2 = r2_score(y_train, train_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, train_pred)

        # Test metrics
        test_pred = model(X_test_tensor).cpu().numpy().flatten()
        test_r2 = r2_score(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_pred)

    # Return both model and metrics as a tuple
    return model, {
        'train_r2': float(train_r2),
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mse': float(test_mse),
        'test_mae': float(test_mae),
        'y_test': y_test,
        'y_pred': test_pred
    }

def calculate_applicability_domain(X_train, X_test, y_train, y_test, y_pred, method='leverage'):
    """
    Calculate Applicability Domain (AD) analysis for model predictions.

    Applicability Domain (AD) determines the chemical space where a QSAR model
    can make reliable predictions. This is a critical validation component
    recommended by OECD guidelines (OECD, 2007).

    References:
        Leverage (Williams Plot):
            - Tropsha et al. (2003), "The Importance of Being Earnest: Validation
              is the Absolute Essential for Successful Application and Interpretation
              of QSPR Models", QSAR & Combinatorial Science, 22(1), 69-77
            - Jaworska et al. (2005), "QSAR applicability domain estimation by
              projection of the training set in descriptor space", ATLA, 33(5), 445-459
            - Still widely used in 2024-2025 publications (Fisher et al., 2024)

        Bounding Box:
            - Netzeva et al. (2005), "Current status of methods for defining the
              applicability domain of (quantitative) structure-activity relationships",
              ATLA, 33(2), 155-173

        Euclidean Distance:
            - Sahigara et al. (2012), "Comparison of Different Approaches to Define
              the Applicability Domain of QSAR Models", Molecules, 17(5), 4791-4810
            - Roy et al. (2015), "On Various Metrics Used for Validation of
              Predictive QSAR Models", Comb. Chem. High Throughput Screen., 18(12)
            - Kar et al. (2018), "Applicability Domain: A Step Toward Confident
              Predictions and Decidability for QSAR Modeling", Springer

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: True test values
        y_pred: Predicted test values
        method: AD method ('leverage', 'bounding_box', 'euclidean')

    Returns:
        dict: AD analysis results including in_domain flags and statistics
    """
    from scipy.spatial import distance

    results = {
        'method': method,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    if method == 'leverage':
        # Leverage (hat values) based AD
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Calculate hat matrix diagonal
        H = X_train_scaled @ np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T
        h_train = np.diag(H)

        # Warning leverage (3 * mean)
        h_warning = 3 * (X_train_scaled.shape[1] / X_train_scaled.shape[0])

        # Calculate leverage for test samples
        h_test = np.sum((X_test_scaled @ np.linalg.pinv(X_train_scaled.T @ X_train_scaled)) * X_test_scaled, axis=1)

        # Residuals
        residuals = np.abs(y_test - y_pred)
        residual_threshold = np.mean(residuals) + 3 * np.std(residuals)

        # AD classification
        in_domain = (h_test < h_warning) & (residuals < residual_threshold)

        results.update({
            'h_warning': float(h_warning),
            'h_test_mean': float(np.mean(h_test)),
            'h_test_max': float(np.max(h_test)),
            'residual_threshold': float(residual_threshold),
            'in_domain_count': int(np.sum(in_domain)),
            'in_domain_percentage': float(100 * np.mean(in_domain)),
            'leverage_values': h_test.tolist(),
            'residuals': residuals.tolist(),
            'in_domain_flags': in_domain.tolist()
        })

    elif method == 'bounding_box':
        # Bounding box AD
        X_train_min = np.min(X_train, axis=0)
        X_train_max = np.max(X_train, axis=0)

        in_range = np.all((X_test >= X_train_min) & (X_test <= X_train_max), axis=1)

        results.update({
            'in_domain_count': int(np.sum(in_range)),
            'in_domain_percentage': float(100 * np.mean(in_range)),
            'in_domain_flags': in_range.tolist()
        })

    elif method == 'euclidean':
        # Euclidean distance to training set centroid
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        centroid = np.mean(X_train_scaled, axis=0)
        train_distances = np.linalg.norm(X_train_scaled - centroid, axis=1)
        test_distances = np.linalg.norm(X_test_scaled - centroid, axis=1)

        # Threshold: mean + 3*std of training distances
        distance_threshold = np.mean(train_distances) + 3 * np.std(train_distances)
        in_domain = test_distances < distance_threshold

        results.update({
            'distance_threshold': float(distance_threshold),
            'test_distance_mean': float(np.mean(test_distances)),
            'test_distance_max': float(np.max(test_distances)),
            'in_domain_count': int(np.sum(in_domain)),
            'in_domain_percentage': float(100 * np.mean(in_domain)),
            'distances': test_distances.tolist(),
            'in_domain_flags': in_domain.tolist()
        })

    return results

def detect_overfitting(train_metrics, test_metrics, cv_metrics=None):
    """
    Detect potential overfitting issues by analyzing train-test performance gaps.

    Overfitting occurs when a model learns training data too well, including noise,
    resulting in poor generalization to new data. This is detected by comparing
    training and test set performance.

    Severity Thresholds (empirically determined):
        - R² gap > 0.3: Severe overfitting (requires regularization/simplification)
        - R² gap > 0.15: Moderate overfitting (investigate further)
        - R² gap > 0.05: Mild overfitting (acceptable in many cases)
        - RMSE ratio > 1.5: Test errors significantly higher than training

    References:
        External Validation:
            - Golbraikh & Tropsha (2002), "Beware of q²!", J. Mol. Graphics Modell.,
              20(4), 269-276 (1000+ citations)
            - Tropsha et al. (2003), "The Importance of Being Earnest: Validation
              is the Absolute Essential", QSAR Comb. Sci., 22(1), 69-77
            - Chirico & Gramatica (2012), "Real External Predictivity of QSAR Models.
              Part 2. New Intercomparable Thresholds", J. Chem. Inf. Model., 52(8), 2044-2058

        Model Assessment:
            - Rácz et al. (2015), "Beware of R²: Simple, Unambiguous Assessment of
              the Prediction Accuracy of QSAR and QSPR Models", J. Chem. Inf. Model.,
              55(12), 2480-2488
            - Empirical thresholds: 5% difference warrants investigation (general ML practice)

    Args:
        train_metrics: dict with R², RMSE, MAE for training set
        test_metrics: dict with R², RMSE, MAE for test set
        cv_metrics: optional dict with CV mean/std metrics

    Returns:
        dict: Overfitting analysis results with severity and recommendations
    """
    results = {
        'has_overfitting': False,
        'severity': 'none',  # none, mild, moderate, severe
        'indicators': []
    }

    # Calculate performance gaps
    r2_gap = train_metrics['r2'] - test_metrics['r2']
    rmse_ratio = test_metrics['rmse'] / train_metrics['rmse'] if train_metrics['rmse'] > 0 else 1.0
    mae_ratio = test_metrics['mae'] / train_metrics['mae'] if train_metrics['mae'] > 0 else 1.0

    results['metrics'] = {
        'train_r2': float(train_metrics['r2']),
        'test_r2': float(test_metrics['r2']),
        'r2_gap': float(r2_gap),
        'train_rmse': float(train_metrics['rmse']),
        'test_rmse': float(test_metrics['rmse']),
        'rmse_ratio': float(rmse_ratio),
        'train_mae': float(train_metrics['mae']),
        'test_mae': float(test_metrics['mae']),
        'mae_ratio': float(mae_ratio)
    }

    # Overfitting indicators
    if r2_gap > 0.3:
        results['indicators'].append('Large R² gap (>0.3)')
        results['severity'] = 'severe'
        results['has_overfitting'] = True
    elif r2_gap > 0.15:
        results['indicators'].append('Moderate R² gap (>0.15)')
        results['severity'] = 'moderate' if results['severity'] == 'none' else results['severity']
        results['has_overfitting'] = True
    elif r2_gap > 0.05:
        results['indicators'].append('Mild R² gap (>0.05)')
        results['severity'] = 'mild' if results['severity'] == 'none' else results['severity']
        results['has_overfitting'] = True

    if rmse_ratio > 1.5:
        results['indicators'].append('RMSE ratio >1.5 (test >> train)')
        results['severity'] = 'moderate' if results['severity'] in ['none', 'mild'] else results['severity']
        results['has_overfitting'] = True
    elif rmse_ratio > 1.2:
        results['indicators'].append('RMSE ratio >1.2')
        results['severity'] = 'mild' if results['severity'] == 'none' else results['severity']
        results['has_overfitting'] = True

    if mae_ratio > 1.5:
        results['indicators'].append('MAE ratio >1.5 (test >> train)')
        results['severity'] = 'moderate' if results['severity'] in ['none', 'mild'] else results['severity']
        results['has_overfitting'] = True

    # CV comparison (if available)
    if cv_metrics:
        cv_test_gap = cv_metrics.get('cv_r2_mean', test_metrics['r2']) - test_metrics['r2']
        results['metrics']['cv_r2_mean'] = float(cv_metrics.get('cv_r2_mean', 0))
        results['metrics']['cv_test_gap'] = float(cv_test_gap)

        if abs(cv_test_gap) > 0.1:
            results['indicators'].append(f'CV-Test R² gap: {cv_test_gap:.3f}')

    # Overall assessment
    if not results['has_overfitting']:
        results['assessment'] = 'No significant overfitting detected'
    elif results['severity'] == 'mild':
        results['assessment'] = 'Mild overfitting - acceptable for most applications'
    elif results['severity'] == 'moderate':
        results['assessment'] = 'Moderate overfitting - consider regularization'
    else:
        results['assessment'] = 'Severe overfitting - model not reliable'

    return results

def create_ad_plots(ad_results, y_test, y_pred, output_dir, model_name):
    """
    Create Applicability Domain visualization plots.

    Args:
        ad_results: AD analysis results from calculate_applicability_domain
        y_test: True test values
        y_pred: Predicted values
        output_dir: Output directory for plots
        model_name: Model identifier for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method = ad_results['method']
    in_domain = np.array(ad_results['in_domain_flags'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Applicability Domain Analysis - {model_name}\nMethod: {method.upper()}',
                 fontsize=14, fontweight='bold')

    # 1. Predictions colored by AD
    ax = axes[0, 0]
    ax.scatter(y_test[in_domain], y_pred[in_domain], alpha=0.6, s=30,
               c='green', label=f'In Domain ({np.sum(in_domain)})')
    ax.scatter(y_test[~in_domain], y_pred[~in_domain], alpha=0.6, s=30,
               c='red', label=f'Out of Domain ({np.sum(~in_domain)})')

    # Diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.5)

    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title('Predictions by AD Status', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Residuals by AD
    ax = axes[0, 1]
    residuals = np.abs(y_test - y_pred)

    if method == 'leverage':
        leverage = np.array(ad_results['leverage_values'])
        h_warning = ad_results['h_warning']

        ax.scatter(leverage[in_domain], residuals[in_domain], alpha=0.6, s=30,
                  c='green', label='In Domain')
        ax.scatter(leverage[~in_domain], residuals[~in_domain], alpha=0.6, s=30,
                  c='red', label='Out of Domain')
        ax.axvline(h_warning, color='orange', linestyle='--', linewidth=2,
                  label=f'h* = {h_warning:.3f}')
        ax.axhline(ad_results['residual_threshold'], color='purple', linestyle='--',
                  linewidth=2, label=f'Residual threshold')
        ax.set_xlabel('Leverage (h)', fontsize=11)
        ax.set_ylabel('Absolute Residuals', fontsize=11)
        ax.set_title('Williams Plot (Leverage vs Residuals)', fontsize=12)

    elif method == 'euclidean':
        distances = np.array(ad_results['distances'])
        threshold = ad_results['distance_threshold']

        ax.scatter(distances[in_domain], residuals[in_domain], alpha=0.6, s=30,
                  c='green', label='In Domain')
        ax.scatter(distances[~in_domain], residuals[~in_domain], alpha=0.6, s=30,
                  c='red', label='Out of Domain')
        ax.axvline(threshold, color='orange', linestyle='--', linewidth=2,
                  label=f'Distance threshold')
        ax.set_xlabel('Distance to Centroid', fontsize=11)
        ax.set_ylabel('Absolute Residuals', fontsize=11)
        ax.set_title('Distance Plot', fontsize=12)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. AD statistics
    ax = axes[1, 0]
    categories = ['In Domain', 'Out of Domain']
    counts = [ad_results['in_domain_count'],
              ad_results['n_test'] - ad_results['in_domain_count']]
    colors = ['green', 'red']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title(f'AD Coverage: {ad_results["in_domain_percentage"]:.1f}% In Domain',
                fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # 4. Performance by AD status
    ax = axes[1, 1]

    # Calculate metrics for in/out domain
    from sklearn.metrics import r2_score, mean_absolute_error

    if np.sum(in_domain) > 0:
        r2_in = r2_score(y_test[in_domain], y_pred[in_domain])
        mae_in = mean_absolute_error(y_test[in_domain], y_pred[in_domain])
    else:
        r2_in, mae_in = 0, 0

    if np.sum(~in_domain) > 0:
        r2_out = r2_score(y_test[~in_domain], y_pred[~in_domain])
        mae_out = mean_absolute_error(y_test[~in_domain], y_pred[~in_domain])
    else:
        r2_out, mae_out = 0, 0

    metrics = ['R²', 'MAE']
    in_vals = [r2_in, mae_in]
    out_vals = [r2_out, mae_out]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, in_vals, width, label='In Domain', color='green', alpha=0.7)
    ax.bar(x + width/2, out_vals, width, label='Out of Domain', color='red', alpha=0.7)

    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Performance by AD Status', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / f'{model_name}_ad_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  AD plot saved: {plot_file}")

    return str(plot_file)

def create_overfitting_plot(overfitting_results, output_dir, model_name):
    """
    Create overfitting analysis visualization.

    Args:
        overfitting_results: Results from detect_overfitting
        output_dir: Output directory for plots
        model_name: Model identifier for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = overfitting_results['metrics']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Overfitting Analysis - {model_name}\nSeverity: {overfitting_results["severity"].upper()}',
                 fontsize=14, fontweight='bold')

    # 1. Train vs Test R²
    ax = axes[0, 0]
    categories = ['Train', 'Test']
    r2_values = [metrics['train_r2'], metrics['test_r2']]
    colors = ['blue', 'orange']

    bars = ax.bar(categories, r2_values, color=colors, alpha=0.7)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title(f'R² Comparison (Gap: {metrics["r2_gap"]:.3f})', fontsize=12)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 2. Train vs Test RMSE
    ax = axes[0, 1]
    rmse_values = [metrics['train_rmse'], metrics['test_rmse']]

    bars = ax.bar(categories, rmse_values, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title(f'RMSE Comparison (Ratio: {metrics["rmse_ratio"]:.2f})', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 3. Train vs Test MAE
    ax = axes[1, 0]
    mae_values = [metrics['train_mae'], metrics['test_mae']]

    bars = ax.bar(categories, mae_values, color=colors, alpha=0.7)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title(f'MAE Comparison (Ratio: {metrics["mae_ratio"]:.2f})', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 4. Overfitting indicators
    ax = axes[1, 1]
    ax.axis('off')

    # Create text summary
    severity_colors = {
        'none': 'green',
        'mild': 'yellow',
        'moderate': 'orange',
        'severe': 'red'
    }

    summary_text = f"Assessment: {overfitting_results['assessment']}\n\n"
    summary_text += "Indicators:\n"

    if overfitting_results['indicators']:
        for indicator in overfitting_results['indicators']:
            summary_text += f"• {indicator}\n"
    else:
        summary_text += "• No overfitting indicators detected\n"

    # Add recommendations
    summary_text += "\nRecommendations:\n"
    if overfitting_results['severity'] == 'none':
        summary_text += "• Model is well-generalized\n"
        summary_text += "• Safe to deploy for predictions\n"
    elif overfitting_results['severity'] == 'mild':
        summary_text += "• Minor overfitting detected\n"
        summary_text += "• Acceptable for most applications\n"
        summary_text += "• Monitor performance on new data\n"
    elif overfitting_results['severity'] == 'moderate':
        summary_text += "• Consider increasing regularization\n"
        summary_text += "• Add more training data if possible\n"
        summary_text += "• Use ensemble methods\n"
    else:  # severe
        summary_text += "• ⚠️ Model not reliable for predictions\n"
        summary_text += "• Retrain with stronger regularization\n"
        summary_text += "• Increase training data\n"
        summary_text += "• Simplify model architecture\n"

    ax.text(0.1, 0.5, summary_text,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=severity_colors[overfitting_results['severity']],
                     alpha=0.3))

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / f'{model_name}_overfitting_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Overfitting plot saved: {plot_file}")

    return str(plot_file)

# @profiler.profile_function(module_name="8_ANO_final_model_training", save_args=False)
def main():
    """Main training function"""
    print("="*80)
    print("[MODULE 8] ANO FINAL MODEL TRAINING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Module Focus: Final Model Training with Best Parameters")
    print("="*80)

    # Check renew setting from config
    renew = MODEL_CONFIG.get('renew', False)
    print(f"⚙️  Renew setting: {renew} ({'Fresh start' if renew else 'Resume mode'})")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ANO Final Model Training')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {MODEL_CONFIG["epochs"]} from config.py)')
    parser.add_argument('--split-types', nargs='+', default=['rm'],
                        help='Split types to process (default: rm only). Options: rm, ac, cl, cs, en, pc, sa, sc, ti')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to process (default: from CODE_SPECIFIC_DATASETS). Options: ws, de, lo, hu')
    args = parser.parse_args()

    # Set EPOCHS with proper priority: args > MODEL_CONFIG['epochs'] > module_epochs
    global EPOCHS
    EPOCHS = get_epochs_for_module('8', args)
    epochs = EPOCHS  # For backward compatibility

    print("Starting ANO Final Model Training...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Epochs: {epochs}")
    print("=" * 60)

    # Priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    if args.datasets:
        datasets = args.datasets
        print(f"Datasets from argparse: {datasets}")
    elif '8' in CODE_SPECIFIC_DATASETS:
        datasets = CODE_SPECIFIC_DATASETS['8']
        print(f"Datasets from CODE_SPECIFIC_DATASETS: {datasets}")
    else:
        # Fallback: scan data directory
        datasets = []
        for split_type in ACTIVE_SPLIT_TYPES:
            split_dir = Path(DATA_PATH) / 'train' / split_type
            if split_dir.exists():
                for csv_file in split_dir.glob('*_train.csv'):
                    dataset = csv_file.stem.split('_')[1]  # Extract dataset from filename
                    if dataset not in datasets:
                        datasets.append(dataset)
        print(f"Datasets from data directory scan: {datasets}")

    modules = ['FO', 'MO', 'FOMO', 'MOFO']
    split_types = args.split_types  # Use split types from command line arguments

    print(f"Split types: {split_types}")
    print(f"Modules: {modules}")
    
    # Results storage
    all_results = {}
    
    for dataset in datasets:
        for split_type in split_types:
            dataset_key = f"{dataset}_{split_type}"
            all_results[dataset_key] = {}

            print(f"\nProcessing {dataset.upper()}-{split_type}...")
            print("-" * 40)

            for module in modules:
                print(f"\n  Module {module}:")

                try:
                    # Load data and prepare features
                    X_train, X_test, y_train, y_test = load_data_and_features(dataset, split_type, module)

                    if X_train is None:
                        print(f"    Skipping {module} due to data loading error")
                        continue

                    # Get model parameters for this module
                    model_params = get_model_params(dataset, split_type, module)
                    
                    print(f"    Architecture: {model_params['n_layers']} layers, dims={model_params['hidden_dims']}")
                    print(f"    Activation: {model_params['activation']}, Optimizer: {model_params['optimizer']}")
                    
                    # Train and evaluate with CV
                    model, metrics = train_and_evaluate_with_cv(
                        X_train, X_test, y_train, y_test, model_params,
                        epochs=epochs, dataset_name=dataset, module_name=module, output_dir=OUTPUT_DIR
                    )

                    # Check if model is None
                    if model is None:
                        raise ValueError("train_and_evaluate_with_cv returned None model")

                    # Also create the final scatter plot (in addition to CV plots)
                    # Save plots in plots subfolder within each dataset/split directory
                    plot_dir = OUTPUT_DIR / dataset / split_type / 'plots'
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = plot_dir / f"{dataset}_{split_type}_{module}_final.png"

                    create_scatter_plot(
                        metrics['y_test'], metrics['y_pred'],
                        f"{dataset.upper()}-{split_type.upper()} {module} Model\nFinal Test Results",
                        plot_path
                    )
                    print(f"    Saved final scatter plot: {plot_path}")

                    # Applicability Domain Analysis
                    print(f"    Running Applicability Domain analysis...")
                    ad_results = calculate_applicability_domain(
                        X_train, X_test, y_train,
                        metrics['y_test'], metrics['y_pred'],
                        method='leverage'  # Using Williams plot (leverage method)
                    )

                    # Save AD results as JSON
                    ad_json_path = plot_dir / f"{dataset}_{split_type}_{module}_ad_analysis.json"
                    with open(ad_json_path, 'w') as f:
                        json.dump(ad_results, f, indent=2)
                    print(f"    AD analysis saved: {ad_json_path}")

                    # Create AD visualization plots
                    ad_plot_path = create_ad_plots(
                        ad_results, metrics['y_test'], metrics['y_pred'],
                        plot_dir, f"{dataset}_{split_type}_{module}"
                    )

                    # Save AD results as CSV for easy viewing
                    ad_csv_path = plot_dir / f"{dataset}_{split_type}_{module}_ad_results.csv"
                    ad_df = pd.DataFrame({
                        'actual_value': metrics['y_test'],
                        'predicted_value': metrics['y_pred'],
                        'in_domain': ad_results['in_domain_flags'],
                        'leverage': ad_results.get('leverage_values', [0]*len(metrics['y_test'])),
                        'residual': ad_results.get('residuals', [0]*len(metrics['y_test']))
                    })
                    ad_df.to_csv(ad_csv_path, index=False)
                    print(f"    AD results CSV saved: {ad_csv_path}")
                    print(f"    In-domain coverage: {ad_results['in_domain_percentage']:.1f}%")

                    # Overfitting Detection Analysis
                    print(f"    Running overfitting detection...")
                    train_metrics_for_check = {
                        'r2': metrics['train_r2'],
                        'rmse': metrics['train_rmse'],
                        'mae': metrics.get('train_mae', 0)
                    }
                    test_metrics_for_check = {
                        'r2': metrics['test_r2'],
                        'rmse': metrics['test_rmse'],
                        'mae': metrics.get('test_mae', 0)
                    }

                    # Include CV metrics if available
                    cv_metrics_for_check = None
                    if 'cv_test_r2' in metrics:
                        cv_metrics_for_check = {
                            'r2': metrics['cv_test_r2'],
                            'rmse': metrics.get('cv_test_rmse', 0),
                            'mae': metrics.get('cv_test_mae', 0)
                        }

                    overfitting_results = detect_overfitting(
                        train_metrics_for_check,
                        test_metrics_for_check,
                        cv_metrics_for_check
                    )

                    # Save overfitting results as JSON
                    overfit_json_path = plot_dir / f"{dataset}_{split_type}_{module}_overfitting_analysis.json"
                    with open(overfit_json_path, 'w') as f:
                        json.dump(overfitting_results, f, indent=2)
                    print(f"    Overfitting analysis saved: {overfit_json_path}")

                    # Create overfitting visualization plot
                    overfit_plot_path = create_overfitting_plot(
                        overfitting_results,
                        plot_dir,
                        f"{dataset}_{split_type}_{module}"
                    )

                    # Save overfitting metrics as CSV
                    overfit_csv_path = plot_dir / f"{dataset}_{split_type}_{module}_overfitting_metrics.csv"
                    overfit_df = pd.DataFrame({
                        'metric': ['R²', 'RMSE', 'MAE'],
                        'train': [train_metrics_for_check['r2'], train_metrics_for_check['rmse'], train_metrics_for_check['mae']],
                        'test': [test_metrics_for_check['r2'], test_metrics_for_check['rmse'], test_metrics_for_check['mae']],
                        'gap': [overfitting_results['metrics']['r2_gap'], overfitting_results['metrics']['rmse_ratio'] - 1, overfitting_results['metrics']['mae_ratio'] - 1]
                    })
                    overfit_df.to_csv(overfit_csv_path, index=False)
                    print(f"    Overfitting metrics CSV saved: {overfit_csv_path}")
                    print(f"    Overfitting severity: {overfitting_results['severity']}")

                    # Save complete model with dataset/split_type folder structure
                    model_dir = OUTPUT_DIR / dataset / split_type
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = model_dir / f"{dataset}_{split_type}_{module}_model.pt"

                    # Also prepare save_model directory
                    save_model_dir = Path("save_model") / dataset / split_type
                    save_model_dir.mkdir(parents=True, exist_ok=True)
                    save_model_path = save_model_dir / f"{dataset}_{split_type}_{module}_model.pt"

                    # Save model architecture info for proper loading
                    model_info = {
                        'module_type': module,
                        'input_dim': X_train.shape[1],
                        'model_params': model_params,
                        'is_flexible': module != 'FO',  # FO uses SimpleDNN, others use FlexibleDNNModel
                        'descriptor_info': None  # Will be filled based on module type
                    }

                    # Add descriptor selection info based on module type
                    if module == 'FO' or module == 'FOMO':
                        # Load FO best_params for descriptor selection
                        fo_result_file = Path(RESULT_PATH) / f'4_ANO_FeatureOptimization_FO/{dataset}/{split_type}/{dataset}_{split_type}_FO_results.json'
                        if fo_result_file.exists():
                            with open(fo_result_file, 'r') as f:
                                fo_results = json.load(f)
                            if 'best_params' in fo_results:
                                model_info['descriptor_info'] = {
                                    'type': 'best_params',
                                    'best_params': fo_results['best_params']
                                }
                    elif module == 'MOFO':
                        # Load MOFO best_params for descriptor selection (MOFO uses best_params like FO)
                        mofo_result_file = Path(RESULT_PATH) / f'7_ANO_NetworkOptimization_MOFO/{dataset}/{split_type}/{dataset}_{split_type}_MOFO_results.json'
                        if mofo_result_file.exists():
                            with open(mofo_result_file, 'r') as f:
                                mofo_results = json.load(f)
                            if 'best_params' in mofo_results:
                                model_info['descriptor_info'] = {
                                    'type': 'best_params',
                                    'best_params': mofo_results['best_params']
                                }
                    # MO doesn't use descriptors, only fingerprints

                    # Save model in TorchScript format for better compatibility
                    model.eval()
                    # Create example input for tracing using larger batch size to avoid BatchNorm issues
                    batch_size = max(2, min(8, len(X_train)))  # Use batch size between 2-8
                    example_input = torch.randn(batch_size, X_train.shape[1])

                    # Convert to TorchScript with eval mode to handle BatchNorm properly
                    try:
                        with torch.inference_mode():
                            scripted_model = torch.jit.trace(model, example_input)
                        # Save TorchScript model to both locations
                        torch.jit.save(scripted_model, str(model_path))
                        torch.jit.save(scripted_model, str(save_model_path))

                        # Save model info
                        info_path = model_path.with_suffix('.json')
                        save_info_path = save_model_path.with_suffix('.json')
                        with open(info_path, 'w') as f:
                            json.dump(model_info, f, indent=2)
                        with open(save_info_path, 'w') as f:
                            json.dump(model_info, f, indent=2)

                    except Exception as e:
                        print(f"    Warning: TorchScript conversion failed: {e}")
                        print(f"    Saving as regular PyTorch model instead")
                        # Fallback: save as regular .pth file
                        model_path = model_path.with_suffix('.pth')
                        save_model_path = save_model_path.with_suffix('.pth')
                        torch.save({
                            'model': model,
                            'model_state_dict': model.state_dict(),
                            'model_info': model_info
                        }, str(model_path))
                        torch.save({
                            'model': model,
                            'model_state_dict': model.state_dict(),
                            'model_info': model_info
                        }, str(save_model_path))

                    # Store comprehensive results including type1 and type2
                    stored_metrics = {k: v for k, v in metrics.items() if k not in ['y_test', 'y_pred']}

                    # Add type1 and type2 results if available
                    if 'type1_cv_r2' in metrics:
                        stored_metrics['type1_cv_r2'] = metrics['type1_cv_r2']
                        stored_metrics['type1_cv_rmse'] = metrics.get('type1_cv_rmse', 0)
                        stored_metrics['type1_cv_mae'] = metrics.get('type1_cv_mae', 0)
                    if 'type2_cv_r2' in metrics:
                        stored_metrics['type2_cv_r2'] = metrics['type2_cv_r2']
                        stored_metrics['type2_cv_rmse'] = metrics.get('type2_cv_rmse', 0)
                        stored_metrics['type2_cv_mae'] = metrics.get('type2_cv_mae', 0)
                        stored_metrics['type2_final_r2'] = metrics.get('type2_final_r2', 0)
                        stored_metrics['type2_final_rmse'] = metrics.get('type2_final_rmse', 0)

                    all_results[dataset_key][module] = stored_metrics

                    print(f"    Final Test R²: {metrics['test_r2']:.4f}, RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics.get('test_mae', 0):.4f}")
                    if 'cv_test_r2' in metrics:
                        print(f"    CV Test R²: {metrics['cv_test_r2']:.4f}, RMSE: {metrics.get('cv_test_rmse', 0):.4f}")
                    if 'type1_cv_r2' in metrics:
                        print(f"    Type1 CV R²: {metrics['type1_cv_r2']:.4f}, RMSE: {metrics.get('type1_cv_rmse', 0):.4f}")
                    if 'type2_final_r2' in metrics:
                        print(f"    Type2 Final R²: {metrics['type2_final_r2']:.4f}, RMSE: {metrics.get('type2_final_rmse', 0):.4f}")
                    print(f"    Model saved: {model_path}")
                    print(f"    Model copied to: {save_model_path}")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"    ⚠️  Error in {module}: {error_msg}")

                    # Skip if no valid parameters or data
                    if "No parameters found" in error_msg or "Dataset files not found" in error_msg:
                        print(f"    → Skipping {module} (no data available)")
                        all_results[dataset_key][module] = {'skipped': True, 'reason': error_msg}
                    else:
                        # Real error - still record it but mark as failed
                        all_results[dataset_key][module] = {
                            'train_r2': 0.0,
                            'test_r2': 0.0,
                            'test_rmse': 999.0,
                            'test_mae': 999.0,
                            'error': error_msg,
                            'skipped': False
                        }
    
    # Save summary as JSON
    summary_path = OUTPUT_DIR / 'training_summary_complete.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary as CSV
    csv_data = []
    for dataset_split, modules_results in all_results.items():
        dataset, split_type = dataset_split.split('_')
        for module, metrics in modules_results.items():
            row = {
                'dataset': dataset,
                'split_type': split_type,
                'module': module,
                'train_r2': metrics.get('train_r2', -999),
                'test_r2': metrics.get('test_r2', -999),
                'test_rmse': metrics.get('test_rmse', 999),
                'test_mse': metrics.get('test_mse', 999999),
                'test_mae': metrics.get('test_mae', 999),
                'cv_test_r2': metrics.get('cv_test_r2', -999),
                'cv_method1_test_rmse': metrics.get('cv_method1_test_rmse', 999),
                'error': metrics.get('error', '')
            }
            csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'training_summary_complete.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    for module in modules:
        print(f"\n{module} Module:")
        for dataset in datasets:
            for split_type in split_types:
                key = f"{dataset}_{split_type}"
                if key in all_results and module in all_results[key]:
                    metrics = all_results[key][module]
                    if 'error' not in metrics:
                        print(f"  {dataset.upper()}-{split_type}: Test R²={metrics['test_r2']:.4f}, RMSE={metrics['test_rmse']:.4f}")
    
    # Calculate average performance
    print("\n" + "-" * 40)
    print("Average Performance:")
    for module in modules:
        r2_scores = []
        for dataset in datasets:
            key = f"{dataset}_{split_type}"
            if key in all_results and module in all_results[key]:
                if 'error' not in all_results[key][module]:
                    r2_scores.append(all_results[key][module]['test_r2'])
        
        if r2_scores:
            avg_r2 = np.mean(r2_scores)
            print(f"  {module}: Average R² = {avg_r2:.4f}")
    
    # Generate comprehensive summary report
    generate_summary_report(all_results)
    
    # Create visualizations
    create_visualizations(all_results)
    
    # Clean up temporary model files
    import glob
    temp_patterns = [
        'temp/temp_*.pth', 'temp/temp_*.pkl', 'temp/temp_*.csv', 
        'temp/temp_*.json', 'temp/temp_*.npz',
        'temp_*.pkl', 'temp_*.csv', 'temp_*.json', 'temp_*.npz', '*.tmp'
    ]
    
    print("\nCleaning up temporary files...")
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"  Removed: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"  Error removing {os.path.basename(temp_file)}: {e}")
    
    print("\n" + "=" * 60)
    print("MODULE 8 COMPLETE!")
    print("=" * 60)

def generate_summary_report(all_results):
    """Generate comprehensive summary report"""
    print("\n" + "=" * 70)
    print("MODULE 8: DETAILED SUMMARY REPORT")
    print("=" * 70)
    
    # Convert to DataFrame for better display
    data = []
    for dataset_split, modules in all_results.items():
        parts = dataset_split.split('_')
        dataset = parts[0].upper()
        split_type = parts[1] if len(parts) > 1 else 'rm'
        for module, metrics in modules.items():
            if 'error' not in metrics:
                data.append({
                    'Dataset': dataset,
                    'Split': split_type,
                    'Module': module,
                    'Train R²': f"{metrics['train_r2']:.4f}",
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    'Test RMSE': f"{metrics['test_rmse']:.4f}",
                    'Test MAE': f"{metrics['test_mae']:.4f}"
                })
    
    if data:
        df = pd.DataFrame(data)

        # Print by module type
        for module in ['FO', 'MO', 'FOMO', 'MOFO']:
            print(f"\n{module} MODULE RESULTS:")
            print("-" * 50)
            module_df = df[df['Module'] == module]
            if not module_df.empty:
                print(module_df.to_string(index=False))
                avg_r2 = module_df['Test R²'].apply(float).mean()
                print(f"\n{module} Average Test R²: {avg_r2:.4f}")

def create_visualizations(all_results):
    """Create visualization plots"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create output directory for plots
        plot_dir = OUTPUT_DIR / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)
        
        # Prepare data
        datasets = []
        modules = ['FO', 'MO', 'FOMO', 'MOFO']
        
        # Extract unique datasets
        for key in all_results.keys():
            dataset = key.split('_')[0]
            if dataset not in datasets:
                datasets.append(dataset)
        
        # Create performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Module 8: Performance Comparison Across Modules', fontsize=16)
        
        for idx, module in enumerate(modules):
            ax = axes[idx // 2, idx % 2]
            
            test_scores = []
            train_scores = []
            dataset_labels = []
            
            for dataset in datasets:
                key = f"{dataset}_rm"  # Assuming rm split
                if key in all_results and module in all_results[key]:
                    if 'error' not in all_results[key][module]:
                        test_scores.append(all_results[key][module]['test_r2'])
                        train_scores.append(all_results[key][module]['train_r2'])
                        dataset_labels.append(dataset.upper())
            
            if test_scores:
                x = np.arange(len(dataset_labels))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, train_scores, width, label='Train', color='lightblue')
                bars2 = ax.bar(x + width/2, test_scores, width, label='Test', color='steelblue')
                
                # Add value labels on bars
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{module} Module Performance')
                ax.set_ylabel('R² Score')
                ax.set_xticks(x)
                ax.set_xticklabels(dataset_labels)
                ax.legend()
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plot_dir / 'module_performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create matrix for heatmap
        test_r2_matrix = []
        for module in modules:
            row = []
            for dataset in datasets:
                key = f"{dataset}_rm"
                if key in all_results and module in all_results[key]:
                    if 'error' not in all_results[key][module]:
                        row.append(all_results[key][module]['test_r2'])
                    else:
                        row.append(0)
                else:
                    row.append(0)
            test_r2_matrix.append(row)
        
        if test_r2_matrix and any(any(row) for row in test_r2_matrix):
            sns.heatmap(test_r2_matrix, 
                        annot=True, fmt='.3f', 
                        xticklabels=[d.upper() for d in datasets],
                        yticklabels=modules,
                        cmap='RdYlGn', vmin=0, vmax=1,
                        cbar_kws={'label': 'Test R² Score'})
            
            plt.title('Module 8: Test Performance Heatmap')
            plt.xlabel('Dataset')
            plt.ylabel('Module')
            
            plt.tight_layout()
            heatmap_path = plot_dir / 'performance_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved heatmap: {heatmap_path}")
        
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    
    # Set up logging to file
    module_name = MODULE_NAMES.get('8', '8_ANO_final_model_training')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get dataset from command line arguments or use all datasets
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--split-types', nargs='+', default=['rm'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--trials', type=int, default=None)
    args, _ = parser.parse_known_args()
    
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
            self.log.write(f"Module 8 Execution Started: {datetime.now()}\n")
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
            self.log.write(f"Module 8 Execution Completed: {datetime.now()}\n")
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