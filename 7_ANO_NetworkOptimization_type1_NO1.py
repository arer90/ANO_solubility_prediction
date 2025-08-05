#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

ANO Network Feature-Structure Optimization with PyTorch - Fixed Module
======================================================================

PURPOSE:
This module implements a two-stage optimization approach: Feature selection followed
by structure optimization. It leverages the best features found in module 5 and then
optimizes the neural network architecture for those specific features.

APPROACH:
1. Start with base fingerprints (Morgan + MACCS + Avalon = 2727 features)
2. Load best feature selection from module 5 (typically adds ~882 descriptors)
3. Combine base fingerprints with selected descriptors (~3609 total features)
4. Optimize neural network structure for this specific feature set
5. Use same evaluation protocol: 5-fold CV on training data

KEY INNOVATIONS:
- Two-stage optimization: Features first (module 5), then structure
- Reuses proven feature selections instead of starting from scratch
- Maintains feature consistency while exploring architectural space
- Ensures fair comparison with other methods by using same features

TECHNICAL DETAILS:
- Base features: 2048 (Morgan) + 167 (MACCS) + 512 (Avalon) = 2727
- Additional features: ~882 from selected molecular descriptors
- Total features: ~3609 (varies based on 3D descriptor availability)
- Architecture search: 2-4 layers, variable hidden units
- Optimization: Batch size, learning rate, dropout, layer dimensions

FIXED ISSUES:
- Corrected feature concatenation (was duplicating to 6336 features)
- Added study deletion option for clean reruns
- Fixed column name from 'y' to 'target'
- Study name: ano_network_FOMO_{dataset}_{split_type}
- All splits supported (not just rm)
- Epochs: 100 (was 30)
- Trials: 1
- All fingerprints + module 5's best features combined correctly
- Model hyperparameter tuning with layer count
- All metrics saved (R², RMSE, MSE, MAE, time, resources)
- Model saved as full_model.pth

Feature → Structure Network:
1. Use all fingerprints (morgan + maccs + avalon)
2. Get best feature selection from module 5 results
3. Combine all + selected features (without duplication)
4. Optimize model structure with hyperparameters including layer count
"""

import os
import sys
import time
import subprocess
import logging
import warnings
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tempfile
import shutil
import optuna
import psutil
import json

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from rdkit.Chem import AllChem

# Import from molecular_loader_with_fp
from extra_code.molecular_loader_with_fp import (
    load_data_ws, load_data_de, load_data_lo, load_data_hu,
    prefilter_3d_conformers, get_fingerprints
)

# Import feature selection functions
from extra_code.ano_feature_search import search_data_descriptor_compress_fixed
from extra_code.ano_feature_selection import (
    selection_data_descriptor_compress,
    selection_fromStudy_compress,
    convert_params_to_selection
)

# Configuration - Global variables
# Enable all split types for comprehensive analysis
# SPLIT_TYPES = ["rm", "sc", "cs", "cl", "pc", "ac", "sa", "ti", "en"]  # All split types
SPLIT_TYPES = ["rm", "sc", "cs", "cl", "pc", "ac", "sa", "ti", "en"]  # All split types
N_TRIALS = 1
EPOCHS = 100  # Fixed to 100 epochs for consistent training
OUTPUT_DIR = Path("result/7_ANO_network_feature_structure_pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global data storage
PREPARED_DATA = {}

def train_and_evaluate_cv(X, y, model_params, X_test=None, y_test=None):
    """
    Train and evaluate with CV-5 and return all metrics
    
    This function implements 5-fold cross-validation for robust model evaluation.
    It trains separate models for each fold and aggregates the results to get
    reliable performance estimates.
    
    Process:
    1. Split data into 5 folds
    2. For each fold:
       - Train on 4 folds, validate on 1 fold
       - Use subprocess for clean memory management
       - Collect all metrics (R², RMSE, MSE, MAE)
    3. Calculate mean, std, and best fold performance
    4. If test data provided, train on full training data and evaluate on test
    
    Args:
        X: Feature matrix (n_samples × n_features)
        y: Target values
        model_params: Dict with batch_size, epochs, learning_rate
        X_test: Optional test feature matrix
        y_test: Optional test target values
    
    Returns:
        Tuple of 16 values: mean, std, and best for each metric + test metrics
    """
    
    # Convert to numpy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).flatten()
    
    # CV-5 setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mse_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Save data to temporary files
        np.save("temp_X_train.npy", X_train)
        np.save("temp_y_train.npy", y_train)
        np.save("temp_X_test.npy", X_val)
        np.save("temp_y_test.npy", y_val)
        
        # Subprocess 실행
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            "temp_X_train.npy", "temp_y_train.npy",
            "temp_X_test.npy", "temp_y_test.npy",
            "save_model/full_model.pth"  # 고정된 모델 파일명
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 결과 파싱 - 모든 메트릭 추출
        try:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if ',' in line and line.count(',') == 3:  # R²,RMSE,MSE,MAE format
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
                            break
                        except ValueError:
                            continue
            else:
                print(f"Fold {fold+1}: No valid metrics found")
                cv_r2_scores.append(0.0)
                cv_rmse_scores.append(0.0)
                cv_mse_scores.append(0.0)
                cv_mae_scores.append(0.0)
        except Exception as e:
            print(f"Fold {fold+1}: Error parsing metrics - {e}")
            cv_r2_scores.append(0.0)
            cv_rmse_scores.append(0.0)
            cv_mse_scores.append(0.0)
            cv_mae_scores.append(0.0)
        
        # Clean up temporary files
        for f in ["temp_X_train.npy", "temp_y_train.npy", "temp_X_test.npy", "temp_y_test.npy"]:
            if os.path.exists(f):
                os.remove(f)
    
    # Calculate final metrics
    if len(cv_r2_scores) == 0:
        print("Warning: No valid CV scores obtained")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    mean_r2 = np.mean(cv_r2_scores)
    std_r2 = np.std(cv_r2_scores)
    mean_rmse = np.mean(cv_rmse_scores)
    std_rmse = np.std(cv_rmse_scores)
    mean_mse = np.mean(cv_mse_scores)
    std_mse = np.std(cv_mse_scores)
    mean_mae = np.mean(cv_mae_scores)
    std_mae = np.std(cv_mae_scores)
    
    # Best fold metrics (highest R²)
    best_fold_idx = np.argmax(cv_r2_scores)
    best_r2 = cv_r2_scores[best_fold_idx]
    best_rmse = cv_rmse_scores[best_fold_idx]
    best_mse = cv_mse_scores[best_fold_idx]
    best_mae = cv_mae_scores[best_fold_idx]
    
    print(f"CV Results: R2={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}")
    print(f"Best Fold: R2={best_r2:.4f}, RMSE={best_rmse:.4f}, MSE={best_mse:.4f}, MAE={best_mae:.4f}")
    
    # Train final model on full training data and evaluate on test set if provided
    test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
    
    if X_test is not None and y_test is not None:
        print("\nTraining final model on full training data and evaluating on test set...")
        
        # Save full training and test data
        np.save("temp_X_train.npy", X)
        np.save("temp_y_train.npy", y)
        np.save("temp_X_test.npy", X_test)
        np.save("temp_y_test.npy", y_test)
        
        # Run final training
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch.py",
            str(model_params['batch_size']),
            str(model_params['epochs']),
            str(model_params['learning_rate']),
            "temp_X_train.npy", "temp_y_train.npy",
            "temp_X_test.npy", "temp_y_test.npy",
            "save_model/full_model.pth"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse test results
        try:
            lines = result.stdout.strip().split('\n')
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
        
        # Clean up temporary files
        for f in ["temp_X_train.npy", "temp_y_train.npy", "temp_X_test.npy", "temp_y_test.npy"]:
            if os.path.exists(f):
                os.remove(f)
    
    return (mean_r2, std_r2, best_r2, 
            mean_rmse, std_rmse, best_rmse,
            mean_mse, std_mse, best_mse,
            mean_mae, std_mae, best_mae,
            test_r2, test_rmse, test_mse, test_mae)

# PyTorch DNN Model with hyperparameter tuning including layer count
class DNNModel(nn.Module):
    """
    Deep Neural Network with optimized architecture for molecular property prediction
    
    This model uses several advanced techniques:
    - SiLU activation: Smooth approximation of ReLU, often performs better
    - BatchNorm: Normalizes inputs to each layer, stabilizes training
    - Xavier initialization: Proper weight initialization for deep networks
    - Variable depth: Number of layers is optimized (2-4)
    
    Architecture pattern for each hidden layer:
    Linear -> SiLU -> BatchNorm -> Dropout
    
    The final layer is a single linear unit for regression output.
    
    Args:
        input_dim: Number of input features (~3609 after feature selection)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability for regularization
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(DNNModel, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0], momentum=0.01))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.SiLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1], momentum=0.01))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))  # 마지막 hidden unit = 1
        
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

def prepare_data_for_optimization():
    """
    Prepare data for optimization using preprocessed train data
    
    This function loads and prepares molecular data for all datasets. It:
    1. Loads SMILES and targets from preprocessed files
    2. Removes duplicates across different splits
    3. Generates 3D conformers for 3D descriptor calculation
    4. Computes base fingerprints (Morgan + MACCS + Avalon)
    5. Stores everything in global PREPARED_DATA dictionary
    
    The data preparation is done once and reused across all trials
    to ensure consistency and efficiency.
    
    Global side effect:
        Updates PREPARED_DATA dictionary with prepared molecular data
    """
    global PREPARED_DATA
    PREPARED_DATA = {}
    
    dataset_mapping = {
        'ws': 'ws496_logS',
        'de': 'delaney-processed',
        'lo': 'Lovric2020_logS0',
        'hu': 'huusk'
    }
    
    # Also prepare test data for each split
    TEST_DATA = {}
    
    for dataset_key, dataset_name in dataset_mapping.items():
        print(f"Preparing {dataset_key.upper()} dataset from preprocessed data...")
        try:
            # Collect data from all splits for training
            all_smiles = []
            all_y = []
            
            for split_type in SPLIT_TYPES:
                train_file = Path(f"data/train/{split_type}/{split_type}_{dataset_name}_train.csv")
                if train_file.exists():
                    df = pd.read_csv(train_file)
                    all_smiles.extend(df['smiles'].tolist())
                    all_y.extend(df['target'].tolist())
            
            # Remove duplicates
            unique_data = {}
            for smiles, y in zip(all_smiles, all_y):
                unique_data[smiles] = y
            
            smiles_list = list(unique_data.keys())
            y_list = list(unique_data.values())
            
            print(f"  Total unique molecules: {len(smiles_list)}")
            
            # Generate and filter 3D conformers
            smiles_filtered, y_filtered, mols_filtered, mols_3d_filtered = prefilter_3d_conformers(smiles_list, y_list)
            
            print(f"  After 3D filtering: {len(smiles_filtered)} molecules")
            
            # All fingerprints: morgan + maccs + avalon
            morgan_fps, maccs_fps, avalon_fps = get_fingerprints(mols_filtered)
            fps = np.hstack([morgan_fps, maccs_fps, avalon_fps])
            
            print(f"  All fingerprint shape: {fps.shape}")
            
            # Store prepared data
            PREPARED_DATA[dataset_key] = {
                'smiles': smiles_filtered,
                'y': y_filtered, 
                'mols': mols_filtered,
                'mols_3d': mols_3d_filtered,
                'fps': fps
            }
            
            print(f"  {dataset_key.upper()}: {len(smiles_filtered)} molecules, {fps.shape[1]} features")
            
            # Prepare test data for each split type
            TEST_DATA[dataset_key] = {}
            for split_type in SPLIT_TYPES:
                test_file = Path(f"data/test/{split_type}/{split_type}_{dataset_name}_test.csv")
                if test_file.exists():
                    df_test = pd.read_csv(test_file)
                    test_smiles = df_test['smiles'].tolist()
                    test_y = df_test['target'].tolist()
                    
                    # Generate and filter 3D conformers for test data
                    test_smiles_filtered, test_y_filtered, test_mols_filtered, test_mols_3d_filtered = prefilter_3d_conformers(test_smiles, test_y)
                    
                    # Generate fingerprints for test data
                    test_morgan_fps, test_maccs_fps, test_avalon_fps = get_fingerprints(test_mols_filtered)
                    test_fps = np.hstack([test_morgan_fps, test_maccs_fps, test_avalon_fps])
                    
                    TEST_DATA[dataset_key][split_type] = {
                        'smiles': test_smiles_filtered,
                        'y': test_y_filtered,
                        'mols': test_mols_filtered,
                        'mols_3d': test_mols_3d_filtered,
                        'fps': test_fps
                    }
                    
                    print(f"    Test data for {split_type}: {len(test_smiles_filtered)} molecules")
            
        except Exception as e:
            print(f"Error preparing {dataset_key.upper()} data: {e}")
            raise
    
    # Store TEST_DATA globally
    PREPARED_DATA['TEST_DATA'] = TEST_DATA

def get_best_feature_selection(dataset_name, split_type):
    """
    Get best feature selection from module 5 results
    
    This function loads the optimal feature selection discovered by module 5
    from the Optuna database. Module 5 performs exhaustive feature selection
    to find the best combination of molecular descriptors.
    
    The feature selection is stored as binary flags (0/1) for each of the
    51 possible molecular descriptor categories.
    
    Args:
        dataset_name: Dataset identifier ('ws', 'de', 'lo', 'hu')
        split_type: Data split strategy ('rm', 'sc', etc.)
    
    Returns:
        Dictionary mapping descriptor names to inclusion flags (0/1)
        Falls back to all features (all 1s) if loading fails
    """
    try:
        # Load study from Optuna storage - 5번 모듈 결과
        study_name = f"ano_feature_{dataset_name}_{split_type}"
        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///ano_final.db"
        )
        
        # Get best parameters
        best_params = study.best_params
        print(f"Successfully loaded best feature selection for {dataset_name}-{split_type}")
        return best_params
        
    except Exception as e:
        print(f"Error loading best feature selection for {dataset_name}-{split_type}: {e}")
        # Return default parameters if loading fails
        return {
            'MolWt': 1, 'MolLogP': 1, 'MolMR': 1, 'TPSA': 1, 'NumRotatableBonds': 1,
            'HeavyAtomCount': 1, 'NumHAcceptors': 1, 'NumHDonors': 1, 'NumHeteroatoms': 1,
            'NumValenceElectrons': 1, 'NHOHCount': 1, 'NOCount': 1, 'RingCount': 1,
            'NumAromaticRings': 1, 'NumSaturatedRings': 1, 'NumAliphaticRings': 1,
            'LabuteASA': 1, 'BalabanJ': 1, 'BertzCT': 1, 'Ipc': 1,
            'kappa_Series[1-3]_ind': 1, 'Chi_Series[13]_ind': 1, 'Phi': 1, 'HallKierAlpha': 1,
            'NumAmideBonds': 1, 'NumSpiroAtoms': 1, 'NumBridgeheadAtoms': 1, 'FractionCSP3': 1,
            'PEOE_VSA_Series[1-14]_ind': 1, 'SMR_VSA_Series[1-10]_ind': 1, 'SlogP_VSA_Series[1-12]_ind': 1,
            'EState_VSA_Series[1-11]_ind': 1, 'VSA_EState_Series[1-10]': 1, 'MQNs': 1,
            'AUTOCORR2D': 1, 'BCUT2D': 1, 'Asphericity': 1, 'PBF': 1, 'RadiusOfGyration': 1,
            'InertialShapeFactor': 1, 'Eccentricity': 1, 'SpherocityIndex': 1,
            'PMI_series[1-3]_ind': 1, 'NPR_series[1-2]_ind': 1, 'AUTOCORR3D': 1,
            'RDF': 1, 'MORSE': 1, 'WHIM': 1, 'GETAWAY': 1
        }

def create_objective_function(dataset_name, split_type, test_data):
    """
    Create objective function for Feature → Structure network optimization
    
    This is the core optimization logic that combines:
    1. Fixed feature selection from module 5
    2. Dynamic structure optimization
    
    The two-stage approach ensures:
    - Features are optimized first (in module 5)
    - Structure is then optimized for those specific features
    - This decomposition makes the optimization more tractable
    
    Hyperparameters optimized:
    - n_layers: Network depth (2-4 layers)
    - hidden_dim_i: Units in each layer (funnel architecture)
    - dropout_rate: Regularization strength (0.1-0.5)
    - learning_rate: Optimization step size (1e-4 to 1e-2)
    - batch_size: Mini-batch size (16, 32, or 64)
    
    Args:
        dataset_name: Dataset to optimize for
        split_type: Data splitting strategy
    
    Returns:
        Objective function for Optuna
    """
    def objective_function(trial):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            data = PREPARED_DATA[dataset_name]
            fps = data['fps']  # All fingerprints
            y_filtered = data['y']
            mols_filtered = data['mols']
            mols_3d_filtered = data.get('mols_3d', None)
            
            print(f"{dataset_name.upper()}-{split_type} Trial {trial.number}: Starting feature-structure optimization...")
            print(f"  All fingerprints shape: {fps.shape}")
            
            # Step 1: Get best feature selection from module 5 results
            # This loads the optimal descriptor combination found earlier
            best_feature_params = get_best_feature_selection(dataset_name, split_type)
            
            # Convert best_feature_params to selection list using helper function
            # This transforms the dictionary format to a list format needed by the function
            selection = convert_params_to_selection(best_feature_params)
            
            # Step 2: Use selection-based approach to get additional features
            # This function applies the feature selection and returns:
            # - fps_combined: Base fingerprints + selected descriptors
            # - selected_descriptors: List of descriptor names that were selected
            # IMPORTANT: This does NOT duplicate features (fixed from 6336 to ~3609)
            fps_combined, selected_descriptors = selection_data_descriptor_compress(
                selection, fps, mols_filtered, dataset_name,
                target_path=str(OUTPUT_DIR), 
                save_res="np",
                mols_3d=mols_3d_filtered
            )
            
            print(f"  Combined features shape: {fps_combined.shape}")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            
            # Step 3: Structure optimization - suggest hyperparameters including layer count
            # Now that we have the optimal features, find the best network architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)  # 2-4 layers
            hidden_dims = []
            
            # Suggest hidden dimensions for each layer
            # Create a funnel architecture: each layer has fewer or equal neurons
            for i in range(n_layers):
                if i == 0:
                    # First hidden layer: Can be large to capture feature interactions
                    hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', [256, 512, 1024])
                else:
                    # Subsequent layers: Progressive dimension reduction
                    # This creates an information bottleneck that forces the network
                    # to learn increasingly abstract representations
                    prev_dim = hidden_dims[-1]
                    max_dim = min(prev_dim, 512)
                    hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', 
                        [dim for dim in [64, 128, 256, 512] if dim <= max_dim])
                hidden_dims.append(hidden_dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create and save model with hyperparameters
            model = DNNModel(
                input_dim=fps_combined.shape[1], 
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
            os.makedirs("save_model", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': fps_combined.shape[1],
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate
            }, "save_model/full_model.pth")
            
            # Model parameters
            model_params = {
                'batch_size': batch_size,
                'epochs': EPOCHS,  # 10 epochs
                'learning_rate': learning_rate
            }
            
            # Get test data for this dataset and split
            test_info = test_data.get(dataset_name, {}).get(split_type, None)
            X_test, y_test = None, None
            
            if test_info:
                # Apply same feature selection to test data
                test_fps = test_info['fps']
                test_mols = test_info['mols']
                test_mols_3d = test_info.get('mols_3d', None)
                
                # Apply same descriptor selection to test data
                test_fps_combined, _ = selection_data_descriptor_compress(
                    selection, test_fps, test_mols, dataset_name,
                    target_path=str(OUTPUT_DIR), 
                    save_res="np",
                    mols_3d=test_mols_3d
                )
                
                X_test = test_fps_combined
                y_test = np.array(test_info['y'])
                print(f"  Test data shape: {X_test.shape}")
            
            # CV-5 evaluation with test evaluation
            (mean_r2, std_r2, best_r2, 
             mean_rmse, std_rmse, best_rmse,
             mean_mse, std_mse, best_mse,
             mean_mae, std_mae, best_mae,
             test_r2, test_rmse, test_mse, test_mae) = train_and_evaluate_cv(
                fps_combined, y_filtered, model_params, X_test, y_test)
            
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
            trial.set_user_attr('n_features', fps_combined.shape[1])
            trial.set_user_attr('selected_descriptors', selected_descriptors)
            trial.set_user_attr('execution_time', execution_time)
            trial.set_user_attr('memory_used_mb', memory_used)
            trial.set_user_attr('dataset', dataset_name)
            trial.set_user_attr('split_type', split_type)
            trial.set_user_attr('n_layers', n_layers)
            trial.set_user_attr('hidden_dims', hidden_dims)
            trial.set_user_attr('dropout_rate', dropout_rate)
            trial.set_user_attr('learning_rate', learning_rate)
            trial.set_user_attr('batch_size', batch_size)
            
            # Add test metrics
            trial.set_user_attr('test_r2', test_r2)
            trial.set_user_attr('test_rmse', test_rmse)
            trial.set_user_attr('test_mse', test_mse)
            trial.set_user_attr('test_mae', test_mae)
            
            print(f"  Trial completed: CV R2={mean_r2:.4f}±{std_r2:.4f}, Test R2={test_r2:.4f}, Time={execution_time:.2f}s, Memory={memory_used:.1f}MB")
            
            return mean_r2
            
        except Exception as e:
            print(f"Error in {dataset_name.upper()}-{split_type} trial: {e}")
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
                'test_mae': trial.user_attrs.get('test_mae', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    results_file = OUTPUT_DIR / f"{dataset}_{split_type}_feature_structure_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    return results

def main():
    """
    Main function for ANO feature-structure network optimization
    
    This implements the Feature→Structure optimization strategy:
    1. Use features optimized by module 5
    2. Find best network structure for those features
    
    The main loop:
    1. Prepares molecular data for all datasets
    2. For each dataset and split:
       - Loads best features from module 5
       - Optimizes network structure
       - Saves results
    
    Key configuration:
    - DELETE_EXISTING_STUDIES: Set to True to start fresh
    - N_TRIALS: Number of optimization trials (1 for quick run)
    - EPOCHS: Training epochs per trial (100)
    """
    print("Starting ANO Feature-Structure Network Optimization...")
    print(f"Split types: {SPLIT_TYPES}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS}")
    
    # Add option to delete existing studies
    # This is useful when you want to start fresh optimization
    # Set to False to continue from previous runs
    DELETE_EXISTING_STUDIES = True  # Set to True to delete existing studies
    
    # Prepare data
    print("\nPreparing data...")
    prepare_data_for_optimization()
    
    datasets = ['ws', 'de', 'lo', 'hu']
    test_data = PREPARED_DATA.get('TEST_DATA', {})
    
    for split_type in SPLIT_TYPES:
        print(f"\nRunning optimization for split type: {split_type}")
        
        for dataset in datasets:
            print(f"\nProcessing {dataset.upper()} with {split_type} split...")
            
            # Create study name: ano_network_FOMO_{dataset}_{split_type}
            study_name = f"ano_network_FOMO_{dataset}_{split_type}"
            storage_name = "sqlite:///ano_final.db"
            
            # Delete existing study if requested
            if DELETE_EXISTING_STUDIES:
                try:
                    optuna.delete_study(study_name=study_name, storage=storage_name)
                    print(f"  Deleted existing study: {study_name}")
                except KeyError:
                    print(f"  No existing study found: {study_name}")
            
            # Create Optuna study with storage
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage_name,
                study_name=study_name,
                load_if_exists=False  # Always create new study
            )
            
            # Create objective function
            objective_func = create_objective_function(dataset, split_type, test_data)
            
            # Run optimization
            study.optimize(
                objective_func,
                n_trials=N_TRIALS,
                timeout=1800,  # 30 minutes timeout
                show_progress_bar=True
            )
            
            # Print best results
            try:
                print(f"{dataset.upper()}-{split_type} optimization completed!")
                print(f"Best R2 score: {study.best_value:.4f}")
                print(f"Best parameters: {study.best_params}")
                
                # Save results
                save_study_results(study, dataset, split_type)
                
            except ValueError as e:
                print(f"Warning: {dataset.upper()}-{split_type} optimization completed but no valid trials found.")
            
            # Memory cleanup
            gc.collect()
    
    print("\nAll feature-structure network optimizations completed!")

if __name__ == "__main__":
    main()