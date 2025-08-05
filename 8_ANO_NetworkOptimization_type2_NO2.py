#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

ANO Network Structure-Feature Optimization with PyTorch - Fixed Module
======================================================================

PURPOSE:
This module implements the reverse optimization strategy: Structure → Feature.
It uses the optimal network architecture from module 6 and then searches for
the best feature combination for that specific architecture.

APPROACH:
1. Load optimal network structure from module 6 (layer count, hidden dims, etc.)
2. Fix the network architecture with those parameters
3. Search for optimal feature selection using the fixed architecture
4. This ensures features are tailored to the specific network topology

KEY INNOVATIONS:
- Reverse optimization order: Structure first, then features
- Architecture-specific feature selection
- Tests hypothesis that optimal features depend on network structure
- Allows comparison with Feature→Structure approach (module 7)

TECHNICAL DETAILS:
- Uses best structure from module 6: Typically 2-4 layers
- Feature search space: Same as module 5 (51 descriptor categories)
- Base features: 2727 (Morgan + MACCS + Avalon)
- Additional features: Selected from ~882 molecular descriptors
- Evaluation: 5-fold CV on training set

Fixed issues:
- Study name: ano_network_MOFO_{dataset}_{split_type}
- All splits supported (not just rm)
- Epochs: 100 (was 30)
- Trials: 1
- All fingerprints + module 6's best structure used
- Feature selection optimization
- All metrics saved (R², RMSE, MSE, MAE, time, resources)
- Model saved as full_model.pth

Structure → Feature Network:
1. Use all fingerprints (morgan + maccs + avalon)
2. Get best structure from module 6 results
3. Use fixed model structure with best hyperparameters
4. Optimize feature selection for that structure
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
SPLIT_TYPES = ["rm"]
N_TRIALS = 1
EPOCHS = 100  # Fixed to 100 epochs for consistent training
OUTPUT_DIR = Path("result/8_ANO_network_structure_feature_pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global data storage
PREPARED_DATA = {}

def train_and_evaluate_cv(X, y, model_params, X_test=None, y_test=None):
    """
    Train and evaluate with CV-5 and return all metrics
    
    Implements 5-fold cross-validation for robust evaluation. Each fold
    serves as validation once while the other 4 folds are used for training.
    This provides better generalization estimates than single train/test split.
    
    Process:
    1. Split data into 5 stratified folds
    2. For each fold:
       - Train model on 4 folds
       - Validate on remaining fold
       - Collect performance metrics
    3. Aggregate metrics across all folds
    4. If test data provided, train on full training data and evaluate on test
    
    Args:
        X: Feature matrix (n_samples × n_features)
        y: Target values
        model_params: Training parameters (batch_size, epochs, learning_rate)
        X_test: Optional test feature matrix
        y_test: Optional test target values
    
    Returns:
        Tuple of 16 metrics: mean, std, and best for R², RMSE, MSE, MAE + test metrics
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

# PyTorch DNN Model with fixed structure from module 6 results
class DNNModel(nn.Module):
    """
    Deep Neural Network with architecture fixed from module 6 optimization
    
    This model uses the optimal architecture found in module 6, which was
    determined through extensive hyperparameter search. The architecture is
    fixed, and only the feature selection varies during optimization.
    
    Architecture components:
    - SiLU activation: Smooth, self-gated activation function
    - BatchNorm: Stabilizes training and improves convergence
    - Dropout: Prevents overfitting through random neuron deactivation
    - Xavier initialization: Ensures proper gradient flow
    
    The key difference from module 7 is that here the architecture is
    predetermined, allowing focus on feature optimization.
    
    Args:
        input_dim: Number of selected features (varies per trial)
        hidden_dims: Fixed hidden layer dimensions from module 6
        dropout_rate: Fixed dropout rate from module 6
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
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

def prepare_data_for_optimization():
    """
    Prepare data for optimization using preprocessed train data
    
    Loads and prepares molecular data for all datasets. This includes:
    1. Loading preprocessed SMILES and target values
    2. Removing duplicates across splits
    3. Generating 3D conformers for 3D descriptor calculation
    4. Computing base molecular fingerprints
    
    The prepared data is stored globally to avoid recomputation
    during optimization trials.
    
    Side effects:
        Updates global PREPARED_DATA dictionary
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

def get_best_structure_selection(dataset_name, split_type):
    """
    Get best structure from module 6 results
    
    Loads the optimal neural network architecture discovered by module 6
    from the Optuna database. Module 6 performed extensive architecture
    search to find the best network topology for each dataset.
    
    The loaded parameters include:
    - n_layers: Number of hidden layers (2-4)
    - hidden_dim_i: Neurons in each hidden layer
    - dropout_rate: Dropout probability
    - learning_rate: Optimization learning rate
    - batch_size: Training batch size
    
    Args:
        dataset_name: Dataset identifier
        split_type: Data splitting strategy
    
    Returns:
        Dictionary of optimal hyperparameters
        Falls back to sensible defaults if loading fails
    """
    try:
        # Load study from Optuna storage - 6번 모듈 결과
        study_name = f"ano_structure_{dataset_name}_{split_type}"
        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///ano_final.db"
        )
        
        # Get best parameters
        best_params = study.best_params
        print(f"Successfully loaded best structure for {dataset_name}-{split_type}")
        return best_params
        
    except Exception as e:
        print(f"Error loading best structure for {dataset_name}-{split_type}: {e}")
        # Return default parameters if loading fails
        return {
            'n_layers': 2,
            'hidden_dim_0': 512,
            'hidden_dim_1': 256,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32
        }

def create_objective_function(dataset_name, split_type, test_data):
    """
    Create objective function for Structure → Feature network optimization
    
    This implements the Structure→Feature optimization strategy:
    1. Fix the network architecture using module 6's best structure
    2. Search for optimal features for that specific architecture
    
    The hypothesis is that different network architectures may benefit
    from different feature combinations. By fixing the architecture first,
    we can find features specifically tailored to that topology.
    
    Key steps in each trial:
    1. Load best network structure from module 6
    2. Suggest feature selection (51 binary choices)
    3. Create model with fixed architecture but variable input size
    4. Train and evaluate using 5-fold CV
    5. Return CV performance for optimization
    
    Args:
        dataset_name: Dataset to optimize for
        split_type: Data splitting strategy
    
    Returns:
        Objective function closure for Optuna
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
            
            print(f"{dataset_name.upper()}-{split_type} Trial {trial.number}: Starting structure-feature optimization...")
            print(f"  All fingerprints shape: {fps.shape}")
            
            # Step 1: Get best structure from module 6 results
            # This loads the optimal architecture found through extensive search
            best_structure_params = get_best_structure_selection(dataset_name, split_type)
            
            # Step 2: Extract hidden dimensions from best structure
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
            
            dropout_rate = best_structure_params.get('dropout_rate', 0.2)
            learning_rate = best_structure_params.get('learning_rate', 0.001)
            batch_size = best_structure_params.get('batch_size', 32)
            
            # Step 3: Feature selection optimization using search_data_descriptor_compress_fixed
            # With the architecture fixed, search for the best feature combination
            # This explores 51 binary choices for molecular descriptor categories
            fps_selected, selected_descriptors, excluded_descriptors = search_data_descriptor_compress_fixed(
                trial, fps, mols_filtered, dataset_name, str(OUTPUT_DIR), "np", mols_3d_filtered
            )
            
            print(f"  Selected features shape: {fps_selected.shape}")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            
            # Step 4: Create and save model with fixed structure
            # The model uses the predetermined architecture from module 6
            # Only the input dimension varies based on feature selection
            model = DNNModel(
                input_dim=fps_selected.shape[1], 
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )
            os.makedirs("save_model", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': fps_selected.shape[1],
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate
            }, "save_model/full_model.pth")
            
            # Model parameters (using best structure from 6번)
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
                
                # Get same feature selection for test data
                selection_params = trial.params
                selection = convert_params_to_selection(selection_params)
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
                fps_selected, y_filtered, model_params, X_test, y_test)
            
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
                'test_mae': trial.user_attrs.get('test_mae', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    results_file = OUTPUT_DIR / f"{dataset}_{split_type}_structure_feature_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    return results

def main():
    """
    Main function for ANO structure-feature network optimization
    
    Implements the Structure→Feature optimization strategy:
    1. Use the best network architecture from module 6
    2. Search for optimal features for that architecture
    
    This approach tests whether feature selection should be
    architecture-dependent. Results can be compared with:
    - Module 5: Feature selection with fixed simple architecture
    - Module 7: Feature→Structure optimization
    
    The comparison helps determine the best optimization strategy
    for molecular property prediction models.
    """
    print("Starting ANO Structure-Feature Network Optimization...")
    print(f"Split types: {SPLIT_TYPES}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS}")
    
    # Prepare data
    print("\nPreparing data...")
    prepare_data_for_optimization()
    
    datasets = ['ws', 'de', 'lo', 'hu']
    test_data = PREPARED_DATA.get('TEST_DATA', {})
    
    for split_type in SPLIT_TYPES:
        print(f"\nRunning optimization for split type: {split_type}")
        
        for dataset in datasets:
            print(f"\nProcessing {dataset.upper()} with {split_type} split...")
            
            # Create study name: ano_network_MOFO_{dataset}_{split_type}
            study_name = f"ano_network_MOFO_{dataset}_{split_type}"
            storage_name = "sqlite:///ano_final.db"
            
            # Create Optuna study with storage
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                storage=storage_name,
                study_name=study_name,
                load_if_exists=True
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
    
    print("\nAll structure-feature network optimizations completed!")

if __name__ == "__main__":
    main()