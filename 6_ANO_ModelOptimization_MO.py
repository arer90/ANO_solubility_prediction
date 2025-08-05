#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

ANO Structure Optimization with PyTorch - Using Preprocessed Train/Test Data with CV-5
======================================================================================

PURPOSE:
This module implements neural network architecture search for molecular property prediction.
It optimizes the structure (layers, hidden units, dropout, etc.) while keeping features fixed,
finding the best network topology for the given molecular representation.

APPROACH:
1. Use fixed molecular fingerprints (Morgan + MACCS + Avalon = 2727 features)
2. Search optimal DNN architecture using Optuna hyperparameter optimization
3. Variable network depth: 2-4 layers with decreasing hidden units
4. Optimize: layer count, hidden dimensions, dropout rate, learning rate, batch size
5. Evaluate using 5-fold CV on training set, final test on hold-out set

KEY INNOVATIONS:
- Dynamic layer count optimization (not fixed 3-layer architecture)
- Hierarchical hidden dimension selection (each layer ≤ previous layer)
- Uses all three fingerprint types for comprehensive molecular representation
- Separate CV performance tracking to prevent overfitting to test set

TECHNICAL DETAILS:
- Input features: 2048 (Morgan) + 167 (MACCS) + 512 (Avalon) = 2727
- Layer range: 2-4 layers (dynamically optimized)
- Hidden units: First layer [256, 512, 1024], subsequent layers [64, 128, 256, 512]
- Activation: ReLU throughout (simple but effective)
- Regularization: Dropout (0.1-0.5) between layers
- Output: Single unit for regression

Updates:
- Uses preprocessed train/test data from result/1_preprocess/
- CV-5 on train data for robust evaluation
- Final test evaluation on hold-out test set
- Storage: ano_final.db
- Study name: ano_structure_{dataset}_{split_type}
- Layer count hyperparameter tuning (2-4 layers)
"""

import os
import sys
import time
import gc
import json
import subprocess
import numpy as np
import pandas as pd
import optuna
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import AllChem

# Import from molecular_loader_with_fp
from extra_code.molecular_loader_with_fp import get_fingerprints

# Configuration
# Enable all split types for comprehensive analysis
SPLIT_TYPES = ["rm", "sc", "cs", "cl", "pc", "ac", "sa", "ti", "en"]
N_TRIALS = 1
EPOCHS = 100  # Fixed to 100 epochs for consistent training
OUTPUT_DIR = Path("result/6_ANO_structure_all_pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_NAME = "sqlite:///ano_final.db"

# Dataset name mapping
DATASET_MAPPING = {
    'ws': 'ws496_logS',
    'de': 'delaney-processed',
    'lo': 'Lovric2020_logS0',
    'hu': 'huusk'
}

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
    dataset_full = DATASET_MAPPING[dataset_short]
    
    # Train data path - now loading from data folder
    train_path = Path(f"data/train/{split_type}/{split_type}_{dataset_full}_train.csv")
    test_path = Path(f"data/test/{split_type}/{split_type}_{dataset_full}_test.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found for {dataset_short}-{split_type}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df['smiles'].tolist(), train_df['target'].tolist(), \
           test_df['smiles'].tolist(), test_df['target'].tolist()

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
        
        # Calculate fingerprints (all: morgan + maccs + avalon)
        train_morgan, train_maccs, train_avalon = get_fingerprints(train_mols_filtered)
        train_fps = np.hstack([train_morgan, train_maccs, train_avalon])
        
        test_morgan, test_maccs, test_avalon = get_fingerprints(test_mols_filtered)
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

def train_and_evaluate_cv(X_train_full, y_train_full, X_test, y_test, model_params):
    """
    Train with CV-5 on train set and final evaluation on test set
    
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
        X_train_full: Full training features (n_samples × n_features)
        y_train_full: Full training targets
        X_test: Test features
        y_test: Test targets
        model_params: Dictionary with batch_size, epochs, learning_rate
    
    Returns:
        Tuple of 16 metrics: CV mean/std/best for R², RMSE, MSE, MAE + test metrics
    """
    
    # Convert to numpy arrays
    X_train_full = np.asarray(X_train_full, dtype=np.float32)
    y_train_full = np.asarray(y_train_full, dtype=np.float32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).flatten()
    
    # CV-5 on training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mse_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_train = X_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_train = y_train_full[train_idx]
        y_val = y_train_full[val_idx]
        
        # Save temporary files for subprocess
        np.save("temp_X_train.npy", X_train)
        np.save("temp_y_train.npy", y_train)
        np.save("temp_X_test.npy", X_val)
        np.save("temp_y_test.npy", y_val)
        
        # Run training subprocess
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
    
    print(f"CV Results: R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}, RMSE={cv_rmse_mean:.4f}±{cv_rmse_std:.4f}")
    
    # Train final model on full training data and evaluate on test set
    np.save("temp_X_train.npy", X_train_full)
    np.save("temp_y_train.npy", y_train_full)
    np.save("temp_X_test.npy", X_test)
    np.save("temp_y_test.npy", y_test)
    
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
    test_r2, test_rmse, test_mse, test_mae = 0.0, 0.0, 0.0, 0.0
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
    
    return (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
            cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
            test_r2, test_rmse, test_mse, test_mae)

# DNN Model with hyperparameter tuning including number of layers
class SimpleDNN(nn.Module):
    """
    Simple Deep Neural Network with variable architecture
    
    This model implements a flexible feedforward neural network where the
    number of layers and their dimensions are determined by hyperparameter
    optimization. Each layer follows the pattern: Linear -> ReLU -> Dropout.
    
    Architecture details:
    - Variable depth: 2-4 layers (optimized)
    - Hidden dimensions: Decreasing or equal sizes (e.g., 1024->256->64)
    - Activation: ReLU (computationally efficient, avoids vanishing gradients)
    - Regularization: Dropout after each hidden layer
    - Output: Single unit for regression (no activation)
    
    Args:
        input_dim: Number of input features (2727 for combined fingerprints)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability (0.1-0.5)
    """
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2):
        super(SimpleDNN, self).__init__()
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

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
    - Subsequent layers must be ≤ previous layer size
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
            
            print(f"{dataset_name.upper()} Trial {trial.number}: Starting structure optimization...")
            print(f"  Train shape: {train_fps.shape}, Test shape: {test_fps.shape}")
            
            # Suggest hyperparameters for structure optimization
            # Key innovation: Dynamic layer count (not fixed 3 layers)
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_dims = []
            
            # Suggest hidden dimensions for each layer
            # Architecture constraint: Create funnel-shaped network
            # Each layer should progressively reduce dimensionality
            for i in range(n_layers):
                if i == 0:
                    # First layer: Can be large to capture complex patterns
                    hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', [256, 512, 1024])
                else:
                    # Subsequent layers: Must be smaller or equal to previous
                    # This creates information bottleneck effect
                    prev_dim = hidden_dims[-1]
                    max_dim = min(prev_dim, 512)
                    hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', 
                        [dim for dim in [64, 128, 256, 512] if dim <= max_dim])
                hidden_dims.append(hidden_dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create and save model with hyperparameters
            model = SimpleDNN(input_dim=train_fps.shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate)
            os.makedirs("save_model", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': train_fps.shape[1],
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate
            }, "save_model/full_model.pth")
            
            # Model parameters
            model_params = {
                'batch_size': batch_size,
                'epochs': EPOCHS,
                'learning_rate': learning_rate
            }
            
            # Train with CV-5 and evaluate on test
            (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
             cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
             test_r2, test_rmse, test_mse, test_mae) = train_and_evaluate_cv(
                train_fps, train_y, test_fps, test_y, model_params
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
            trial.set_user_attr('n_layers', n_layers)
            trial.set_user_attr('hidden_dims', hidden_dims)
            trial.set_user_attr('dropout_rate', dropout_rate)
            trial.set_user_attr('learning_rate', learning_rate)
            trial.set_user_attr('batch_size', batch_size)
            trial.set_user_attr('n_features', train_fps.shape[1])
            trial.set_user_attr('execution_time', execution_time)
            trial.set_user_attr('memory_used_mb', memory_used)
            trial.set_user_attr('dataset', dataset_name)
            trial.set_user_attr('split_type', split_type)
            
            print(f"  Trial completed: CV R2={cv_r2_mean:.4f}±{cv_r2_std:.4f}, Test R2={test_r2:.4f}, Time={execution_time:.2f}s")
            
            return cv_r2_mean  # Optimize based on CV performance
            
        except Exception as e:
            print(f"Error in {dataset_name.upper()} trial: {e}")
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
                'n_layers': trial.user_attrs.get('n_layers', 0),
                'hidden_dims': trial.user_attrs.get('hidden_dims', []),
                'dropout_rate': trial.user_attrs.get('dropout_rate', 0.0),
                'learning_rate': trial.user_attrs.get('learning_rate', 0.0),
                'batch_size': trial.user_attrs.get('batch_size', 0),
                'execution_time': trial.user_attrs.get('execution_time', 0.0),
                'memory_used_mb': trial.user_attrs.get('memory_used_mb', 0.0)
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    results_file = OUTPUT_DIR / f"{dataset}_{split_type}_structure_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
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
    print("Starting ANO Structure Optimization with Preprocessed Data...")
    print(f"Split types: {SPLIT_TYPES}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS}")
    print(f"Storage: {STORAGE_NAME}")
    
    datasets = ['ws', 'de', 'lo', 'hu']
    
    for split_type in SPLIT_TYPES:
        print(f"\nRunning optimization for split type: {split_type}")
        
        for dataset in datasets:
            print(f"\nProcessing {dataset.upper()} with {split_type} split...")
            
            try:
                # Create study name
                study_name = f"ano_structure_{dataset}_{split_type}"
                
                # Create Optuna study
                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    storage=STORAGE_NAME,
                    study_name=study_name,
                    load_if_exists=True
                )
                
                # Create objective function
                objective_func = create_objective_function(dataset, split_type)
                
                # Run optimization
                study.optimize(
                    objective_func,
                    n_trials=N_TRIALS,
                    timeout=1800,
                    show_progress_bar=True
                )
                
                # Print results
                print(f"{dataset.upper()}-{split_type} optimization completed!")
                print(f"Best CV R2 score: {study.best_value:.4f}")
                print(f"Best parameters: {study.best_params}")
                
                # Save results
                save_study_results(study, dataset, split_type)
                
            except Exception as e:
                print(f"Error processing {dataset.upper()}-{split_type}: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
    
    print("\nAll structure optimizations completed!")

if __name__ == "__main__":
    main()