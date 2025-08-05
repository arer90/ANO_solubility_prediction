#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

ANO Feature Selection with PyTorch - Using Preprocessed Train/Test Data with CV-5
================================================================================

PURPOSE:
This module implements automated feature selection for molecular property prediction.
It searches for the optimal combination of molecular descriptors and fingerprints
to maximize prediction accuracy while minimizing feature redundancy.

APPROACH:
1. Start with base fingerprints (Morgan, MACCS, Avalon) = 2727 features
2. Use Optuna to search through 51 different molecular descriptor categories
3. Each descriptor category can be included (1) or excluded (0) 
4. Train a simple DNN model with selected features using 5-fold CV
5. Optimize based on mean CV R² score to avoid overfitting

KEY INNOVATIONS:
- Binary selection approach: each descriptor type is fully included or excluded
- Combines multiple fingerprint types (Morgan, MACCS, Avalon) as baseline
- Uses both 2D and 3D molecular descriptors when available
- Employs early stopping via CV performance to prevent overfitting

TECHNICAL DETAILS:
- Base features: 2048 (Morgan) + 167 (MACCS) + 512 (Avalon) = 2727
- Additional descriptors: ~51 categories yielding ~882 features when all selected
- Final feature count: typically 3000-4000 features after selection
- Model: 3-layer DNN (input → 1024 → 496 → 1)
- Training: Adam optimizer, 100 epochs, batch size 32
- Evaluation: 5-fold CV on training set, final test on hold-out set

Updates:
- Uses preprocessed train/test data from result/1_preprocess/
- CV-5 on train data for robust evaluation
- Final test evaluation on hold-out test set
- Storage: ano_final.db
- Study name: ano_feature_{dataset}_{split_type}
"""

import os
import sys
import time
import psutil
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import subprocess
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add the extra_code directory to the path
sys.path.append('extra_code')

# Import required modules
from ano_feature_search import search_data_descriptor_compress_fixed, prefilter_3d_conformers
from molecular_loader_with_fp import get_fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem

# Configuration
# Enable all split types for comprehensive analysis
SPLIT_TYPES = ["rm", "sc", "cs", "cl", "pc", "ac", "sa", "ti", "en"]
N_TRIALS = 1
EPOCHS = 100  # Fixed to 100 epochs for consistent training
OUTPUT_DIR = Path("result/5_ANO_feature_all_pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_NAME = "sqlite:///ano_final.db"

# Dataset name mapping
DATASET_MAPPING = {
    'ws': 'ws496_logS',
    'de': 'delaney-processed',
    'lo': 'Lovric2020_logS0',
    'hu': 'huusk'
}

class SimpleDNN(nn.Module):
    """Simple DNN model for regression - output dim = 1"""
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, 496),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(496, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def create_and_save_model(input_dim, model_path="save_model/full_model.pth"):
    """Create and save initial model as full_model.pth"""
    os.makedirs("save_model", exist_ok=True)
    
    model = SimpleDNN(input_dim)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim
    }, model_path)
    
    return model

def load_preprocessed_data(dataset_short, split_type):
    """Load preprocessed train/test data"""
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

def train_and_evaluate_cv(X_train_full, y_train_full, X_test, y_test, model_params, selected_features=None):
    """Train with CV-5 on train set and final evaluation on test set"""
    
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

def prepare_data_for_split(dataset_short, split_type):
    """Prepare data for a specific dataset and split"""
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
        
        # Generate 3D conformers for molecules
        train_mols_3d = []
        for mol in train_mols_filtered:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != -1:
                train_mols_3d.append(mol_3d)
            else:
                train_mols_3d.append(None)
        
        test_mols_3d = []
        for mol in test_mols_filtered:
            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != -1:
                test_mols_3d.append(mol_3d)
            else:
                test_mols_3d.append(None)
        
        # Calculate fingerprints
        train_morgan, train_maccs, train_avalon = get_fingerprints(train_mols_filtered)
        train_fps = np.hstack([train_morgan, train_maccs, train_avalon])
        
        test_morgan, test_maccs, test_avalon = get_fingerprints(test_mols_filtered)
        test_fps = np.hstack([test_morgan, test_maccs, test_avalon])
        
        print(f"  Train fingerprint shape: {train_fps.shape}")
        print(f"  Test fingerprint shape: {test_fps.shape}")
        
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

def create_objective_function(dataset_name, split_type):
    """Create objective function for given dataset and split"""
    def objective_function(trial):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Prepare data for this specific split
            data = prepare_data_for_split(dataset_name, split_type)
            
            train_fps = data['train_fps']
            train_y = data['train_y']
            train_mols = data['train_mols']
            train_mols_3d = data['train_mols_3d']
            
            print(f"{dataset_name.upper()} Trial {trial.number}: Starting feature selection...")
            print(f"  Train input shape: {train_fps.shape}")
            
            # Feature selection on training data
            train_fps_selected, selected_descriptors, excluded_descriptors = search_data_descriptor_compress_fixed(
                trial, train_fps, train_mols, dataset_name, str(OUTPUT_DIR), "np", train_mols_3d
            )
            
            print(f"  Selected features: {train_fps_selected.shape[1]} (from {train_fps.shape[1]})")
            print(f"  Selected descriptors: {len(selected_descriptors)}")
            
            # Apply same feature selection to test data
            test_fps = data['test_fps']
            test_y = data['test_y']
            test_mols = data['test_mols']
            test_mols_3d = data['test_mols_3d']
            
            # Use selection_data_descriptor_compress for test data
            from feature_selection_combined import selection_data_descriptor_compress, convert_params_to_selection
            selection = convert_params_to_selection(trial.params)
            test_fps_selected, _ = selection_data_descriptor_compress(
                selection, test_fps, test_mols, dataset_name,
                target_path=str(OUTPUT_DIR), save_res="np", mols_3d=test_mols_3d
            )
            
            # Model parameters
            model_params = {
                'batch_size': 32,
                'epochs': EPOCHS,
                'learning_rate': 0.001
            }
            
            # Train with CV-5 and evaluate on test
            (cv_r2_mean, cv_r2_std, best_r2, cv_rmse_mean, cv_rmse_std, best_rmse,
             cv_mse_mean, cv_mse_std, best_mse, cv_mae_mean, cv_mae_std, best_mae,
             test_r2, test_rmse, test_mse, test_mae) = train_and_evaluate_cv(
                train_fps_selected, train_y, test_fps_selected, test_y, model_params
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
            trial.set_user_attr('n_features', train_fps_selected.shape[1])
            trial.set_user_attr('selected_descriptors', selected_descriptors)
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
                'n_features': trial.user_attrs.get('n_features', 0),
                'execution_time': trial.user_attrs.get('execution_time', 0.0),
                'memory_used_mb': trial.user_attrs.get('memory_used_mb', 0.0),
                'selected_descriptors': trial.user_attrs.get('selected_descriptors', [])
            }
            results['trial_details'].append(trial_detail)
    
    # Save to file
    import json
    results_file = OUTPUT_DIR / f"{dataset}_{split_type}_feature_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    return results

def main():
    """Main function"""
    print("Starting ANO Feature Selection with Preprocessed Data...")
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
                # Create initial model with dummy dimensions
                create_and_save_model(4096)  # Default fingerprint size
                
                # Create study name
                study_name = f"ano_feature_{dataset}_{split_type}"
                
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
                study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=False)
                
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
    
    print("\nAll feature selection optimizations completed!")

if __name__ == "__main__":
    main()