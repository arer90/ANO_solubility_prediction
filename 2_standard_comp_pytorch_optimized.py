import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gc
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import polars as pl
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import tempfile
import shutil

# RDKit imports
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

# PyTorch imports with optimizations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# PyTorch device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Set memory fraction for PyTorch
    torch.cuda.set_per_process_memory_fraction(0.8)

# Import custom modules
from pathlib import Path
from extra_code.molecular_loader_with_fp import (
    load_split_data,
    extract_xy_from_data,
    build_fingerprints_for_splits
)

# Configuration
target_path = "result/2_standard_comp"
os.makedirs(target_path, exist_ok=True)
out_root = Path("./result/fingerprint")
out_root.mkdir(parents=True, exist_ok=True)

# Create temp directory for intermediate files
TEMP_DIR = Path(tempfile.mkdtemp(prefix="standard_comp_"))
print(f"Using temporary directory: {TEMP_DIR}")

# Model hyperparameters
RANDOM_STATE = 42
VAL_SIZE = 0.05
EPOCHS = 100
BATCHSIZE = 16
REGULARIZER = 1e-5
lr = 0.001
CV = 5

# Chunk size for processing large datasets
CHUNK_SIZE = 10000

# Split mapping
SPLIT_MAP = {
    "rm": "random",
    "cs": "chemical_space_coverage", 
    "cl": "cluster",
    "pc": "physchem",
    "ac": "activity_cliff",
    "sa": "solubility_aware",
    "en": "ensemble"
}

# Dataset abbreviation mapping
DATASET_MAP = {
    "delaney-processed": "de",
    "Lovric2020_logS0": "lo",
    "ws496_logS": "ws",
    "huusk": "hu"
}

# Font sizes for publication quality
FONT_SIZES = {
    'title': 30,
    'subtitle': 24,
    'label': 26,
    'tick': 20,
    'legend': 22,
    'legend_small': 18
}

# PyTorch DNN Model with LeakyReLU
class DNNModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2, negative_slope=0.01):
        super(DNNModel, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(1024, 496),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.BatchNorm1d(496),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(496, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

def save_temp_data(data, filename):
    """Save data to temporary file"""
    filepath = TEMP_DIR / filename
    np.save(filepath, data)
    return filepath

def load_temp_data(filename):
    """Load data from temporary file"""
    filepath = TEMP_DIR / filename
    if filepath.exists():
        data = np.load(filepath)
        return data
    return None

def cleanup_temp_files():
    """Clean up temporary directory"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DIR}")

def process_data_in_chunks(data, chunk_size=CHUNK_SIZE):
    """Process large data arrays in chunks to manage memory"""
    n_samples = len(data)
    for i in range(0, n_samples, chunk_size):
        yield data[i:i + chunk_size]

def new_model(input_shape):
    """Create a new PyTorch model with the same architecture"""
    model = DNNModel(input_shape)
    return model

def save_model(x_data, verbose: bool = False) -> Path:
    """Save model architecture for PyTorch"""
    model_dir = Path("save_model")
    model_path = model_dir / "pytorch_model.pth"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    input_dim = x_data.shape[1]
    model = new_model(input_dim)
    
    # Save model state dict (weights will be loaded later)
    torch.save({
        'input_dim': input_dim,
        'model_state_dict': model.state_dict(),
    }, model_path)
    
    if verbose:
        print(f"[save_model] Model saved at {model_path} with input_dim={input_dim}")
    
    return model_path

def metric_prediction(y_true, y_pred):
    """Calculate regression metrics"""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }

def train_dnn_pytorch_optimized(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lr=0.001, device='cuda'):
    """Train DNN model using PyTorch with optimizations"""
    try:
        # Save data to temp files for memory efficiency
        save_temp_data(X_train, 'X_train_temp.npy')
        save_temp_data(y_train, 'y_train_temp.npy')
        save_temp_data(X_test, 'X_test_temp.npy')
        save_temp_data(y_test, 'y_test_temp.npy')
        
        # Clear original arrays
        del X_train, y_train
        gc.collect()
        
        # Load data back as tensors
        X_train = torch.FloatTensor(load_temp_data('X_train_temp.npy'))
        y_train = torch.FloatTensor(load_temp_data('y_train_temp.npy')).reshape(-1, 1)
        X_test = torch.FloatTensor(load_temp_data('X_test_temp.npy'))
        y_test = torch.FloatTensor(load_temp_data('y_test_temp.npy')).reshape(-1, 1)
        
        # Create model
        input_dim = X_train.shape[1]
        model = DNNModel(input_dim).to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Mixed precision training
        scaler = GradScaler() if device == 'cuda' else None
        
        # Create data loaders with pin_memory for faster GPU transfer
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if device == 'cuda' else False,
            num_workers=2
        )
        
        # Training loop with mixed precision
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if device == 'cuda' and scaler is not None:
                    # Mixed precision training
                    with autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Clear cache periodically
                if batch_count % 50 == 0 and device == 'cuda':
                    torch.cuda.empty_cache()
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(train_loader)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                
                # Garbage collection
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Process test data in chunks for large datasets
            if len(X_test) > CHUNK_SIZE:
                y_pred_list = []
                for i in range(0, len(X_test), CHUNK_SIZE):
                    chunk_end = min(i + CHUNK_SIZE, len(X_test))
                    X_chunk = X_test[i:chunk_end].to(device)
                    y_pred_chunk = model(X_chunk).cpu().numpy()
                    y_pred_list.append(y_pred_chunk)
                    
                    # Clear GPU memory
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                
                y_pred = np.vstack(y_pred_list)
            else:
                X_test_device = X_test.to(device)
                y_pred = model(X_test_device).cpu().numpy()
            
            y_test_np = y_test.numpy()
            
            # Calculate metrics
            r2 = r2_score(y_test_np, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
            mae = mean_absolute_error(y_test_np, y_pred)
            
            metrics = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mse': rmse ** 2
            }
            
            print(f"R2: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        # Clean up
        del model, X_train, y_train, X_test, y_test
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return y_pred.flatten(), metrics
        
    except Exception as e:
        print(f"Error in PyTorch DNN training: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(y_test)), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

def train_model_simple(model_type, X_train, y_train, X_test, y_test):
    """Train model with simple train/test split - DNN uses PyTorch"""
    
    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    if model_type == 'DNN':
        # Use optimized PyTorch implementation
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_pred, _ = train_dnn_pytorch_optimized(
            X_train, y_train, X_test, y_test,
            epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr, device=device_str
        )
        return y_pred, None
    else:
        # Traditional ML models
        models = {
            'Ridge': Ridge(),
            'SVR': SVR(),
            'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(
                random_state=RANDOM_STATE, 
                n_jobs=-1, 
                verbosity=0,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'auto'
            ),
            'LightGBM': lgb.LGBMRegressor(
                random_state=RANDOM_STATE, 
                n_jobs=-1, 
                verbose=-1,
                enable_categorical=False,
                force_row_wise=True,
                device='gpu' if torch.cuda.is_available() else 'cpu'
            )
        }        
        model = models[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Clean up
        del model
        gc.collect()
        
        return y_pred, None

def train_model_cv(model_type, X_train, y_train, X_test, y_test, n_folds=5):
    """Train model with k-fold cross-validation - DNN uses PyTorch"""
    
    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    cv_predictions = []
    test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{n_folds}", end='\r')
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        if model_type == 'DNN':
            # Use optimized PyTorch implementation
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            test_pred, _ = train_dnn_pytorch_optimized(
                X_tr, y_tr, X_test, y_test,
                epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr, device=device_str
            )
            test_predictions.append(test_pred)
            
        else:
            # Traditional ML models with GPU support where available
            models = {
                'Ridge': Ridge(),
                'SVR': SVR(),
                'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(
                    random_state=RANDOM_STATE, 
                    n_jobs=-1, 
                    verbosity=0,
                    tree_method='gpu_hist' if torch.cuda.is_available() else 'auto'
                ),
                'LightGBM': lgb.LGBMRegressor(
                    random_state=RANDOM_STATE, 
                    n_jobs=-1, 
                    verbose=-1,
                    enable_categorical=False,
                    force_row_wise=True,
                    device='gpu' if torch.cuda.is_available() else 'cpu'
                )
            }
            
            model = models[model_type]
            model.fit(X_tr, y_tr)
            
            # Predict on test set
            test_pred = model.predict(X_test)
            test_predictions.append(test_pred)
            
            # Clean up
            del model
            gc.collect()
    
    # Average predictions across folds
    y_pred = np.mean(test_predictions, axis=0)
    
    # Clean up
    del test_predictions
    gc.collect()
    
    return y_pred, cv_predictions

def run_full_experiment_with_predictions(fp_map, y_map, output_dir="result/2_standard_comp"):
    """Run experiments and save predictions for visualization"""
    
    all_results = []
    all_predictions = {}  # Store predictions for later visualization
    
    # Model types to test
    model_types = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
    
    # Fingerprint types
    fp_types = ['morgan','maccs', 'avalon']
    
    # Get all available split types
    available_splits = list(fp_map.keys())
    
    total_combinations = 0
    completed_combinations = 0
    
    # Count total combinations first
    for split_type in available_splits:
        datasets = set()
        for key in fp_map[split_type].keys():
            dataset = key.split('_')[0]
            datasets.add(dataset)
        
        for dataset in datasets:
            train_key = f"{dataset}_train"
            test_key = f"{dataset}_test"
            if train_key in fp_map[split_type] and test_key in fp_map[split_type]:
                total_combinations += len(fp_types) * len(model_types)
    
    print(f"Total combinations to run: {total_combinations}")
    
    # Process each split type
    for split_idx, split_type in enumerate(available_splits):
        print(f"\n{'='*60}")
        print(f"Processing split {split_idx+1}/{len(available_splits)}: {split_type} ({SPLIT_MAP.get(split_type, 'unknown')})")
        print(f"{'='*60}")
        
        # Initialize predictions storage for this split
        if split_type not in all_predictions:
            all_predictions[split_type] = {}
        
        datasets = set()
        for key in fp_map[split_type].keys():
            dataset = key.split('_')[0]
            datasets.add(dataset)
        
        for dataset in datasets:
            train_key = f"{dataset}_train"
            test_key = f"{dataset}_test"
            
            if train_key not in fp_map[split_type] or test_key not in fp_map[split_type]:
                print(f"Skipping {dataset}: missing train or test data")
                continue
            
            print(f"\nDataset: {dataset}")
            
            # Initialize dataset predictions storage
            if dataset not in all_predictions[split_type]:
                all_predictions[split_type][dataset] = {}
            
            # Get target values
            y_train = np.array(y_map[split_type][train_key])
            y_test = np.array(y_map[split_type][test_key])
            
            print(f"  Train size: {len(y_train)}, Test size: {len(y_test)}")
            
            for fp_type in fp_types:
                print(f"  Fingerprint: {fp_type}")
                
                if fp_type not in all_predictions[split_type][dataset]:
                    all_predictions[split_type][dataset][fp_type] = {}
                
                # Get fingerprints
                X_train = fp_map[split_type][train_key][fp_type]
                X_test = fp_map[split_type][test_key][fp_type]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                for model_type in model_types:
                    completed_combinations += 1
                    progress = (completed_combinations / total_combinations) * 100
                    print(f"    Model: {model_type} ({completed_combinations}/{total_combinations} - {progress:.1f}%)")
                    
                    # Choose scaled or unscaled features based on model
                    if model_type in ['SVR', 'Ridge']:
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    # Simple training
                    try:
                        start_time = time.time()
                        y_pred_simple, _ = train_model_simple(model_type, X_tr, y_train, X_te, y_test)
                        end_time = time.time()
                        
                        # Store predictions
                        all_predictions[split_type][dataset][fp_type][f"{model_type}_simple"] = {
                            'y_true': y_test,
                            'y_pred': y_pred_simple
                        }
                        
                        metrics_simple = metric_prediction(y_test, y_pred_simple)
                        
                        result_simple = {
                            'split': split_type,
                            'dataset': dataset,
                            'fingerprint': fp_type,
                            'model': model_type,
                            'training': 'simple',
                            'train_time': end_time - start_time,
                            **metrics_simple
                        }
                        all_results.append(result_simple)
                        print(f"      Simple training - R²: {metrics_simple['r2']:.3f}, RMSE: {metrics_simple['rmse']:.3f}")
                        
                    except Exception as e:
                        print(f"      Error in simple training: {e}")
                        result_simple = {
                            'split': split_type,
                            'dataset': dataset,
                            'fingerprint': fp_type,
                            'model': model_type,
                            'training': 'simple',
                            'r2': np.nan,
                            'rmse': np.nan,
                            'mae': np.nan,
                            'mse': np.nan,
                            'error': str(e)
                        }
                        all_results.append(result_simple)
                    
                    # CV training
                    try:
                        start_time = time.time()
                        y_pred_cv, cv_preds = train_model_cv(model_type, X_tr, y_train, X_te, y_test, n_folds=CV)
                        end_time = time.time()
                        
                        # Store CV predictions
                        all_predictions[split_type][dataset][fp_type][f"{model_type}_cv"] = {
                            'y_true': y_test,
                            'y_pred': y_pred_cv,
                            'cv_preds': cv_preds  # Store individual fold predictions too
                        }
                        
                        metrics_cv = metric_prediction(y_test, y_pred_cv)
                        
                        # Calculate std from fold predictions
                        if cv_preds:
                            cv_pred_array = np.array(cv_preds)
                            y_pred_std = np.std(cv_pred_array, axis=0)
                            mean_std = np.mean(y_pred_std)
                            
                            fold_metrics = []
                            for fold_pred in cv_preds:
                                fold_metric = metric_prediction(y_test, fold_pred)
                                fold_metrics.append(fold_metric)
                            
                            r2_std = np.std([m['r2'] for m in fold_metrics])
                            rmse_std = np.std([m['rmse'] for m in fold_metrics])
                            mae_std = np.std([m['mae'] for m in fold_metrics])
                        else:
                            mean_std = r2_std = rmse_std = mae_std = 0.0
                        
                        result_cv = {
                            'split': split_type,
                            'dataset': dataset,
                            'fingerprint': fp_type,
                            'model': model_type,
                            'training': 'cv',
                            'train_time': end_time - start_time,
                            'r2': metrics_cv['r2'],
                            'r2_std': r2_std,
                            'rmse': metrics_cv['rmse'],
                            'rmse_std': rmse_std,
                            'mae': metrics_cv['mae'],
                            'mae_std': mae_std,
                            'mse': metrics_cv['mse'],
                            'pred_std': mean_std,
                        }
                        all_results.append(result_cv)
                        
                        print(f"      CV training - R²: {metrics_cv['r2']:.3f}±{r2_std:.3f}, RMSE: {metrics_cv['rmse']:.3f}±{rmse_std:.3f}")
                        
                    except Exception as e:
                        print(f"      Error in CV training: {e}")
                        result_cv = {
                            'split': split_type,
                            'dataset': dataset,
                            'fingerprint': fp_type,
                            'model': model_type,
                            'training': 'cv',
                            'r2': np.nan,
                            'r2_std': np.nan,
                            'rmse': np.nan,
                            'rmse_std': np.nan,
                            'mae': np.nan,
                            'mae_std': np.nan,
                            'mse': np.nan,
                            'pred_std': np.nan,
                            'error': str(e)
                        }
                        all_results.append(result_cv)
                    
                    # Garbage collection after each model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Save intermediate results every dataset
            if all_results:
                interim_df = pd.DataFrame(all_results)
                interim_df.to_csv(f"{output_dir}/interim_results_{completed_combinations}.csv", index=False)
        
        # Major garbage collection after each split
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{output_dir}/all_results_with_predictions.csv", index=False)
        
        # Save predictions in chunks to manage memory
        import pickle
        with open(f"{output_dir}/all_predictions.pkl", 'wb') as f:
            pickle.dump(all_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return results_df, all_predictions
    else:
        print("No results generated!")
        return pd.DataFrame(), {}

def main():
    """Main execution function"""
    print("Starting Optimized PyTorch-based Standard Comparison Experiment")
    print(f"Device: {device}")
    print(f"Configuration: EPOCHS={EPOCHS}, BATCH_SIZE={BATCHSIZE}, LR={lr}, CV_FOLDS={CV}")
    print(f"Optimizations: Mixed Precision Training, LeakyReLU, AdamW, Cosine Annealing LR")
    print(f"Memory Management: Chunk processing, Temporary files, Aggressive garbage collection")
    
    try:
        # Load data
        print("\nLoading data...")
        data_dict = load_split_data()
        
        # Extract features and targets
        print("Extracting features and targets...")
        x_map, y_map = extract_xy_from_data(data_dict)
        
        # Build fingerprints
        print("Building fingerprints...")
        fp_map = build_fingerprints_for_splits(x_map, out_root)
        
        # Run experiments
        print("\nRunning experiments...")
        results_df, all_predictions = run_full_experiment_with_predictions(fp_map, y_map)
        
        print("\nExperiment completed!")
        print(f"Results saved to: {target_path}")
        
        # Print summary statistics
        if not results_df.empty:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            
            # Best models by dataset
            for training in ['simple', 'cv']:
                print(f"\n{training.upper()} Training Results:")
                print("-" * 40)
                
                training_df = results_df[results_df['training'] == training]
                
                for dataset in ['ws', 'de', 'lo', 'hu']:
                    dataset_df = training_df[training_df['dataset'] == dataset]
                    if not dataset_df.empty and 'r2' in dataset_df.columns:
                        best_row = dataset_df.loc[dataset_df['r2'].idxmax()]
                        print(f"\n{dataset.upper()} Dataset:")
                        print(f"  Best: {best_row['model']} - {best_row['fingerprint']}")
                        print(f"  R² = {best_row['r2']:.3f}")
                        if training == 'cv' and 'r2_std' in best_row:
                            print(f"  R² std = {best_row['r2_std']:.3f}")
    
    finally:
        # Clean up temporary files
        cleanup_temp_files()
        
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()