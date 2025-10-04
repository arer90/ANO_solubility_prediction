#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning Process for PyTorch Models with TorchScript Support
============================================================

Trains models and saves them as TorchScript for subprocess compatibility.
Supports both old (npy files) and new (pkl file) formats.
"""

import sys
import os
import pickle
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import OS-specific optimizations
try:
    from device_utils import (get_optimal_device, get_optimal_batch_size,
                             get_optimal_num_workers, optimize_model_for_device,
                             cleanup_memory, get_training_optimizations)
    USE_DEVICE_UTILS = True
except ImportError:
    USE_DEVICE_UTILS = False
    print("Note: device_utils not found, using basic device selection")

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# SimpleDNN class moved to ano_feature_selection.py for centralized definition
try:
    from ano_feature_selection import SimpleDNN
except ImportError:
    # If direct import fails, try with explicit path
    sys.path.insert(0, os.path.join(os.getcwd(), 'extra_code'))
    from ano_feature_selection import SimpleDNN

# Import config for early stopping patience
try:
    from config import MODEL_CONFIG
    # More aggressive early stopping for high-dimensional data
    EARLY_STOPPING_PATIENCE = MODEL_CONFIG.get('early_stopping_patience', 30)
except ImportError:
    EARLY_STOPPING_PATIENCE = 30  # Reduced default for better generalization

def train_model(X_train, y_train, X_val, y_val, batch_size, epochs, lr, architecture=None, dropout_rate=0.2, model_path=None):
    """Train the model and return it"""
    # Use OS-specific device optimization if available
    if USE_DEVICE_UTILS:
        device, device_info = get_optimal_device()

        # Optimize batch size based on device
        optimal_batch = get_optimal_batch_size(device_info, X_train.shape[1])
        if batch_size != optimal_batch:
            print(f"Suggested batch size: {optimal_batch} (using {batch_size})")

        # Get OS-specific optimizations
        train_opts = get_training_optimizations(device_info)
    else:
        # Fallback to basic device selection
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"Using device: {device}")
        device_info = None
        train_opts = {}

    # Data validation
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Target stats - Train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"Target stats - Val: mean={y_val.mean():.3f}, std={y_val.std():.3f}")

    # Check for potential issues
    if np.std(y_train) < 1e-6:
        print("⚠️ WARNING: Very low target variance - may lead to R² ≈ 0")
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        print("⚠️ WARNING: NaN values detected in training data")
        # Clean NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        print("✅ NaN values cleaned with nan_to_num")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        print("⚠️ WARNING: Inf values detected in training data")
        # Clean Inf values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
        print("✅ Inf values cleaned with nan_to_num")

    # Convert to tensors (keep training data on CPU for DataLoader)
    # This avoids multiprocessing issues with MPS/CUDA
    X_train_tensor_cpu = torch.FloatTensor(X_train)
    y_train_tensor_cpu = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    # Create DataLoader with OS-specific optimizations
    # Use CPU tensors for DataLoader to avoid multiprocessing issues
    train_dataset = TensorDataset(X_train_tensor_cpu, y_train_tensor_cpu)

    # Get optimal number of workers for DataLoader
    if USE_DEVICE_UTILS and device_info:
        num_workers = get_optimal_num_workers(device_info)
        pin_memory = train_opts.get('pin_memory', False)

        # Force num_workers to 0 for MPS to avoid issues
        if device_info.get('device_type') == 'mps':
            num_workers = 0
            pin_memory = False
    else:
        num_workers = 0  # Safe default
        pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Try to load existing model first
    input_dim = X_train.shape[1]
    model_loaded = False

    if model_path and os.path.exists(model_path):
        try:
            print(f"Attempting to load existing model from {model_path}")

            # Try loading as TorchScript first
            try:
                model = torch.jit.load(model_path, map_location=device)
                # Check if input dimension matches using actual data
                test_input = torch.FloatTensor(X_train[:1]).to(device)
                try:
                    _ = model(test_input)
                    print(f"Successfully loaded TorchScript model with matching input_dim={input_dim}")
                    model_loaded = True
                except Exception as dim_err:
                    print(f"Loaded model has incompatible input dimension. Creating new model.")
                    model_loaded = False
            except Exception as jit_err:
                # Try loading as regular PyTorch model
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    saved_input_dim = checkpoint.get('input_dim', None)

                    if saved_input_dim == input_dim:
                        # Input dimensions match, can load the model
                        saved_architecture = checkpoint.get('architecture', None)
                        saved_dropout = checkpoint.get('dropout_rate', 0.5)

                        if saved_architecture:
                            if isinstance(saved_architecture, dict):
                                hidden_dims = saved_architecture.get('hidden_dims', [256, 128])[:-1]
                            else:
                                hidden_dims = saved_architecture[1:-1]
                        else:
                            # Use default architecture if not specified
                            hidden_dims = [1024, 496] if input_dim > 1000 else [256, 128]

                        model = SimpleDNN(input_dim, hidden_dims, saved_dropout).to(device)
                        model.load_state_dict(checkpoint['state_dict'])
                        print(f"Successfully loaded PyTorch model with matching input_dim={input_dim}")
                        model_loaded = True
                    else:
                        print(f"Saved model has input_dim={saved_input_dim}, current data has input_dim={input_dim}. Creating new model.")
                        model_loaded = False
                except Exception as pt_err:
                    print(f"Failed to load model: {pt_err}. Creating new model.")
                    model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model.")
            model_loaded = False

    # Create new model if loading failed or no model path provided
    if not model_loaded:
        print(f"Creating new model with input_dim={input_dim}")
        if architecture:
            # Handle both dict and list formats
            if isinstance(architecture, dict):
                # New format from Module 8: {'n_layers': 3, 'hidden_dims': [1024, 496, 256, 1]}
                hidden_dims = architecture.get('hidden_dims', [256, 128])[:-1]  # Remove output dim
            else:
                # Old format: list of dimensions
                hidden_dims = architecture[1:-1]  # Remove input and output dims
            model = SimpleDNN(input_dim, hidden_dims, dropout_rate).to(device)
        else:
            # Fixed architecture regardless of input dimension
            # This prevents overfitting with high-dimensional inputs
            hidden_dims = [1024, 496]
            # Use moderate dropout for high-dimensional inputs
            actual_dropout = 0.3 if input_dim > 2000 else dropout_rate
            model = SimpleDNN(input_dim, hidden_dims, dropout_rate=actual_dropout).to(device)
            dropout_rate = actual_dropout  # Update for consistency

    # Apply OS-specific model optimizations
    if USE_DEVICE_UTILS and device_info:
        model = optimize_model_for_device(model, device, device_info)

    print(f"Model architecture: input_dim={input_dim}, hidden_dims={hidden_dims}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Model complexity check
    param_count = sum(p.numel() for p in model.parameters())
    data_size = len(X_train)
    if param_count > data_size * 0.1:
        # Model complexity warning - commented out for cleaner result display
        # print(f"⚠️ WARNING: Model may be too complex ({param_count:,} params for {data_size:,} samples)")
        # print("Consider reducing architecture size if performance is poor")
        pass

    # Improved weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # Loss and optimizer with moderate regularization
    criterion = nn.MSELoss()
    # SimpleDNN configuration: use consistent weight_decay with l2_reg
    weight_decay = 1e-5  # Match SimpleDNN l2_reg setting
    # Use original learning rate with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Regularization: dropout={dropout_rate}, weight_decay={weight_decay}, l2_reg={weight_decay}")
    # Learning rate scheduler with patience for stable convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, min_lr=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Move batch to device (important for MPS/CUDA)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Moderate gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        with torch.inference_mode():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Calculate metrics
            y_val_pred = val_outputs.cpu().numpy().flatten()
            # Clean predictions for NaN/Inf values
            y_val_pred = np.nan_to_num(y_val_pred, nan=0.0, posinf=0.0, neginf=0.0)
            r2 = r2_score(y_val, y_val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            mae = mean_absolute_error(y_val, y_val_pred)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            # Epoch progress output - commented out for cleaner result display
            # print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            #       f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            # Periodic memory cleanup during training
            if USE_DEVICE_UTILS and device_info:
                cleanup_memory(device_info)
            else:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.inference_mode():
        val_outputs = model(X_val_tensor)
        y_val_pred = val_outputs.cpu().numpy().flatten()
        # Clean final predictions for NaN/Inf values
        y_val_pred = np.nan_to_num(y_val_pred, nan=0.0, posinf=0.0, neginf=0.0)
        final_r2 = r2_score(y_val, y_val_pred)
        final_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        final_mae = mean_absolute_error(y_val, y_val_pred)
        
    print(f"\nFinal Results - R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    
    return model

def save_as_torchscript(model, model_file, input_dim, architecture=None, dropout_rate=0.2,
                        batch_size=32, epochs=100, lr=0.001):
    """Save model with fallback chain: torch.save → TorchScript → in-script model → error"""
    model.eval()
    model.cpu()  # Move to CPU for compatibility
    
    # Set BatchNorm to eval mode properly
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    
    # Method 1: Try regular torch.save first (most reliable) - prioritize .pth
    try:
        # If user requests .pt, still save as .pth first
        if model_file.endswith('.pt'):
            model_file_pth = model_file.replace('.pt', '.pth')
        else:
            model_file_pth = model_file if model_file.endswith('.pth') else model_file + '.pth'
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'model': model,  # Save entire model for compatibility
            'architecture': str(model),  # Model architecture as string
            'input_dim': input_dim,
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs,
                'optimizer': 'Adam',
                'loss_function': 'MSE'
            }
        }, model_file_pth)
        print(f"[Method 1] Model saved with torch.save to {model_file_pth}")
        return True
    except Exception as e1:
        print(f"[Method 1 Failed] torch.save error: {e1}")
        
    # Method 2: Try TorchScript tracing
    try:
        custom_input = torch.randn(batch_size, input_dim)  # Use actual batch_size for proper BatchNorm behavior
        traced_model = torch.jit.trace(model, custom_input, check_trace=False)
        traced_model.save(model_file)
        print(f"[Method 2] Model saved as TorchScript to {model_file}")
        return True
    except Exception as e2:
        print(f"[Method 2 Failed] TorchScript trace error: {e2}")
    
    # Method 3: Save model architecture and weights separately for manual reconstruction
    try:
        # Save architecture info and weights for manual reconstruction
        reconstruction_file = model_file.replace('.pt', '_reconstruction.pth')

        # Generate layer structure dynamically from actual model
        layers = []
        if hasattr(model, 'layers'):
            # Extract structure from SimpleDNN
            layers.append({'type': 'Linear', 'in': input_dim, 'out': model.layers[0].out_features})
            layers.append({'type': 'ReLU'})
            if len(model.layers) > 2:  # Has BatchNorm and Dropout
                layers.append({'type': 'BatchNorm1d', 'features': model.layers[0].out_features, 'track_running_stats': False})
                layers.append({'type': 'Dropout', 'p': dropout_rate})

            # Add hidden layers
            for i in range(1, len(model.layers) - 1):
                if isinstance(model.layers[i], nn.Linear):
                    layers.append({'type': 'Linear', 'in': model.layers[i].in_features, 'out': model.layers[i].out_features})
                    layers.append({'type': 'ReLU'})
                    if i < len(model.layers) - 2:  # Not last layer
                        layers.append({'type': 'BatchNorm1d', 'features': model.layers[i].out_features, 'track_running_stats': False})
                        layers.append({'type': 'Dropout', 'p': dropout_rate})

            # Final layer
            if len(model.layers) > 0:
                final_layer = model.layers[-1]
                if isinstance(final_layer, nn.Linear):
                    layers.append({'type': 'Linear', 'in': final_layer.in_features, 'out': final_layer.out_features})

        torch.save({
            'state_dict': model.state_dict(),
            'architecture': architecture,
            'dropout_rate': dropout_rate,
            'model_class': 'SimpleDNN',
            'input_dim': input_dim,
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs
            },
            'layers': layers
        }, reconstruction_file)
        print(f"[Method 3] Model architecture and weights saved for reconstruction to {reconstruction_file}")
        return True
    except Exception as e3:
        print(f"[Method 3 Failed] Architecture save error: {e3}")
    
    # Method 4: All methods failed - raise error
    error_msg = f"ERROR: All save methods failed!\n" \
                f"  Method 1 (torch.save): {e1}\n" \
                f"  Method 2 (TorchScript): {e2}\n" \
                f"  Method 3 (Architecture): {e3 if 'e3' in locals() else 'Not attempted'}"
    print(error_msg)
    raise RuntimeError(error_msg)

def main():
    if len(sys.argv) == 3:
        # New format: train_data.pkl model.pt
        train_data_file = sys.argv[1]
        model_file = sys.argv[2]
        
        # Load training data
        with open(train_data_file, 'rb') as f:
            train_data = pickle.load(f)
        
        # Handle both old and new data format
        X_train = train_data.get('X_train', train_data.get('X'))
        y_train = train_data.get('y_train', train_data.get('y'))
        
        # Use validation data if provided, otherwise use test data as validation
        if 'X_val' in train_data and 'y_val' in train_data:
            X_val = train_data['X_val']
            y_val = train_data['y_val']
        elif 'X_test' in train_data and 'y_test' in train_data:
            # Module 8 passes test data for evaluation
            X_val = train_data['X_test']
            y_val = train_data['y_test']
        else:
            # Simple split for validation
            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]

        # For test predictions, use test data if available, otherwise validation data
        if 'X_test' in train_data and 'y_test' in train_data:
            X_test = train_data['X_test']
            y_test = train_data['y_test']
        else:
            # Fallback: use validation data as test data
            X_test = X_val
            y_test = y_val
        
        batch_size = train_data.get('batch_size', 32)
        epochs = train_data.get('epochs', 100)
        lr = train_data.get('lr', 0.001)
        architecture = train_data.get('architecture', None)
        dropout_rate = train_data.get('dropout_rate', 0.2)
        
    elif len(sys.argv) == 9:
        # Old format: Module 4~7 compatibility
        batch_size = int(sys.argv[1])
        epochs = int(sys.argv[2])
        lr = float(sys.argv[3])

        X_train = np.load(sys.argv[4])
        y_train = np.load(sys.argv[5])
        X_val = np.load(sys.argv[6])
        y_val = np.load(sys.argv[7])

        # For old format, use validation data as test data
        X_test = X_val
        y_test = y_val

        model_file = sys.argv[8]

        # Try to load architecture from temp file if exists
        arch_file = "save_model/temp_architecture.json"
        if os.path.exists(arch_file):
            with open(arch_file, 'r') as f:
                arch_data = json.load(f)
                architecture = arch_data.get('hidden_dims', None)
                dropout_rate = arch_data.get('dropout_rate', 0.2)
        else:
            architecture = None
            dropout_rate = 0.2

    else:
        print("Usage:")
        print("  New format: python learning_process_pytorch_torchscript.py train_data.pkl model.pt")
        print("  Old format: python learning_process_pytorch_torchscript.py <batch_size> <epochs> <lr> X_train.npy y_train.npy X_val.npy y_val.npy model.pt")
        sys.exit(1)
    
    # Train model
    print(f"Training with {len(X_train)} samples, batch_size={batch_size}, epochs={epochs}, lr={lr}")
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Target stats - Train: mean={np.mean(y_train):.3f}, std={np.std(y_train):.3f}")
    print(f"Target stats - Val: mean={np.mean(y_val):.3f}, std={np.std(y_val):.3f}")

    try:
        model = train_model(X_train, y_train, X_val, y_val, batch_size, epochs, lr, architecture, dropout_rate, model_file)
        print("Model training completed successfully")
    except Exception as e:
        print(f"ERROR in train_model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save model with fallback chain
    input_dim = X_train.shape[1]
    save_as_torchscript(model, model_file, input_dim, architecture=architecture, dropout_rate=dropout_rate,
                        batch_size=batch_size, epochs=epochs, lr=lr)
    
    # Final evaluation and print metrics for parsing
    model.eval()
    # Support GPU on all platforms
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    with torch.inference_mode():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        model = model.to(device)

        val_outputs = model(X_val_tensor)
        y_val_pred = val_outputs.cpu().numpy().flatten()
        # Clean final predictions for NaN/Inf values
        y_val_pred = np.nan_to_num(y_val_pred, nan=0.0, posinf=0.0, neginf=0.0)
        final_r2 = r2_score(y_val, y_val_pred)
        final_mse = mean_squared_error(y_val, y_val_pred)
        final_rmse = np.sqrt(final_mse)
        final_mae = mean_absolute_error(y_val, y_val_pred)

        # Test predictions
        test_outputs = model(X_test_tensor)
        y_test_pred = test_outputs.cpu().numpy().flatten()

        # Save predictions to file for main process to load
        if len(sys.argv) > 8:
            pred_file = model_file.replace('.pt', '_pred.npy').replace('.pth', '_pred.npy')
            np.save(pred_file, y_test_pred)
            print(f"Predictions saved to {pred_file}")

    # Print in comma-separated format for parsing: r2,rmse,mse,mae
    print(f"{final_r2},{final_rmse},{final_mse},{final_mae}")

    # Print test predictions in format: PREDICTIONS:pred1,pred2,pred3...
    # pred_str = ','.join([f"{p:.6f}" for p in y_test_pred])
    # PREDICTIONS output - commented out for cleaner result display
    # print(f"PREDICTIONS:{pred_str}")
    
    # Memory cleanup before exit
    del model, X_train, y_train, X_val, y_val, X_test, y_test
    if 'X_train_tensor' in locals():
        del X_train_tensor, y_train_tensor, X_val_tensor, X_test_tensor
    # Final memory cleanup
    if USE_DEVICE_UTILS:
        # Get device info for cleanup
        _, cleanup_info = get_optimal_device()
        cleanup_memory(cleanup_info)
    else:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Ensure output is flushed
    sys.stdout.flush()

if __name__ == "__main__":
    main()