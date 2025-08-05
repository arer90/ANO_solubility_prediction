#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer: Lee, Seungjin (arer90)

Learning Process for PyTorch Models
===================================

PURPOSE:
This module provides a subprocess-based training pipeline for PyTorch models.
It's designed to be called from the main optimization scripts to train neural
networks in isolated processes, ensuring clean memory management and preventing
memory leaks during hyperparameter optimization.

KEY FEATURES:
- Subprocess-based training for memory isolation
- Simple DNN architecture optimized for molecular property prediction
- Efficient batch processing with DataLoader
- Comprehensive metric calculation (R², RMSE, MSE, MAE)
- Model state persistence for reuse

ARCHITECTURE:
- 3-layer feedforward network
- SiLU activation (smooth version of ReLU)
- Batch normalization for stable training
- Dropout for regularization
- Xavier weight initialization

USAGE:
This script is called as a subprocess with command-line arguments:
python learning_process_pytorch.py <batch_size> <epochs> <learning_rate> \
       <X_train.npy> <y_train.npy> <X_test.npy> <y_test.npy> <model.pth>
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class SimpleDNN(nn.Module):
    """
    Simple Deep Neural Network for molecular property regression.
    
    This architecture is specifically designed for molecular solubility prediction
    with the following characteristics:
    
    Architecture:
    - Input layer: Variable size (depends on fingerprints/descriptors)
    - Hidden layer 1: 1024 units with SiLU activation
    - Hidden layer 2: 496 units with SiLU activation
    - Output layer: 1 unit (regression target)
    
    Regularization:
    - Batch normalization after each hidden layer (momentum=0.01)
    - Dropout (p=0.2) to prevent overfitting
    - Xavier uniform weight initialization
    
    The SiLU (Sigmoid Linear Unit) activation provides smooth gradients
    and often performs better than ReLU for molecular property prediction.
    """
    
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.Dropout(0.2),
            nn.Linear(1024, 496),
            nn.SiLU(),
            nn.BatchNorm1d(496, momentum=0.01),
            nn.Dropout(0.2),
            nn.Linear(496, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

def load_model(model_file, input_dim):
    """
    Load a pre-trained model from file or create a new one.
    
    This function attempts to load a saved model state from disk. If loading
    fails (e.g., architecture mismatch), it creates a new initialized model.
    Non-strict loading is used to handle minor architecture differences.
    
    Parameters:
    -----------
    model_file : str
        Path to the saved model file (.pth)
    input_dim : int
        Number of input features
    
    Returns:
    --------
    model : SimpleDNN
        Loaded or newly initialized model
    """
    try:
        # Load state dict
        state_dict = torch.load(model_file, map_location='cpu')
        
        # Create model
        model = SimpleDNN(input_dim)
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
            print("DEBUG - Model state dict loaded successfully (non-strict)")
        except Exception as e:
            print(f"DEBUG - Error loading state dict: {e}")
            print("DEBUG - Using initialized model")
        
        return model
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using initialized model")
        return SimpleDNN(input_dim)

def train_model(X_train, y_train, model, batch_size, epochs, learning_rate):
    """
    Train the model and return comprehensive evaluation metrics.
    
    This function implements the complete training pipeline:
    1. Convert numpy arrays to PyTorch tensors
    2. Create DataLoader for efficient batch processing
    3. Train using Adam optimizer with MSE loss
    4. Evaluate on training data
    5. Calculate multiple regression metrics
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features (n_samples x n_features)
    y_train : numpy.ndarray
        Training targets (n_samples,)
    model : SimpleDNN
        Model to train
    batch_size : int
        Number of samples per batch
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    
    Returns:
    --------
    metrics : tuple
        (r2_score, rmse, mse, mae) calculated on training data
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate on training data
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_predictions = train_outputs.numpy().flatten()
        train_targets = y_train_tensor.numpy().flatten()
        
        # Calculate all metrics
        r2 = r2_score(train_targets, train_predictions)
        mse = mean_squared_error(train_targets, train_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(train_targets, train_predictions)
        
        print(f"DEBUG - Training completed: R²={r2:.6f}, RMSE={rmse:.6f}, MSE={mse:.6f}, MAE={mae:.6f}")
        print(f"DEBUG - Predictions: min={train_predictions.min():.4f}, max={train_predictions.max():.4f}, mean={train_predictions.mean():.4f}")
        print(f"DEBUG - Targets: min={train_targets.min():.4f}, max={train_targets.max():.4f}, mean={train_targets.mean():.4f}")
        
        return r2, rmse, mse, mae

def main():
    if len(sys.argv) != 9:
        print("Usage: python learning_process_pytorch.py <batch_size> <epochs> <learning_rate> <xtr_file> <ytr_file> <xval_file> <yval_file> <model_file>")
        sys.exit(1)
    
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    xtr_file = sys.argv[4]
    ytr_file = sys.argv[5]
    xval_file = sys.argv[6]
    yval_file = sys.argv[7]
    model_file = sys.argv[8]
    
    print(f"DEBUG - Input parameters:")
    print(f"DEBUG - batch_size: {batch_size}")
    print(f"DEBUG - epochs: {epochs}")
    print(f"DEBUG - learning_rate: {learning_rate}")
    print(f"DEBUG - xtr_file: {xtr_file}")
    print(f"DEBUG - ytr_file: {ytr_file}")
    print(f"DEBUG - xval_file: {xval_file}")
    print(f"DEBUG - yval_file: {yval_file}")
    print(f"DEBUG - model_file: {model_file}")
        
    # Load data
    X_train = np.load(xtr_file)
    y_train = np.load(ytr_file)
    X_val = np.load(xval_file)
    y_val = np.load(yval_file)
    
    print(f"DEBUG - Data shapes:")
    print(f"DEBUG - X_train: {X_train.shape}")
    print(f"DEBUG - y_train: {y_train.shape}")
    print(f"DEBUG - X_val: {X_val.shape}")
    print(f"DEBUG - y_val: {y_val.shape}")
    print(f"DEBUG - X_train stats: min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
    print(f"DEBUG - y_train stats: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")
    
    # Get input dimension from data
    input_dim = X_train.shape[1]
    print(f"DEBUG - Determined input_dim: {input_dim}")
    
    # Load model
    model = load_model(model_file, input_dim)
    print(f"DEBUG - Model loaded with input_dim: {input_dim}")
    
    # Train model
    train_r2, train_rmse, train_mse, train_mae = train_model(X_train, y_train, model, batch_size, epochs, learning_rate)
    
    # Print all metrics in comma-separated format for easy parsing
    print(f"{train_r2},{train_rmse},{train_mse},{train_mae}")
        
if __name__ == "__main__":
    main()