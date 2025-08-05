#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

ANO Test-Only Dataset Prediction and Analysis
=============================================

PURPOSE:
This module loads the best models from ANO modules 5-8 and makes predictions
on test-only datasets (FreeSolv, Lipophilicity, AqSolDB, BigSolDB).
It generates comprehensive visualizations and saves all results.

APPROACH:
1. Load best models from each ANO optimization strategy
2. Load test-only datasets from data/test/to/
3. Generate predictions for each model-dataset combination
4. Create visualizations comparing predictions vs actual values
5. Calculate and save performance metrics
6. Generate summary reports and plots

KEY ANALYSES:
- Scatter plots of predicted vs actual values
- Performance metrics (R², RMSE, MAE) for each dataset
- Model comparison across different test datasets
- Error distribution analysis
- Summary tables and reports
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import warnings
warnings.filterwarnings('ignore')

# Add extra_code to path
sys.path.append('extra_code')

# Import required functions
from molecular_loader_with_fp import get_fingerprints
from ano_feature_search import prefilter_3d_conformers
from ano_feature_selection import (
    selection_data_descriptor_compress,
    convert_params_to_selection
)

from rdkit import Chem
from rdkit.Chem import AllChem

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
OUTPUT_DIR = Path("result/10_ANO_testonly_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_NAME = "sqlite:///ano_final.db"

# Dataset mapping
TEST_ONLY_DATASETS = {
    'FreeSolv': 'FreeSolv_test.csv',
    'Lipophilicity': 'Lipophilicity_test.csv', 
    'AqSolDB': 'curated-solubility-dataset_test.csv',
    'BigSolDB': 'BigSolDB_test.csv'
}

# Module names for loading results
MODULE_NAMES = {
    '5_feature': 'ano_feature',
    '6_structure': 'ano_structure',
    '7_feature_structure': 'ano_network_FOMO',
    '8_structure_feature': 'ano_network_MOFO'
}

class FlexibleDNN(nn.Module):
    """Flexible DNN that can handle different architectures"""
    def __init__(self, input_dim, hidden_dims=None, dropout_rate=0.2):
        super(FlexibleDNN, self).__init__()
        
        if hidden_dims is None:
            # Default architecture from module 5
            hidden_dims = [1024, 496]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.BatchNorm1d(hidden_dim, momentum=0.01),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
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

def load_test_dataset(dataset_name, file_path, check_3d_conformers=True):
    """
    Load and process test dataset with optional 3D conformer filtering.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    file_path : Path
        Path to the CSV file
    check_3d_conformers : bool
        If True, pre-filter molecules that cannot generate 3D conformers
    
    Returns:
    --------
    tuple : (fps_all, y_valid, mols) or None if loading fails
    """
    print(f"\nLoading {dataset_name} from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        if 'SMILES' in df.columns:
            df['smiles'] = df['SMILES']
        elif 'Smiles' in df.columns:
            df['smiles'] = df['Smiles']
        
        # Find target column
        target_col = None
        for col in ['target', 'Target', 'Solubility', 'solubility', 'expt', 'y']:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print(f"Warning: No target column found in {dataset_name}")
            return None, None, None
        
        smiles_list = df['smiles'].tolist()
        y_list = df[target_col].tolist()
        
        # Pre-filter molecules for 3D conformer generation if needed
        if check_3d_conformers:
            print(f"  Pre-filtering molecules for 3D conformer generation...")
            smiles_list, y_list, mols, _ = prefilter_3d_conformers(smiles_list, y_list)
            
            if not mols:
                print(f"Error: No molecules could generate 3D conformers in {dataset_name}")
                return None, None, None
        else:
            # Standard validation without 3D check
            mols = []
            valid_indices = []
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mols.append(mol)
                    valid_indices.append(i)
            
            if not mols:
                print(f"Error: No valid molecules in {dataset_name}")
                return None, None, None
            
            # Filter y values
            y_list = [y_list[i] for i in valid_indices]
        
        # Generate fingerprints
        morgan_fps, maccs_fps, avalon_fps = get_fingerprints(mols)
        fps_all = np.hstack([morgan_fps, maccs_fps, avalon_fps])
        
        print(f"  Loaded {len(mols)} valid molecules")
        print(f"  Fingerprint shape: {fps_all.shape}")
        
        return fps_all, y_list, mols
        
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None, None

def get_best_model_params(module_name, dataset='ws', split='rm'):
    """Get best model parameters from Optuna study"""
    study_name = f"{MODULE_NAMES[module_name]}_{dataset}_{split}"
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=STORAGE_NAME
        )
        
        best_trial = study.best_trial
        
        # Extract model architecture
        model_params = {
            'best_value': best_trial.value,
            'n_features': best_trial.user_attrs.get('n_features', 2727),
            'selected_descriptors': best_trial.user_attrs.get('selected_descriptors', []),
            'params': best_trial.params
        }
        
        # Extract architecture for modules 6, 7, 8
        if module_name in ['6_structure', '7_feature_structure', '8_structure_feature']:
            model_params['n_layers'] = best_trial.user_attrs.get('n_layers', 2)
            model_params['hidden_dims'] = best_trial.user_attrs.get('hidden_dims', [512, 256])
            model_params['dropout_rate'] = best_trial.user_attrs.get('dropout_rate', 0.2)
        
        return model_params
        
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None

def predict_with_model(fps, mols, model_params, module_name):
    """Make predictions using specified model architecture"""
    
    # For feature selection modules (5, 7, 8), apply feature selection
    if module_name in ['5_feature', '7_feature_structure', '8_structure_feature']:
        if 'params' in model_params and model_params['params']:
            # Convert params to selection format
            selection = convert_params_to_selection(model_params['params'])
            
            # Apply feature selection
            fps_selected, _ = selection_data_descriptor_compress(
                selection, fps, mols, 'test',
                target_path=str(OUTPUT_DIR),
                save_res="np"
            )
            fps = fps_selected
    
    # Create model with appropriate architecture
    input_dim = fps.shape[1]
    
    if module_name in ['6_structure', '7_feature_structure', '8_structure_feature']:
        hidden_dims = model_params.get('hidden_dims', [512, 256])
        dropout_rate = model_params.get('dropout_rate', 0.2)
        model = FlexibleDNN(input_dim, hidden_dims, dropout_rate)
    else:
        # Default architecture for module 5
        model = FlexibleDNN(input_dim)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(fps)
        predictions = model(X_tensor).numpy().flatten()
    
    return predictions

def requires_3d_descriptors(model_params, module_name):
    """
    Check if the model uses any 3D descriptors.
    
    3D descriptor indices in the selection array:
    - Index 37: PMI - Principal Moments of Inertia
    - Index 38: NPR - Normalized Principal Moments Ratios  
    - Index 39: 3D-MoRSE - 3D Molecular Representation of Structures
    - Index 40: WHIM - Weighted Holistic Invariant Molecular
    - Index 41: GETAWAY - Geometry, Topology, and Atom-Weights Assembly
    - Index 42: Autocorr3D - 3D Autocorrelation
    - Index 43: RDF - Radial Distribution Function
    - Index 44: BCUT2D - Burden-CAS-University of Texas eigenvalues
    - Index 45: SpherocityIndex - Molecular spherocity
    - Index 46: PBF - Plane of Best Fit
    - Index 47: PEOE_VSA Series - Partial Equalization of Orbital Electronegativity
    - Index 48: SMR_VSA Series - Molecular Refractivity
    - Index 49: SlogP_VSA Series - Log of partition coefficient
    - Index 50: EState_VSA Series - Electrotopological state
    """
    if module_name in ['5_feature', '7_feature_structure', '8_structure_feature']:
        if 'params' in model_params and model_params['params']:
            # Convert params to selection format
            selection = convert_params_to_selection(model_params['params'])
            
            # Check if any 3D descriptors are selected (indices 37-50)
            for i in range(37, 51):
                if i < len(selection) and selection[i] == 1:
                    print(f"  Model uses 3D descriptor at index {i}")
                    return True
    
    return False

def create_prediction_plots(results_df, output_dir):
    """Create comprehensive prediction visualizations"""
    
    # 1. Individual scatter plots for each model-dataset combination
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle('Test-Only Dataset Predictions: All Models', fontsize=16, fontweight='bold')
    
    modules = ['5_feature', '6_structure', '7_feature_structure', '8_structure_feature']
    datasets = list(TEST_ONLY_DATASETS.keys())
    
    for i, module in enumerate(modules):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            
            # Filter data
            mask = (results_df['module'] == module) & (results_df['dataset'] == dataset)
            data = results_df[mask]
            
            if len(data) > 0:
                # Scatter plot
                ax.scatter(data['actual'], data['predicted'], alpha=0.6, s=30)
                
                # Add y=x line
                min_val = min(data['actual'].min(), data['predicted'].min())
                max_val = max(data['actual'].max(), data['predicted'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                # Calculate metrics
                r2 = r2_score(data['actual'], data['predicted'])
                rmse = np.sqrt(mean_squared_error(data['actual'], data['predicted']))
                mae = mean_absolute_error(data['actual'], data['predicted'])
                
                # Add text
                ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{module.split("_")[1].title()} - {dataset}')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{module.split("_")[1].title()} - {dataset}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance on Test-Only Datasets', fontsize=14, fontweight='bold')
    
    # Calculate average metrics per module
    metrics_summary = results_df.groupby(['module', 'dataset']).agg({
        'r2': 'first',
        'rmse': 'first',
        'mae': 'first'
    }).reset_index()
    
    # R² comparison
    pivot_r2 = metrics_summary.pivot(index='dataset', columns='module', values='r2')
    pivot_r2.plot(kind='bar', ax=ax1)
    ax1.set_title('R² Score Comparison')
    ax1.set_ylabel('R² Score')
    ax1.set_xlabel('Dataset')
    ax1.legend(title='Module', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    pivot_rmse = metrics_summary.pivot(index='dataset', columns='module', values='rmse')
    pivot_rmse.plot(kind='bar', ax=ax2)
    ax2.set_title('RMSE Comparison')
    ax2.set_ylabel('RMSE')
    ax2.set_xlabel('Dataset')
    ax2.legend(title='Module', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Prediction Error Distribution by Module', fontsize=14, fontweight='bold')
    
    for i, module in enumerate(modules):
        ax = axes[i//2, i%2]
        
        module_data = results_df[results_df['module'] == module]
        
        for dataset in datasets:
            data = module_data[module_data['dataset'] == dataset]
            if len(data) > 0:
                errors = data['predicted'] - data['actual']
                ax.hist(errors, bins=30, alpha=0.6, label=dataset, density=True)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title(f'{module.split("_")[1].title()} Module')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_tables(results_df, summary_df, output_dir):
    """Save results as CSV and formatted tables"""
    
    # Save raw results
    results_df.to_csv(output_dir / 'all_predictions.csv', index=False)
    
    # Save summary metrics
    summary_df.to_csv(output_dir / 'performance_summary.csv', index=False)
    
    # Create formatted performance table
    performance_table = summary_df.pivot_table(
        index='dataset', 
        columns='module', 
        values=['r2', 'rmse', 'mae']
    ).round(3)
    
    performance_table.to_csv(output_dir / 'performance_table.csv')
    
    # Create LaTeX table for publication
    latex_table = performance_table.to_latex(
        caption="Performance of ANO models on test-only datasets",
        label="tab:testonly_performance"
    )
    
    with open(output_dir / 'performance_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nPerformance Summary:")
    print(performance_table)

def main():
    """Main function to run test-only predictions"""
    print("="*80)
    print("ANO Test-Only Dataset Prediction and Analysis")
    print("="*80)
    
    # Load test datasets
    test_data_dir = Path("data/test/to")
    all_results = []
    summary_results = []
    
    # First pass: Check which models require 3D descriptors
    print("\nChecking which models use 3D descriptors...")
    models_requiring_3d = {}
    for module_name in MODULE_NAMES.keys():
        model_params = get_best_model_params(module_name, 'ws', 'rm')
        if model_params is not None:
            requires_3d = requires_3d_descriptors(model_params, module_name)
            models_requiring_3d[module_name] = requires_3d
            if requires_3d:
                print(f"  {module_name}: Requires 3D conformer filtering")
    
    for dataset_name, file_name in TEST_ONLY_DATASETS.items():
        file_path = test_data_dir / file_name
        
        if not file_path.exists():
            print(f"\nWarning: {file_path} not found, skipping {dataset_name}")
            continue
        
        # Load dataset - check if any model needs 3D conformers
        needs_3d_check = any(models_requiring_3d.values())
        fps, y_true, mols = load_test_dataset(dataset_name, file_path, check_3d_conformers=needs_3d_check)
        
        if fps is None:
            continue
        
        # Test each model
        for module_name in MODULE_NAMES.keys():
            print(f"\nTesting {module_name} on {dataset_name}...")
            
            # Get best model parameters (using ws/rm as reference)
            model_params = get_best_model_params(module_name, 'ws', 'rm')
            
            if model_params is None:
                print(f"  Skipping {module_name} - no model found")
                continue
            
            # Skip if model requires 3D but we don't have molecules with conformers
            if models_requiring_3d.get(module_name, False) and not needs_3d_check:
                print(f"  Skipping {module_name} - requires 3D descriptors but conformers not available")
                continue
            
            try:
                # Make predictions
                predictions = predict_with_model(fps, mols, model_params, module_name)
                
                # Calculate metrics
                r2 = r2_score(y_true, predictions)
                rmse = np.sqrt(mean_squared_error(y_true, predictions))
                mae = mean_absolute_error(y_true, predictions)
                
                print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
                
                # Store results
                for i in range(len(y_true)):
                    all_results.append({
                        'module': module_name,
                        'dataset': dataset_name,
                        'actual': y_true[i],
                        'predicted': predictions[i],
                        'error': predictions[i] - y_true[i],
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae
                    })
                
                summary_results.append({
                    'module': module_name,
                    'dataset': dataset_name,
                    'n_samples': len(y_true),
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'cv_r2': model_params.get('best_value', 0.0)
                })
                
            except Exception as e:
                print(f"  Error making predictions: {e}")
                continue
    
    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_prediction_plots(results_df, OUTPUT_DIR)
    
    # Save results
    print("\nSaving results...")
    save_results_tables(results_df, summary_df, OUTPUT_DIR)
    
    # Add dataset source information
    source_info = {
        'FreeSolv': 'FreeSolv Database - Experimental and calculated hydration free energy (https://github.com/MobleyLab/FreeSolv)',
        'Lipophilicity': 'ChEMBL Lipophilicity Dataset - Octanol/water partition coefficient',
        'AqSolDB': 'Aqueous Solubility Database - Curated aqueous solubility data (Sorkun et al., 2019)',
        'BigSolDB': 'Big Solubility Database - Large-scale solubility dataset'
    }
    
    with open(OUTPUT_DIR / 'dataset_sources.txt', 'w') as f:
        f.write("Test-Only Dataset Sources:\n")
        f.write("="*50 + "\n\n")
        for dataset, source in source_info.items():
            f.write(f"{dataset}:\n{source}\n\n")
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - all_predictions_scatter.png: Scatter plots for all model-dataset combinations")
    print("  - performance_comparison.png: Bar plots comparing model performance")
    print("  - error_distribution.png: Error distribution histograms")
    print("  - all_predictions.csv: Raw prediction results")
    print("  - performance_summary.csv: Summary metrics")
    print("  - performance_table.csv: Formatted performance table")
    print("  - performance_table.tex: LaTeX table for publication")
    print("  - dataset_sources.txt: Dataset source information")

if __name__ == "__main__":
    main()