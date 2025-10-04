#!/usr/bin/env python3
"""
Standard ML Comparison with PyTorch Optimization - ANO Framework Module 2
=========================================================================

PURPOSE:
This module provides baseline model comparisons using standard ML algorithms.
It serves as a performance benchmark for the ANO optimization modules (4-7).

KEY FEATURES:
1. **Six ML Models**: Ridge, SVR, RandomForest, XGBoost, LightGBM, DNN
2. **Dual CV Methods**: Type1 (Research) and Type2 (Production) validation
3. **Multi-level Visualization**: Dataset → Split → Fingerprint hierarchy
4. **Automated Plot Generation**: Real-time plots after each fingerprint
5. **Memory Isolation**: Subprocess training for DNN to prevent memory leaks
6. **No StandardScaler**: Raw features used directly

RECENT UPDATES (2024):
- Fixed epochs handling: Now properly uses get_epochs_for_module('2')
- Fixed plot data: Corrected prediction key mismatches (Ridge_cv vs Ridge_cv_method1)
- Added 3-level folder structure: dataset/split_type/fp_type/
- Added split-level comparison plots (heatmaps, boxplots, bar charts)
- Default epochs: 200 (from config.py module_epochs['2'])

SUPPORTED MODELS:
- Ridge: Linear model with L2 regularization (baseline)
- SVR: Support Vector Regression with RBF kernel
- RandomForest: Ensemble of 100 decision trees
- XGBoost: Gradient boosting with early stopping
- LightGBM: Light gradient boosting machine
- DNN: 2-layer neural network [1024, 496] with PyTorch

OUTPUT STRUCTURE:
result/2_standard_comp/
├── {dataset}/                          # Dataset level (ws, de, lo, hu)
│   └── split_{split}/                  # Split level (rm, ac, cl, etc.)
│       └── fp_{fingerprint}/           # Fingerprint level
│           ├── results.csv             # Model performance metrics
│           ├── {model}_scatter.png     # Individual model plots
│           └── all_models_comparison.png # Combined comparison
├── split_{split}_comparison/           # Split-level comparisons
│   ├── {split}_all_results.csv        # All results for this split
│   ├── {split}_r2_heatmap.png         # R² heatmap (dataset × fingerprint)
│   ├── {split}_model_boxplot.png      # Model performance distribution
│   └── {split}_fingerprint_performance.png # Fingerprint averages
└── all_results_with_predictions.csv    # Complete results

USAGE:
python 2_standard_comp_pytorch_optimized.py [options]
  --dataset: Specific dataset (ws/de/lo/hu)
  --split: Specific split type (rm/ac/cl/cs/en/pc/sa/sc/ti)
  --fingerprint: Specific fingerprint (morgan/maccs/avalon/all)
  --model: Specific model (Ridge/SVR/RandomForest/XGBoost/LightGBM/DNN)
  --epochs: Override epochs (default from config: 200)
  --failed-only: Only retry failed experiments (R² ≤ 0)
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent dock icons
import matplotlib.pyplot as plt
import gc
import time
import subprocess
import psutil

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import tempfile
import shutil
from pathlib import Path

# Import configuration
from config import (
    DATASETS, DATASET_NAME_TO_KEY, ACTIVE_SPLIT_TYPES,
    MODEL_CONFIG, DATA_PATH, RESULT_PATH, MODEL_PATH,
    get_dataset_display_name, get_dataset_filename, get_split_type_name,
    get_code_datasets, get_code_fingerprints, OS_TYPE, RESTART_CONFIG,
    get_epochs_for_module
    # PARALLEL_CONFIG, FINGERPRINTS
)

# PyTorch imports with optimizations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ML imports
# from sklearn.preprocessing import StandardScaler  # Not used anymore
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

# ===== OS-specific optimization settings =====
# Use different parallelization strategies by OS to prevent OpenMP conflicts and optimize performance
import warnings
print(f"Operating System: {OS_TYPE}")  # OS_TYPE is imported from config

# ===== NaN/Inf handling functions =====
def clean_data(X, y=None):
    """
    Remove NaN/Inf values from features and targets

    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target vector (optional, numpy array or pandas Series)

    Returns:
        X_clean, y_clean (if y provided) or X_clean (if y not provided)
    """
    import numpy as np

    # Convert to numpy arrays if needed
    if hasattr(X, 'values'):
        X = X.values
    if y is not None and hasattr(y, 'values'):
        y = y.values

    # Find rows with NaN/Inf in features
    X_mask = np.isfinite(X).all(axis=1)

    if y is not None:
        # Find rows with NaN/Inf in targets
        y_mask = np.isfinite(y)
        # Combined mask: both X and y must be finite
        valid_mask = X_mask & y_mask

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            print(f"  [CLEAN] Removed {removed_count} samples with NaN/Inf values ({removed_count/len(X)*100:.1f}%)")

        return X_clean, y_clean
    else:
        X_clean = X[X_mask]
        removed_count = len(X) - len(X_clean)
        if removed_count > 0:
            print(f"  [CLEAN] Removed {removed_count} samples with NaN/Inf values ({removed_count/len(X)*100:.1f}%)")

        return X_clean

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
    # np.nan_to_num handles both scalars and arrays automatically
    return np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)

# OpenMP conflict prevention settings
# OpenMP is used by multiple libraries, and duplicate loading can cause conflicts
if OS_TYPE == "Windows":
    # Windows: Allow duplicate OpenMP libraries to prevent conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Configure Intel MKL and OpenMP threads appropriately (balance performance and stability)
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
elif OS_TYPE == "Darwin":  # macOS
    # macOS: Prevent OpenMP conflicts similar to Windows
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
else:  # Linux
    # Linux: Optimized settings using more threads
    os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count()))
    os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count()))

# ===== PyTorch parallelization settings =====
# Optimize PyTorch thread settings for each OS to improve performance and ensure stability
try:
    # CPU thread settings (OS-specific optimization)
    if OS_TYPE == "Windows":
        # Windows: Conservative thread settings (prevent OpenMP conflicts)
        num_threads = min(2, os.cpu_count() // 2)
        torch.set_num_threads(num_threads)  # Computation threads
        torch.set_num_interop_threads(1)    # Inter-thread communication
    elif OS_TYPE == "Darwin":  # macOS
        # macOS: Moderate thread settings
        num_threads = min(4, os.cpu_count() // 2)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(2)
    else:  # Linux
        # Linux: Aggressive thread settings (more parallelization)
        num_threads = min(8, os.cpu_count() - 1)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(4)

    print(f"PyTorch threads: {torch.get_num_threads()}")

    # CUDA settings (if available)
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Print GPU information
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # GPU memory settings
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception as e:
            print(f"Warning: Could not set GPU memory fraction: {e}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (GPU not available)")

except Exception as e:
    # Fall back to sequential processing if parallelization fails
    warnings.warn(f"Failed to set up parallelization: {e}. Falling back to sequential processing.")
    device = torch.device('cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("Using sequential processing (single thread)")

# Import custom modules
from extra_code.mol_fps_maker import (
    load_split_data,
    get_fingerprints_combined
)
# Configuration from config.py
target_path = os.path.join(RESULT_PATH, "2_standard_comp")
os.makedirs(target_path, exist_ok=True)
out_root = Path(os.path.join(RESULT_PATH, "fingerprint"))
out_root.mkdir(parents=True, exist_ok=True)

# Create save_model directory for subprocess
os.makedirs(MODEL_PATH, exist_ok=True)

# Create temp directory for intermediate files
TEMP_DIR = Path(tempfile.mkdtemp(prefix="standard_comp_"))
print(f"Using temporary directory: {TEMP_DIR}")

# Get cache config from config.py
from config import CACHE_CONFIG
FORCE_REBUILD_FINGERPRINTS = CACHE_CONFIG.get('remake_fingerprint', False)

# Model hyperparameters from config
RANDOM_STATE = MODEL_CONFIG['random_state']
BATCHSIZE = MODEL_CONFIG['batch_size']
lr = MODEL_CONFIG['learning_rate']
CV = MODEL_CONFIG['cv_folds']
# EPOCHS will be set in main() using get_epochs_for_module
EPOCHS = None  # Initialize as None, will be set properly in main()

# CV evaluation configuration
evaluate_test_during_cv = False  # Set to True to evaluate test set during each CV fold

# Fingerprint configuration - now uses renew parameter in molecular_loader_with_fp
# This is now controlled by CACHE_CONFIG in config.py

# Split mapping from config
from config import SPLIT_TYPES
SPLIT_MAP = SPLIT_TYPES

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

# QSAR Solubility Criteria Colors
QSAR_COLORS = {
    'High': '#8ECAE6',    # Vibrant light blue
    'Good': '#A7C957',    # Light green
    'Low': '#FFD166',     # Light orange/yellow
    'Poor': '#EF476F'     # Light red/salmon
}

# QSAR Criteria Labels
QSAR_LABELS = [
    'High Solubility (LogS > -2)',
    'Good Solubility (-4 < LogS <= -2)',
    'Low Solubility (-6 < LogS <= -4)',
    'Poor Solubility (LogS <= -6)'
]

def extract_target_values(df: pd.DataFrame, target_column: str) -> np.ndarray:
    """
    Extract target values from dataframe with flexible column matching.

    Attempts to find and extract the target column from a DataFrame. If the specified
    column is not found, it searches for any column containing 'log' in its name.

    Args:
        df (pd.DataFrame): Input dataframe containing target values
        target_column (str): Name of the target column to extract

    Returns:
        np.ndarray: Array of target values as float64

    Raises:
        KeyError: If target column not found and no 'log' columns exist
    """
    if target_column not in df.columns:
        # Try to find any column with 'log' in name
        log_cols = [col for col in df.columns if 'log' in col.lower()]
        if log_cols:
            target_column = log_cols[0]
        else:
            raise KeyError(f"Column '{target_column}' not found in dataframe. "
                          f"Available columns: {list(df.columns)}")

    return df[target_column].astype(float).values

def add_qsar_background(ax: plt.Axes, x_min: float, x_max: float, alpha: float = 0.5):
    """Add QSAR solubility criteria background to axis."""
    ax.axvspan(-2, x_max, facecolor=QSAR_COLORS['High'], alpha=alpha, zorder=0)
    ax.axvspan(-4, -2, facecolor=QSAR_COLORS['Good'], alpha=alpha, zorder=0)
    ax.axvspan(-6, -4, facecolor=QSAR_COLORS['Low'], alpha=alpha*1.2, zorder=0)
    ax.axvspan(x_min, -6, facecolor=QSAR_COLORS['Poor'], alpha=alpha*1.2, zorder=0)

def create_qsar_legend_handles(alpha: float = 0.5):
    """Create legend handles for QSAR criteria."""
    return [
        plt.Rectangle((0,0), 1, 1, facecolor=QSAR_COLORS['High'], alpha=alpha),
        plt.Rectangle((0,0), 1, 1, facecolor=QSAR_COLORS['Good'], alpha=alpha),
        plt.Rectangle((0,0), 1, 1, facecolor=QSAR_COLORS['Low'], alpha=alpha*1.2),
        plt.Rectangle((0,0), 1, 1, facecolor=QSAR_COLORS['Poor'], alpha=alpha*1.2)
    ]

def plot_logs_distribution_multiple(
    datasets: List[Tuple[pd.DataFrame, str, str]],
    output_path: Optional[str] = None,
    plot_type: str = 'subplots',
    figsize: Optional[Tuple[int, int]] = None,
    layout: Optional[Tuple[int, int]] = None,
    show_qsar: bool = True,
    show_stats: bool = True,
    bins: Union[int, str] = 30,
    colors: Optional[List[str]] = None,
    alpha: float = 0.6
) -> plt.Figure:
    """Plot LogS distributions for multiple datasets."""
    
    if plot_type == 'subplots':
        return _plot_multiple_subplots(
            datasets, output_path, figsize, layout, 
            show_qsar, show_stats, bins, colors
        )
    elif plot_type == 'overlay':
        return _plot_multiple_overlay(
            datasets, output_path, figsize, 
            show_qsar, show_stats, bins, colors, alpha
        )
    else:
        raise ValueError(f"plot_type must be 'subplots' or 'overlay', got '{plot_type}'")

def _plot_multiple_subplots(
    datasets: List[Tuple[pd.DataFrame, str, str]],
    output_path: Optional[str],
    figsize: Optional[Tuple[int, int]],
    layout: Optional[Tuple[int, int]],
    show_qsar: bool,
    show_stats: bool,
    bins: Union[int, str],
    colors: Optional[List[str]]
) -> plt.Figure:
    """Internal function for subplot visualization."""
    n_datasets = len(datasets)
    
    # Determine layout
    if layout is None:
        layout = (2, 2) if n_datasets <= 4 else (3, 3)
    
    # Set figure size
    if figsize is None:
        figsize = (24, 16)
    
    # Set style and create figure
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize,
                            sharex=True, sharey=True)
    
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Set colors
    if colors is None:
        colors = sns.color_palette("Set1", n_colors=n_datasets)
    
    # Extract all data for common bins
    all_data = []
    for df, target_col, _ in datasets:
        data = extract_target_values(df, target_col)
        all_data.extend(data)
    
    all_data = np.array(all_data)
    hist_bins = np.histogram_bin_edges(all_data, bins=bins)
    global_min = all_data.min() - 0.5
    global_max = all_data.max() + 0.5
    
    # Plot each dataset
    for i, (df, target_col, name) in enumerate(datasets):
        if i >= len(axes):
            break
        
        ax = axes[i]
        data = extract_target_values(df, target_col)
        
        # Set limits
        ax.set_xlim(global_min, global_max)
        
        # Add QSAR background
        if show_qsar:
            add_qsar_background(ax, global_min, global_max)
        
        # Plot histogram
        ax.hist(data, bins=hist_bins, density=True,
                color=colors[i], label=name, zorder=2,
                alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Set title
        if show_stats:
            stats = {
                'count': len(data),
                'mean': np.mean(data),
                'std': np.std(data)
            }
            title = f"{name}\n(n={stats['count']}, mu={stats['mean']:.2f}, sigma={stats['std']:.2f})"
        else:
            title = name
        
        ax.set_title(title, fontsize=FONT_SIZES['subtitle'])
        ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
        
        # Y-axis label positioning
        if i % layout[1] == 0:  # Left column
            ax.set_ylabel('Density', fontsize=FONT_SIZES['label'])
        elif i % layout[1] == layout[1] - 1:  # Right column
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel('Density', fontsize=FONT_SIZES['label'])
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.grid(axis='x', visible=False)
    
    # Hide unused subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)
    
    # Common labels
    fig.supxlabel('Experimental LogS Value', fontsize=FONT_SIZES['label'], y=0.04)
    fig.suptitle('Distribution of Experimental LogS Values by Dataset',
                 fontsize=FONT_SIZES['title'], y=0.98)
    
    # Add QSAR legend
    if show_qsar:
        qsar_handles = create_qsar_legend_handles()
        fig.legend(qsar_handles, QSAR_LABELS,
                   loc='center left', bbox_to_anchor=(0.95, 0.5),
                   fontsize=FONT_SIZES['legend'],
                   title='QSAR Solubility Criteria',
                   title_fontsize=FONT_SIZES['legend'])
    
    # Adjust layout
    fig.tight_layout(rect=[0.06, 0.06, 0.95, 0.93])
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig

def save_split_comparison(split_type, split_results, split_predictions, output_dir):
    """
    Create comparison plots for all datasets and fingerprints in a split type

    Parameters:
    -----------
    split_type : str
        Split type (e.g., 'rm')
    split_results : list
        List of all results for this split
    split_predictions : dict
        All predictions for this split
    output_dir : str
        Base output directory
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend to prevent dock icons
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from pathlib import Path
    import seaborn as sns

    # Create split-level directory
    split_dir = Path(output_dir) / f"split_{split_type}_comparison"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save split-level CSV
    if split_results:
        csv_path = split_dir / f"{split_type}_all_results.csv"
        pd.DataFrame(split_results).to_csv(csv_path, index=False)
        print(f"  Saved split-level CSV: {csv_path}")

    # Create comparison plots
    models = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
    datasets = ['ws', 'de', 'lo', 'hu']
    fingerprints = ['morgan', 'maccs', 'avalon', 'morgan+maccs', 'morgan+avalon', 'maccs+avalon', 'all']

    # 1. Heatmap: R² scores for all combinations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Create R² matrix
        r2_matrix = []
        y_labels = []

        for dataset in datasets:
            row = []
            for fp in fingerprints:
                # Find R² value
                r2_val = np.nan
                if dataset in split_predictions and fp in split_predictions[dataset]:
                    # Try different key formats
                    if model == 'DNN':
                        keys_to_try = ['DNN_cv_type1', 'DNN_cv_type2']
                    else:
                        keys_to_try = [f'{model}_cv']

                    for key in keys_to_try:
                        if key in split_predictions[dataset][fp]:
                            pred_data = split_predictions[dataset][fp][key]
                            if 'y_true' in pred_data and 'y_pred' in pred_data:
                                y_true = pred_data['y_true']
                                y_pred = pred_data['y_pred']
                                if len(y_true) > 0:
                                    r2_val = r2_score(y_true, y_pred)
                                    break
                row.append(r2_val)
            r2_matrix.append(row)
            y_labels.append(dataset.upper())

        # Create heatmap
        r2_matrix = np.array(r2_matrix)
        im = ax.imshow(r2_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0, aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(fingerprints)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels([fp.replace('+', '\n+') for fp in fingerprints], rotation=45, ha='right')
        ax.set_yticklabels(y_labels)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(datasets)):
            for j in range(len(fingerprints)):
                val = r2_matrix[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                  ha='center', va='center',
                                  color='white' if val < 0.5 else 'black',
                                  fontsize=8)

        ax.set_title(f'{model}', fontsize=12, fontweight='bold')

    plt.suptitle(f'R² Scores Heatmap - Split: {split_type}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    heatmap_path = split_dir / f"{split_type}_r2_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {heatmap_path.name}")

    # 2. Box plot: Model performance across all datasets/fingerprints
    fig, ax = plt.subplots(figsize=(14, 8))

    model_r2_values = {model: [] for model in models}

    for dataset in datasets:
        if dataset not in split_predictions:
            continue
        for fp in fingerprints:
            if fp not in split_predictions[dataset]:
                continue
            for model in models:
                if model == 'DNN':
                    keys_to_try = ['DNN_cv_type1', 'DNN_cv_type2']
                else:
                    keys_to_try = [f'{model}_cv']

                for key in keys_to_try:
                    if key in split_predictions[dataset][fp]:
                        pred_data = split_predictions[dataset][fp][key]
                        if 'y_true' in pred_data and 'y_pred' in pred_data:
                            y_true = pred_data['y_true']
                            y_pred = pred_data['y_pred']
                            if len(y_true) > 0:
                                r2 = r2_score(y_true, y_pred)
                                model_r2_values[model].append(r2)
                                break

    # Create box plot
    box_data = [model_r2_values[model] for model in models]
    bp = ax.boxplot(box_data, labels=models, patch_artist=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title(f'Model Performance Distribution - Split: {split_type}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='R²=0.5')
    ax.legend()

    boxplot_path = split_dir / f"{split_type}_model_boxplot.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved boxplot: {boxplot_path.name}")

    # 3. Bar chart: Average performance by fingerprint
    fig, ax = plt.subplots(figsize=(14, 8))

    fp_avg_r2 = {fp: [] for fp in fingerprints}

    for dataset in datasets:
        if dataset not in split_predictions:
            continue
        for fp in fingerprints:
            if fp not in split_predictions[dataset]:
                continue
            for model in models:
                if model == 'DNN':
                    keys_to_try = ['DNN_cv_type1', 'DNN_cv_type2']
                else:
                    keys_to_try = [f'{model}_cv']

                for key in keys_to_try:
                    if key in split_predictions[dataset][fp]:
                        pred_data = split_predictions[dataset][fp][key]
                        if 'y_true' in pred_data and 'y_pred' in pred_data:
                            y_true = pred_data['y_true']
                            y_pred = pred_data['y_pred']
                            if len(y_true) > 0:
                                r2 = r2_score(y_true, y_pred)
                                fp_avg_r2[fp].append(r2)
                                break

    # Calculate averages
    fp_names = []
    fp_means = []
    fp_stds = []

    for fp in fingerprints:
        if fp_avg_r2[fp]:
            fp_names.append(fp)
            fp_means.append(np.mean(fp_avg_r2[fp]))
            fp_stds.append(np.std(fp_avg_r2[fp]))

    # Create bar chart
    x_pos = np.arange(len(fp_names))
    ax.bar(x_pos, fp_means, yerr=fp_stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(fp_names, rotation=45, ha='right')
    ax.set_ylabel('Average R² Score', fontsize=12)
    ax.set_xlabel('Fingerprint Type', fontsize=12)
    ax.set_title(f'Average Performance by Fingerprint - Split: {split_type}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='R²=0.5')

    barchart_path = split_dir / f"{split_type}_fingerprint_performance.png"
    plt.savefig(barchart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved bar chart: {barchart_path.name}")

def save_fingerprint_results(dataset, split_type, fp_type, fp_results, fp_predictions, output_dir):
    """
    Save results and create plots for a specific dataset/split/fingerprint combination
    Separates DNN results by CV type (type1/type2)

    Parameters:
    -----------
    dataset : str
        Dataset name
    split_type : str
        Split type (e.g., 'rm')
    fp_type : str
        Fingerprint type (e.g., 'morgan')
    fp_results : list
        List of result dictionaries for this fingerprint
    fp_predictions : dict
        Dictionary of predictions for this fingerprint
    output_dir : str
        Base output directory
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend to prevent dock icons
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from pathlib import Path

    # Create base folder structure: dataset/split_type/fingerprint/
    fp_dir = Path(output_dir) / dataset / f"split_{split_type}" / f"fp_{fp_type}"
    fp_dir.mkdir(parents=True, exist_ok=True)

    # Separate results by CV type for DNN
    type1_results = []
    type2_results = []
    other_results = []

    for result in fp_results:
        if result.get('model') == 'DNN' and result.get('cv_type') == 'type1':
            type1_results.append(result)
        elif result.get('model') == 'DNN' and result.get('cv_type') == 'type2':
            type2_results.append(result)
        else:
            other_results.append(result)

    # Save type1 results if they exist
    if type1_results or any('DNN_cv_type1' in fp_predictions for _ in [1]):
        type1_dir = fp_dir / "cv_type1"
        type1_dir.mkdir(parents=True, exist_ok=True)

        if type1_results:
            csv_path = type1_dir / "dnn_results.csv"
            pd.DataFrame(type1_results).to_csv(csv_path, index=False)
            print(f"    Saved Type1 CSV: {csv_path}")

    # Save type2 results if they exist
    if type2_results or any('DNN_cv_type2' in fp_predictions for _ in [1]):
        type2_dir = fp_dir / "cv_type2"
        type2_dir.mkdir(parents=True, exist_ok=True)

        if type2_results:
            csv_path = type2_dir / "dnn_results.csv"
            pd.DataFrame(type2_results).to_csv(csv_path, index=False)
            print(f"    Saved Type2 CSV: {csv_path}")

    # Save other model results in main directory
    if other_results:
        csv_path = fp_dir / "other_models_results.csv"
        pd.DataFrame(other_results).to_csv(csv_path, index=False)
        print(f"    Saved other models CSV: {csv_path}")

    # Save all results together
    if fp_results:
        csv_path = fp_dir / "all_results.csv"
        pd.DataFrame(fp_results).to_csv(csv_path, index=False)
        print(f"    Saved combined CSV: {csv_path}")

    # Create individual model plots
    models = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']

    for model in models:
        # Handle DNN separately for type1 and type2
        if model == 'DNN':
            # Process DNN Type1
            if 'DNN_cv_type1' in fp_predictions:
                pred_data = fp_predictions['DNN_cv_type1']
                if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                    y_true = np.array(pred_data['y_true'])
                    y_pred = np.array(pred_data['y_pred'])

                    if len(y_true) > 0:
                        # Calculate metrics
                        r2 = r2_score(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mae = mean_absolute_error(y_true, y_pred)

                        # Create scatter plot
                        plt.figure(figsize=(8, 8))
                        plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

                        # Plot diagonal line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

                        # Labels and title
                        plt.xlabel('Actual Values', fontsize=12)
                        plt.ylabel('Predicted Values', fontsize=12)
                        plt.title(f'DNN Model (Type1)\n{dataset} - {split_type} - {fp_type}\nR²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}', fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Save plot in cv_type1 folder
                        type1_dir = fp_dir / "cv_type1"
                        type1_dir.mkdir(parents=True, exist_ok=True)
                        plot_path = type1_dir / "dnn_type1_scatter.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"      Saved DNN Type1 plot: R²={r2:.3f}")

            # Process DNN Type2
            if 'DNN_cv_type2' in fp_predictions:
                pred_data = fp_predictions['DNN_cv_type2']
                if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                    y_true = np.array(pred_data['y_true'])
                    y_pred = np.array(pred_data['y_pred'])

                    if len(y_true) > 0:
                        # Calculate metrics
                        r2 = r2_score(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mae = mean_absolute_error(y_true, y_pred)

                        # Create scatter plot
                        plt.figure(figsize=(8, 8))
                        plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

                        # Plot diagonal line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

                        # Labels and title
                        plt.xlabel('Actual Values', fontsize=12)
                        plt.ylabel('Predicted Values', fontsize=12)
                        plt.title(f'DNN Model (Type2)\n{dataset} - {split_type} - {fp_type}\nR²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}', fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Save plot in cv_type2 folder
                        type2_dir = fp_dir / "cv_type2"
                        type2_dir.mkdir(parents=True, exist_ok=True)
                        plot_path = type2_dir / "dnn_type2_scatter.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"      Saved DNN Type2 plot: R²={r2:.3f}")
        else:
            # Process other models
            cv_key = f"{model}_cv"
            if cv_key in fp_predictions:
                pred_data = fp_predictions[cv_key]
                if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                    y_true = np.array(pred_data['y_true'])
                    y_pred = np.array(pred_data['y_pred'])

                    if len(y_true) > 0:
                        # Calculate metrics
                        r2 = r2_score(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        mae = mean_absolute_error(y_true, y_pred)

                        # Create scatter plot
                        plt.figure(figsize=(8, 8))
                        plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

                        # Plot diagonal line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

                        # Labels and title
                        plt.xlabel('Actual Values', fontsize=12)
                        plt.ylabel('Predicted Values', fontsize=12)
                        plt.title(f'{model} Model\n{dataset} - {split_type} - {fp_type}\nR²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}', fontsize=14)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()

                        # Save plot in main directory
                        model_lower = model.lower()
                        plot_path = fp_dir / f"{model_lower}_scatter.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"      Saved plot: {model} (R²={r2:.3f})")

    # Create combined comparison plots for both DNN types if available
    dnn_types_found = []
    if 'DNN_cv_type1' in fp_predictions:
        dnn_types_found.append('type1')
    if 'DNN_cv_type2' in fp_predictions:
        dnn_types_found.append('type2')

    for dnn_type in dnn_types_found if dnn_types_found else ['none']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, model in enumerate(models):
            ax = axes[idx]

            # Find prediction data
            pred_data = None
            model_title = model

            if model == 'DNN':
                if dnn_type == 'type1' and 'DNN_cv_type1' in fp_predictions:
                    pred_data = fp_predictions['DNN_cv_type1']
                    model_title = 'DNN (Type1)'
                elif dnn_type == 'type2' and 'DNN_cv_type2' in fp_predictions:
                    pred_data = fp_predictions['DNN_cv_type2']
                    model_title = 'DNN (Type2)'
                elif dnn_type == 'none':
                    # Try to find any DNN data
                    if 'DNN_cv_type1' in fp_predictions:
                        pred_data = fp_predictions['DNN_cv_type1']
                        model_title = 'DNN (Type1)'
                    elif 'DNN_cv_type2' in fp_predictions:
                        pred_data = fp_predictions['DNN_cv_type2']
                        model_title = 'DNN (Type2)'
            else:
                cv_key = f"{model}_cv"
                if cv_key in fp_predictions:
                    pred_data = fp_predictions[cv_key]

            if pred_data and 'y_true' in pred_data and 'y_pred' in pred_data:
                y_true = np.array(pred_data['y_true'])
                y_pred = np.array(pred_data['y_pred'])

                if len(y_true) > 0:
                    r2 = r2_score(y_true, y_pred)

                    ax.scatter(y_true, y_pred, alpha=0.6, s=30)
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                    ax.set_title(f'{model_title} (R²={r2:.3f})')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(model_title)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(model_title)

        if dnn_type != 'none':
            plt.suptitle(f'Model Comparison (DNN {dnn_type.upper()}): {dataset} - {split_type} - {fp_type}', fontsize=16, fontweight='bold')
        else:
            plt.suptitle(f'Model Comparison: {dataset} - {split_type} - {fp_type}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save combined plot in appropriate directory
        if dnn_type == 'type1':
            save_dir = fp_dir / "cv_type1"
            save_dir.mkdir(parents=True, exist_ok=True)
            combined_path = save_dir / "all_models_comparison_type1.png"
        elif dnn_type == 'type2':
            save_dir = fp_dir / "cv_type2"
            save_dir.mkdir(parents=True, exist_ok=True)
            combined_path = save_dir / "all_models_comparison_type2.png"
        else:
            combined_path = fp_dir / "all_models_comparison.png"

        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved comparison plot: {combined_path.name}")


def _plot_multiple_overlay(
    datasets: List[Tuple[pd.DataFrame, str, str]],
    output_path: Optional[str],
    figsize: Optional[Tuple[int, int]],
    show_qsar: bool,
    show_stats: bool,
    bins: Union[int, str],
    colors: Optional[List[str]],
    alpha: float
) -> plt.Figure:
    """Internal function for overlay visualization."""
    # Implementation similar to subplots but with overlaid histograms
    pass  # Simplified for brevity

def plot_original_datasets_distribution():
    """Plot distribution for original datasets (ws496, delaney, lovric, huuskonen)"""
    
    # Try to load original datasets
    try:
        datasets = []
        
        # ws496
        df1 = pd.read_csv(Path('data') / 'ws496_logS.csv', dtype={'SMILES': 'string'})
        datasets.append((df1, 'exp', 'ws496'))

        # Delaney
        df2 = pd.read_csv(Path('data') / 'delaney-processed.csv', dtype={'smiles': 'string'})
        datasets.append((df2, 'ESOL predicted log solubility in mols per litre', 'Delaney (ESOL)'))

        # Lovric
        df3 = pd.read_csv(Path('data') / 'Lovric2020_logS0.csv', dtype={'isomeric_smiles': 'string'})
        datasets.append((df3, 'logS0', 'Lovrić et al.'))

        # Huuskonen
        df4 = pd.read_csv(Path('data') / 'huusk.csv', dtype={'SMILES': 'string'})
        datasets.append((df4, 'Solubility', get_dataset_display_name('huusk')))
        
        # Create visualization
        plot_logs_distribution_multiple(
            datasets,
            output_path='./result/2_standard_comp/LogS_Original_Datasets.png',
            plot_type='subplots',
            layout=(2, 2),
            figsize=(24, 16)
        )
        
    except FileNotFoundError as e:
        print(f"Warning: {e.filename} not found. Using dummy data.")
        # Generate dummy data for demonstration
        np.random.seed(0)
        datasets = [
            (pd.DataFrame({'logS': np.random.normal(-3, 2, 496)}), 'logS', 'ws496'),
            (pd.DataFrame({'logS': np.random.normal(-2.5, 1.5, 1144)}), 'logS', 'Delaney (ESOL)'),
            (pd.DataFrame({'logS': np.random.normal(-3.5, 2.5, 998)}), 'logS', 'Lovrić et al.'),
            (pd.DataFrame({'logS': np.random.normal(-4, 2.2, 1297)}), 'logS', get_dataset_display_name('huusk'))
        ]
        
        plot_logs_distribution_multiple(
            datasets,
            output_path='./result/2_standard_comp/LogS_Original_Datasets_Dummy.png',
            plot_type='subplots'
        )

def plot_enhanced_results_comparison(results_df, all_predictions, output_dir="result/2_standard_comp"):
    """Create enhanced comparison plots as requested"""
    
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Model colors and markers - More vibrant colors
    model_colors = {
        'Ridge': '#FF1493',      # Deep Pink
        'SVR': '#00CED1',        # Dark Turquoise
        'RandomForest': '#32CD32', # Lime Green
        'XGBoost': '#FF4500',    # Orange Red
        'LightGBM': '#9370DB',   # Medium Purple
        'DNN': '#FFD700'         # Gold
    }
    
    model_markers = {
        'Ridge': 'o',
        'SVR': 's', 
        'RandomForest': '^',
        'XGBoost': 'D',
        'LightGBM': 'v',
        'DNN': 'p'
    }
    
    # Create scatter plots for actual vs predicted values
    scatter_dir = plot_dir / "scatter_plots"
    scatter_dir.mkdir(exist_ok=True)
    
    scatter_count = 0
    
    for split_type, split_data in all_predictions.items():
        for dataset, dataset_data in split_data.items():
            for fp_type, fp_data in dataset_data.items():
                # 1. Create individual dataset plot for Simple training
                plt.figure(figsize=(12, 10))
                
                all_true_simple = []
                all_pred_simple = []
                model_r2_values = {}
                
                for model in ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']:
                    simple_key = f"{model}_simple"
                    
                    if simple_key not in fp_data:
                        continue
                    
                    simple_data = fp_data[simple_key]
                    
                    if 'y_true' not in simple_data or 'y_pred' not in simple_data:
                        continue
                    
                    y_true_simple = simple_data['y_true']
                    y_pred_simple = simple_data['y_pred']
                    
                    # Calculate R² for this model
                    r2_val = r2_score(y_true_simple, y_pred_simple)
                    model_r2_values[model] = r2_val
                    
                    # Plot simple training (points only, no error bars) - Larger markers
                    plt.scatter(y_true_simple, y_pred_simple, alpha=0.9, s=120, 
                              color=model_colors[model], marker='*',
                              label=f'{model} (R²={r2_val:.3f})', edgecolors='black', linewidth=1.5)
                    
                    all_true_simple.extend(y_true_simple)
                    all_pred_simple.extend(y_pred_simple)
                
                # Add diagonal line
                if all_true_simple and all_pred_simple:
                    min_val = min(min(all_true_simple), min(all_pred_simple))
                    max_val = max(max(all_true_simple), max(all_pred_simple))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=4, alpha=0.9, label='Perfect Prediction')
                
                # Calculate overall metrics
                if all_true_simple and all_pred_simple:
                    r2_simple = r2_score(all_true_simple, all_pred_simple)
                    rmse_simple = np.sqrt(mean_squared_error(all_true_simple, all_pred_simple))
                    
                    # Add metrics to plot (much larger font)
                    plt.text(0.05, 0.95, f'Overall R² = {r2_simple:.3f}\nOverall RMSE = {rmse_simple:.3f}', 
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2),
                            verticalalignment='top', fontsize=20, fontweight='bold')
                
                # Labels and title (much larger font)
                plt.xlabel('Actual LogS', fontsize=22, fontweight='bold')
                plt.ylabel('Predicted LogS', fontsize=22, fontweight='bold')
                plt.title(f'{dataset.upper()} Dataset - Simple Training - {split_type.upper()}/{fp_type.upper()}',
                         fontsize=24, fontweight='bold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                
                # Legend (much larger font)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, frameon=True, 
                          fancybox=True, shadow=True, borderpad=1)
                
                # Grid
                plt.grid(True, alpha=0.3)
                
                # Equal aspect ratio
                plt.axis('equal')
                
                # Adjust layout
                plt.tight_layout()
                
                # Create organized folder structure
                split_dir = scatter_dir / f"split_{split_type}"
                split_dir.mkdir(exist_ok=True)
                dataset_dir = split_dir / f"dataset_{dataset}"
                dataset_dir.mkdir(exist_ok=True)
                
                # Save in organized structure
                filename = f"scatter_{split_type}_{dataset}_{fp_type}_simple.png"
                plt.savefig(dataset_dir / filename, dpi=300, bbox_inches='tight')
                
                # Also save in main scatter directory for backward compatibility
                plt.savefig(scatter_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                scatter_count += 1
                
                # 2. Create individual dataset plot for CV Method 1
                plt.figure(figsize=(12, 10))
                
                all_true_cv1 = []
                all_pred_cv1 = []
                model_cv1_r2_values = {}
                
                for model in ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']:
                    cv_key = f"{model}_cv_method1"
                    
                    if cv_key not in fp_data:
                        continue
                    
                    cv_data = fp_data[cv_key]
                    
                    if 'y_true' not in cv_data or 'y_pred' not in cv_data:
                        continue
                    
                    y_true_cv = cv_data['y_true']
                    y_pred_cv = cv_data['y_pred']
                    y_pred_std = cv_data.get('y_pred_std', None)
                    cv_test_r2_values = cv_data.get('cv_test_r2_values', [])
                    
                    # Calculate CV R² mean and std from test evaluations
                    if cv_test_r2_values:
                        cv_r2_mean = np.mean(cv_test_r2_values)
                        cv_r2_std = np.std(cv_test_r2_values)
                        model_cv1_r2_values[model] = (cv_r2_mean, cv_r2_std)
                        
                        # Plot CV training with error bars if available
                        if y_pred_std is not None:
                            plt.errorbar(y_true_cv, y_pred_cv, yerr=y_pred_std, fmt='none', 
                                       color=model_colors[model], alpha=0.3, capsize=2)
                        
                        # CV Method 1 results - Larger markers
                        plt.scatter(y_true_cv, y_pred_cv, alpha=0.9, s=120, 
                                  color=model_colors[model], marker='*',
                                  label=f'{model} (Test R²={cv_r2_mean:.3f}±{cv_r2_std:.3f})', 
                                  edgecolors='black', linewidth=1.5)
                    else:
                        # Fallback if no CV data
                        r2_cv = r2_score(y_true_cv, y_pred_cv)
                        plt.scatter(y_true_cv, y_pred_cv, alpha=0.9, s=120, 
                                  color=model_colors[model], marker='*',
                                  label=f'{model} (R²={r2_cv:.3f})', 
                                  edgecolors='black', linewidth=1.5)
                    
                    all_true_cv1.extend(y_true_cv)
                    all_pred_cv1.extend(y_pred_cv)
                
                # Add diagonal line
                if all_true_cv1 and all_pred_cv1:
                    min_val = min(min(all_true_cv1), min(all_pred_cv1))
                    max_val = max(max(all_true_cv1), max(all_pred_cv1))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=4, alpha=0.9, label='Perfect Prediction')
                
                # Calculate overall metrics
                if all_true_cv1 and all_pred_cv1:
                    r2_cv = r2_score(all_true_cv1, all_pred_cv1)
                    rmse_cv = np.sqrt(mean_squared_error(all_true_cv1, all_pred_cv1))
                    
                    # Add metrics to plot (much larger font)
                    plt.text(0.05, 0.95, f'Overall R² = {r2_cv:.3f}\nOverall RMSE = {rmse_cv:.3f}', 
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2),
                            verticalalignment='top', fontsize=20, fontweight='bold')
                
                # Labels and title (much larger font)
                plt.xlabel('Actual LogS', fontsize=22, fontweight='bold')
                plt.ylabel('Predicted LogS', fontsize=22, fontweight='bold')
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.title(f'{dataset.upper()} Dataset - CV Method 1 (Test during CV) - {split_type.upper()}/{fp_type.upper()}',
                         fontsize=24, fontweight='bold')
                
                # Legend (much larger font)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, frameon=True, 
                          fancybox=True, shadow=True, borderpad=1)
                
                # Grid
                plt.grid(True, alpha=0.3)
                
                # Equal aspect ratio
                plt.axis('equal')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save in organized structure
                filename = f"scatter_{split_type}_{dataset}_{fp_type}_cv_method1.png"
                plt.savefig(dataset_dir / filename, dpi=300, bbox_inches='tight')
                plt.savefig(scatter_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
                scatter_count += 1
                
    
    # 3. Create combined plot for available datasets (Simple training)
    # Get actual datasets from predictions
    available_datasets = set()
    for split_data in all_predictions.values():
        available_datasets.update(split_data.keys())
    datasets = sorted(list(available_datasets))
    
    # Only create combined plot if we have data
    if not datasets:
        print("No datasets available for combined plots")
        return scatter_count
    
    # For single dataset, just skip the combined plots
    if len(datasets) == 1:
        print(f"Only one dataset ({datasets[0]}) available, skipping combined plots")
        # Skip to bar chart generation
        pass
    else:
        # Create subplot grid based on number of datasets
        n_datasets = len(datasets)
        if n_datasets == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        elif n_datasets <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            
            all_true_simple = []
            all_pred_simple = []
            
            for model in ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']:
                simple_key = f"{model}_simple"
                
                # Get data from the first split and fingerprint (for simplicity)
                split_type = list(all_predictions.keys())[0]
                fp_type = list(all_predictions[split_type][dataset].keys())[0]
                fp_data = all_predictions[split_type][dataset][fp_type]
                
                if simple_key not in fp_data:
                    continue
                
                simple_data = fp_data[simple_key]
                
                if 'y_true' not in simple_data or 'y_pred' not in simple_data:
                    continue
                
                y_true_simple = simple_data['y_true']
                y_pred_simple = simple_data['y_pred']
                
                # Calculate R² for this model
                r2_val = r2_score(y_true_simple, y_pred_simple)
                
                # Plot with dataset color and model marker
                ax.scatter(y_true_simple, y_pred_simple, alpha=0.7, s=60, 
                          color=model_colors[model], marker=model_markers[model],
                          label=f'{model} (R²={r2_val:.3f})', edgecolors='black', linewidth=0.8)
                
                all_true_simple.extend(y_true_simple)
                all_pred_simple.extend(y_pred_simple)
            
            # Add diagonal line
            if all_true_simple and all_pred_simple:
                min_val = min(min(all_true_simple), min(all_pred_simple))
                max_val = max(max(all_true_simple), max(all_pred_simple))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='Perfect Prediction')
            
            # Calculate overall metrics
            if all_true_simple and all_pred_simple:
                r2_simple = r2_score(all_true_simple, all_pred_simple)
                rmse_simple = np.sqrt(mean_squared_error(all_true_simple, all_pred_simple))
                
                # Add metrics to plot
                ax.text(0.05, 0.95, f'R² = {r2_simple:.3f}\nRMSE = {rmse_simple:.3f}', 
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        verticalalignment='top', fontsize=12, fontweight='bold')
            
            # Labels and title
            ax.set_xlabel('Actual LogS', fontsize=14)
            ax.set_ylabel('Predicted LogS', fontsize=14)
            ax.set_title(f'{dataset.upper()} Dataset - Simple Training', fontsize=16, fontweight='bold')
            
            # Legend
            ax.legend(fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3)
            
            # Equal aspect ratio
            ax.axis('equal')
        
        plt.suptitle(f'All Datasets - Simple Training - {split_type.upper()}/{fp_type.upper()}', 
                     fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Save
        filename = f"scatter_{split_type}_{fp_type}_all_datasets_simple.png"
        plt.savefig(scatter_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        scatter_count += 1
    
        # 4. Create combined plot for all datasets (CV Method 1) - only if multiple datasets
        if n_datasets == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        elif n_datasets <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        all_true_cv = []
        all_pred_cv = []
        
        for model in ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']:
            cv_key = f"{model}_cv_method1"
            
            # Get data from the first split and fingerprint (for simplicity)
            split_type = list(all_predictions.keys())[0]
            fp_type = list(all_predictions[split_type][dataset].keys())[0]
            fp_data = all_predictions[split_type][dataset][fp_type]
            
            if cv_key not in fp_data:
                continue
            
            cv_data = fp_data[cv_key]
            
            if 'y_true' not in cv_data or 'y_pred' not in cv_data:
                continue
            
            y_true_cv = cv_data['y_true']
            y_pred_cv = cv_data['y_pred']
            y_pred_std = cv_data.get('y_pred_std', None)
            cv_r2_values = cv_data.get('cv_r2_values', [])
            
            # Get test R² values from CV Method 1
            cv_test_r2_values = cv_data.get('cv_test_r2_values', [])
            
            # Calculate CV R² mean and std
            if cv_test_r2_values:
                cv_r2_mean = np.mean(cv_test_r2_values)
                cv_r2_std = np.std(cv_test_r2_values)
                
                # Plot CV training with error bars if available
                if y_pred_std is not None:
                    ax.errorbar(y_true_cv, y_pred_cv, yerr=y_pred_std, fmt='none', 
                               color=model_colors[model], alpha=0.3, capsize=2)
                
                ax.scatter(y_true_cv, y_pred_cv, alpha=0.7, s=60, 
                          color=model_colors[model], marker=model_markers[model],
                          label=f'{model} (R²={cv_r2_mean:.3f}±{cv_r2_std:.3f})', 
                          edgecolors='black', linewidth=0.8)
            else:
                # Fallback if no CV data
                r2_cv = r2_score(y_true_cv, y_pred_cv)
                ax.scatter(y_true_cv, y_pred_cv, alpha=0.7, s=60, 
                          color=model_colors[model], marker=model_markers[model],
                          label=f'{model} (R²={r2_cv:.3f})', 
                          edgecolors='black', linewidth=0.8)
            
            all_true_cv.extend(y_true_cv)
            all_pred_cv.extend(y_pred_cv)
        
        # Add diagonal line
        if all_true_cv and all_pred_cv:
            min_val = min(min(all_true_cv), min(all_pred_cv))
            max_val = max(max(all_true_cv), max(all_pred_cv))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8, label='Perfect Prediction')
        
        # Calculate overall metrics
        if all_true_cv and all_pred_cv:
            r2_cv = r2_score(all_true_cv, all_pred_cv)
            rmse_cv = np.sqrt(mean_squared_error(all_true_cv, all_pred_cv))
            
            # Add metrics to plot
            ax.text(0.05, 0.95, f'R² = {r2_cv:.3f}\nRMSE = {rmse_cv:.3f}', 
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    verticalalignment='top', fontsize=12, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('Actual LogS', fontsize=14)
        ax.set_ylabel('Predicted LogS', fontsize=14)
        ax.set_title(f'{dataset.upper()} Dataset - CV Method 1', fontsize=16, fontweight='bold')
        
        # Legend
        ax.legend(fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        ax.axis('equal')
    
    plt.suptitle(f'All Datasets - CV Method 1 - {split_type.upper()}/{fp_type.upper()}', 
                 fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"scatter_{split_type}_{fp_type}_all_datasets_cv_method1.png"
    plt.savefig(scatter_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    scatter_count += 1
    
    # 5. Create CV methods comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        model_names = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
        cv_method1_r2 = []
        cv_method2_r2 = []
        
        for model in model_names:
            # Get data from the first split and fingerprint (for simplicity)
            split_type = list(all_predictions.keys())[0]
            fp_type = list(all_predictions[split_type][dataset].keys())[0]
            fp_data = all_predictions[split_type][dataset][fp_type]
            
            # Method 1 R²
            cv1_key = f"{model}_cv_method1"
            if cv1_key in fp_data and 'cv_test_r2_values' in fp_data[cv1_key]:
                cv_method1_r2.append(np.mean(fp_data[cv1_key]['cv_test_r2_values']))
            else:
                cv_method1_r2.append(0)
            
            # Method 2 R²
            cv2_key = f"{model}_cv_method2"
            if cv2_key in fp_data:
                y_true = fp_data[cv2_key]['y_true']
                y_pred = fp_data[cv2_key]['y_pred']
                cv_method2_r2.append(r2_score(y_true, y_pred))
            else:
                cv_method2_r2.append(0)
        
        # Create grouped bar plot
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cv_method1_r2, width, label='CV Method 1 (Test during CV)', alpha=0.8)
        # bars2 = ax.bar(x + width/2, cv_method2_r2, width, label='CV Method 2 (Retrain on full)', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('R² Score', fontsize=14)
        ax.set_title(f'{dataset.upper()} Dataset - CV Methods Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
    
    plt.suptitle('CV Method 1 vs Method 2 Comparison', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"cv_methods_comparison_{split_type}_{fp_type}.png"
    plt.savefig(scatter_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    scatter_count += 1
    
    print(f"Saved {scatter_count} scatter plots to {scatter_dir}")
    print(f"All plots saved to {plot_dir}")

# ===== PyTorch DNN model definition =====
# Import SimpleDNN from centralized location
from extra_code.ano_feature_selection import SimpleDNN

# Create alias for backward compatibility
class DNNModel(SimpleDNN):
    """
    DNN model optimized for binary molecular fingerprints

    This is now an alias for SimpleDNN from ano_feature_selection.py
    to maintain backward compatibility while ensuring all models use
    the same centralized definition.
    """
    def __init__(self, input_dim, dropout_rate=0.2, use_batch_norm=True):
        # Call SimpleDNN with compatible interface
        super(DNNModel, self).__init__(input_dim, hidden_dims=[1024, 496], dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=0.1)

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def create_prediction_scatter_plots(all_predictions, results_df, output_dir):
    """Create scatter plots of actual vs predicted values for each split type"""
    
    print("  Creating prediction scatter plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    
    # Process each split type
    for split_type in all_predictions.keys():
        print(f"    Processing split type: {split_type}")
        
        # Create split type folder
        split_dir = os.path.join(output_dir, f'split_{split_type}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Get all fingerprints used
        fingerprints_used = set()
        for dataset_data in all_predictions[split_type].values():
            fingerprints_used.update(dataset_data.keys())
        
        # Process each fingerprint
        for fp_type in fingerprints_used:
            print(f"      Processing fingerprint: {fp_type}")
            
            # Create figure with 1x4 subplots for 4 datasets
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            dataset_order = ['ws', 'de', 'lo', 'hu']
            dataset_names = {
                'ws': 'WS496',
                'de': 'Delaney', 
                'lo': 'Lovric2020',
                'hu': 'Huuskonen'
            }
            
            # Collect results for CSV
            scatter_results = []
            
            for idx, dataset in enumerate(dataset_order):
                if dataset not in all_predictions[split_type]:
                    axes[idx].text(0.5, 0.5, f'No data for {dataset_names[dataset]}',
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(f'{dataset_names[dataset]} Dataset')
                    continue
                
                if fp_type not in all_predictions[split_type][dataset]:
                    axes[idx].text(0.5, 0.5, f'No {fp_type} data',
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(f'{dataset_names[dataset]} Dataset')
                    continue
                
                # Get predictions for all models and training methods
                models = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
                # Use vibrant colors from model_colors dictionary
                vibrant_colors = {
                    'Ridge': '#FF1493',      # Deep Pink
                    'SVR': '#00CED1',        # Dark Turquoise  
                    'RandomForest': '#32CD32', # Lime Green
                    'XGBoost': '#FF4500',    # Orange Red
                    'LightGBM': '#9370DB',   # Medium Purple
                    'DNN': '#FFD700'         # Gold
                }
                
                ax = axes[idx]
                
                # Track min/max for diagonal line
                all_y_true = []
                all_y_pred = []
                
                legend_elements = []
                
                for model_idx, model in enumerate(models):
                    # CV Method 1 training
                    cv1_key = f"{model}_cv_method1"
                    if cv1_key in all_predictions[split_type][dataset][fp_type]:
                        pred_data = all_predictions[split_type][dataset][fp_type][cv1_key]
                        y_true = pred_data['y_true']
                        y_pred = pred_data['y_pred']
                        
                        if len(y_true) > 0:
                            all_y_true.extend(y_true)
                            all_y_pred.extend(y_pred)
                            
                            # Calculate metrics
                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)
                            mse = mean_squared_error(y_true, y_pred)
                            
                            # Get CV results from results_df
                            cv_results = results_df[
                                (results_df['split'] == split_type) &
                                (results_df['dataset'] == dataset) &
                                (results_df['fingerprint'] == fp_type) &
                                (results_df['model'] == model) &
                                (results_df['training'] == 'cv_method1')
                            ]
                            
                            # Get CV statistics if available
                            cv_info = ""
                            if not cv_results.empty:
                                cv_test_mean = cv_results.iloc[0].get('cv_test_r2_mean', r2)
                                cv_test_std = cv_results.iloc[0].get('cv_test_r2_std', 0.0)
                                cv_info = f" (CV: {cv_test_mean:.3f}±{cv_test_std:.3f})"
                            
                            # Plot scatter with vibrant colors and star markers
                            ax.scatter(y_true, y_pred, alpha=0.9, s=100, marker='*',
                                     color=vibrant_colors[model], label=f'{model}{cv_info}',
                                     edgecolors='black', linewidth=0.5)
                            
                            # Store results for CSV
                            result_entry = {
                                'split_type': split_type,
                                'dataset': dataset,
                                'fingerprint': fp_type,
                                'model': model,
                                'training': 'cv_method1',
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'mse': mse,
                                'n_samples': len(y_true)
                            }
                            
                            # Add CV results if available
                            if not cv_results.empty:
                                result_entry['train_time'] = cv_results.iloc[0].get('train_time', 0)
                                result_entry['memory_used_mb'] = cv_results.iloc[0].get('memory_used_mb', 0)
                                result_entry['cv_val_r2_mean'] = cv_results.iloc[0].get('cv_val_r2_mean', 0)
                                result_entry['cv_val_r2_std'] = cv_results.iloc[0].get('cv_val_r2_std', 0)
                                result_entry['cv_test_r2_mean'] = cv_results.iloc[0].get('cv_test_r2_mean', r2)
                                result_entry['cv_test_r2_std'] = cv_results.iloc[0].get('cv_test_r2_std', 0)
                            
                            scatter_results.append(result_entry)
                
                # Plot diagonal line
                if all_y_true:
                    min_val = min(min(all_y_true), min(all_y_pred))
                    max_val = max(max(all_y_true), max(all_y_pred))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.9, linewidth=3)
                
                # Set labels and title with larger fonts
                ax.set_xlabel('Actual Values', fontsize=14, fontweight='bold')
                ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
                ax.set_title(f'{dataset_names[dataset]} Dataset', fontsize=16, fontweight='bold')
                ax.tick_params(axis='both', labelsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add legend with larger font
                if all_y_true:
                    ax.legend(loc='upper left', fontsize=10, frameon=True, 
                             fancybox=True, shadow=True)
            
            # Main title with larger font
            plt.suptitle(f'Actual vs Predicted Values - Split: {split_type}, Fingerprint: {fp_type}',
                        fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(split_dir, f'scatter_{split_type}_{fp_type}.png')
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            
            # Create individual plots for each model
            for model in models:
                model_lower = model.lower()
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                datasets_to_plot = ['ws', 'de', 'lo', 'hu']
                for idx, (dataset, ax) in enumerate(zip(datasets_to_plot, axes)):
                    dataset_name = dataset_names[dataset]
                    
                    # Find CV predictions for this model
                    # Check for different key variations
                    if model == 'DNN':
                        # For DNN, try type1 first, then type2
                        dnn_key1 = f"DNN_cv_type1"
                        dnn_key2 = f"DNN_cv_type2"
                        if dnn_key1 in all_predictions[split_type][dataset][fp_type]:
                            key_to_use = dnn_key1
                        elif dnn_key2 in all_predictions[split_type][dataset][fp_type]:
                            key_to_use = dnn_key2
                        else:
                            key_to_use = None
                    else:
                        # For other models
                        cv1_key = f"{model}_cv_method1"
                        cv_key = f"{model}_cv"

                        if cv1_key in all_predictions[split_type][dataset][fp_type]:
                            key_to_use = cv1_key
                        elif cv_key in all_predictions[split_type][dataset][fp_type]:
                            key_to_use = cv_key
                        else:
                            key_to_use = None

                    if key_to_use:
                        pred_data = all_predictions[split_type][dataset][fp_type][key_to_use]
                        y_true = pred_data['y_true']
                        y_pred = pred_data['y_pred']
                        
                        if len(y_true) > 0:
                            # Calculate metrics
                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)
                            
                            # Get CV results from results_df
                            cv_results = results_df[
                                (results_df['split'] == split_type) &
                                (results_df['dataset'] == dataset) &
                                (results_df['fingerprint'] == fp_type) &
                                (results_df['model'] == model) &
                                (results_df['training'] == 'cv_method1')
                            ]
                            
                            # Get CV statistics if available
                            cv_info = ""
                            if not cv_results.empty:
                                cv_test_mean = cv_results.iloc[0].get('cv_test_r2_mean', r2)
                                cv_test_std = cv_results.iloc[0].get('cv_test_r2_std', 0.0)
                                cv_info = f"\nCV: {cv_test_mean:.3f}±{cv_test_std:.3f}"
                            
                            # Plot scatter with star markers
                            ax.scatter(y_true, y_pred, alpha=0.9, s=100, marker='*',
                                     color=vibrant_colors[model], 
                                     edgecolors='black', linewidth=0.5)
                            
                            # Plot diagonal line
                            min_val = min(min(y_true), min(y_pred))
                            max_val = max(max(y_true), max(y_pred))
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.9, linewidth=2)
                            
                            # Set labels and title
                            ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
                            ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
                            ax.set_title(f'{dataset_name}\nR²={r2:.3f}{cv_info}', fontsize=12, fontweight='bold')
                            ax.tick_params(axis='both', labelsize=10)
                            ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                               ha='center', va='center', fontsize=12)
                        ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
                
                # Main title
                plt.suptitle(f'{model} Model - Split: {split_type}, Fingerprint: {fp_type}',
                            fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save individual model plot
                model_plot_filename = os.path.join(split_dir, f'scatter_{split_type}_{fp_type}_{model_lower}.png')
                plt.savefig(model_plot_filename, bbox_inches='tight')
                plt.close()
            
            # Save CSV results for this combination
            if scatter_results:
                csv_filename = os.path.join(split_dir, f'metrics_{split_type}_{fp_type}.csv')
                scatter_df = pd.DataFrame(scatter_results)
                scatter_df.to_csv(csv_filename, index=False)
    
    print("  Prediction scatter plots completed!")

def plot_cv_comparison_scatter(all_predictions, results_df, output_dir):
    """Create scatter plots comparing CV methods"""
    
    print("  Creating CV comparison scatter plots...")
    
    # Process each split type
    for split_type in all_predictions.keys():
        split_dir = os.path.join(output_dir, f'split_{split_type}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Create comparison plot for each dataset
        for dataset in ['ws', 'de', 'lo', 'hu']:
            if dataset not in all_predictions[split_type]:
                continue
                
            # Create figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            dataset_names = {
                'ws': 'WS496',
                'de': 'Delaney',
                'lo': 'Lovric2020',
                'hu': 'Huuskonen'
            }
            
            models = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
            
            # Vibrant colors for each model
            model_colors_vibrant = {
                'Ridge': '#FF1493',      # Deep Pink
                'SVR': '#00CED1',        # Dark Turquoise  
                'RandomForest': '#32CD32', # Lime Green
                'XGBoost': '#FF4500',    # Orange Red
                'LightGBM': '#9370DB',   # Medium Purple
                'DNN': '#FFD700'         # Gold
            }
            
            for model_idx, model in enumerate(models):
                ax = axes[model_idx]
                
                # Get available fingerprints from data
                available_fps = list(all_predictions[split_type][dataset].keys())
                
                # Collect data for all fingerprints
                for fp_idx, fp_type in enumerate(available_fps):
                    if fp_type not in all_predictions[split_type][dataset]:
                        continue
                    
                    # Use different shades for multiple fingerprints
                    fp_colors = {
                        'morgan': model_colors_vibrant[model],
                        'maccs': model_colors_vibrant[model] + '80',  # Semi-transparent
                        'avalon': model_colors_vibrant[model] + '60',  # More transparent
                        'all': model_colors_vibrant[model]
                    }
                    
                    # Plot simple training
                    simple_key = f"{model}_simple"
                    if simple_key in all_predictions[split_type][dataset][fp_type]:
                        pred_data = all_predictions[split_type][dataset][fp_type][simple_key]
                        y_true = pred_data['y_true']
                        y_pred = pred_data['y_pred']
                        
                        if len(y_true) > 0:
                            r2 = r2_score(y_true, y_pred)
                            
                            # Get CV results
                            cv_results = results_df[
                                (results_df['split'] == split_type) &
                                (results_df['dataset'] == dataset) &
                                (results_df['fingerprint'] == fp_type) &
                                (results_df['model'] == model) &
                                (results_df['training'] == 'cv_method1')
                            ]
                            
                            cv_info = ""
                            if not cv_results.empty:
                                cv_val_mean = cv_results.iloc[0].get('cv_val_r2_mean', 0)
                                cv_val_std = cv_results.iloc[0].get('cv_val_r2_std', 0)
                                cv_info = f"\nCV: {cv_val_mean:.3f}±{cv_val_std:.3f}"
                            
                            # Use vibrant colors and larger markers
                            color = fp_colors.get(fp_type, model_colors_vibrant[model])
                            ax.scatter(y_true, y_pred, alpha=0.9, s=80,
                                     color=color, marker='*',
                                     label=f'{fp_type} (R²={r2:.3f}{cv_info})',
                                     edgecolors='black', linewidth=0.5)
                
                # Plot diagonal
                if ax.collections:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                    ax.plot(lim, lim, 'r--', alpha=0.9, linewidth=2)
                
                ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
                ax.set_title(f'{model}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True, shadow=True)
                ax.tick_params(axis='both', labelsize=10)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{dataset_names[dataset]} Dataset - Split: {split_type} - All Models Comparison',
                        fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            # Save plot only if there's data
            if any(ax.collections for ax in axes.flat):
                plot_filename = os.path.join(split_dir, f'model_comparison_{split_type}_{dataset}.png')
                plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
    
    print("  CV comparison plots completed!")

def plot_enhanced_results_comparison(results_df, all_predictions, output_dir):
    """Create comprehensive visualizations for results comparison"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # Filter valid results
    valid_results = results_df[results_df['r2'].notna()].copy()
    
    if valid_results.empty:
        print("No valid results to plot")
        return
    
    # 1. R² Score Heatmap by Model, Dataset, and Fingerprint
    print("  Creating R² heatmap...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get actual datasets from results
    datasets = valid_results['dataset'].unique()
    
    # Use config display names or uppercase
    from config import get_dataset_display_name
    dataset_names = {d: get_dataset_display_name(d) for d in datasets}
    
    for idx, dataset in enumerate(datasets):
        dataset_results = valid_results[
            (valid_results['dataset'] == dataset) & 
            (valid_results['training'] == 'cv_method1')
        ]
        
        if not dataset_results.empty:
            pivot_data = dataset_results.pivot_table(
                index='model',
                columns='fingerprint',
                values='r2',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='RdYlGn',
                       vmin=-0.5,
                       vmax=1.0,
                       ax=axes[idx],
                       cbar_kws={'label': 'R² Score'})
            
            axes[idx].set_title(f'{dataset_names[dataset]} Dataset - R² Performance', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Fingerprint Type', fontsize=10)
            axes[idx].set_ylabel('Model Type', fontsize=10)
    
    plt.suptitle('R² Performance Heatmap by Model, Dataset, and Fingerprint', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_heatmap_by_dataset.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Model Performance Bar Chart
    print("  Creating model performance bar chart...")
    plt.figure(figsize=(14, 8))
    
    # Average R² by model and fingerprint
    model_perf = valid_results[valid_results['training'] == 'cv_method1'].groupby(['model', 'fingerprint'])['r2'].agg(['mean', 'std']).reset_index()
    
    # Create grouped bar chart
    models = model_perf['model'].unique()
    fingerprints = model_perf['fingerprint'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    for i, fp in enumerate(fingerprints):
        fp_data = model_perf[model_perf['fingerprint'] == fp]
        means = [fp_data[fp_data['model'] == m]['mean'].values[0] if len(fp_data[fp_data['model'] == m]) > 0 else 0 for m in models]
        stds = [fp_data[fp_data['model'] == m]['std'].values[0] if len(fp_data[fp_data['model'] == m]) > 0 else 0 for m in models]
        
        plt.bar(x + i*width, means, width, yerr=stds, capsize=5, label=fp, alpha=0.8)
    
    plt.xlabel('Model Type', fontsize=12, fontweight='bold')
    plt.ylabel('Average R² Score', fontsize=12, fontweight='bold')
    plt.title('Average R² Score by Model Type and Fingerprint', fontsize=14, fontweight='bold')
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend(title='Fingerprint', loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_bar_chart.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Split Type Performance Comparison (Box Plot)
    print("  Creating split type performance box plot...")
    plt.figure(figsize=(16, 8))
    
    # Filter for CV method 1 training only
    split_results = valid_results[valid_results['training'] == 'cv_method1']
    
    # Create box plot
    split_order = ['rm', 'ac', 'cl', 'cs', 'en', 'pc', 'sa', 'sc', 'ti']
    split_names = {
        'rm': 'Random',
        'ac': 'Activity Cliff',
        'cl': 'Cluster',
        'cs': 'Chemical Space',
        'en': 'Ensemble',
        'pc': 'Physchem',
        'sa': 'Solubility Aware',
        'sc': 'Scaffold',
        'ti': 'Time'
    }
    
    # Rename splits for display
    split_results['split_display'] = split_results['split'].map(split_names)
    
    sns.boxplot(data=split_results, x='split', y='r2', order=split_order, palette='Set3')
    plt.xlabel('Split Type', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('R² Distribution across Split Types', fontsize=14, fontweight='bold')
    plt.xticks(range(len(split_order)), [split_names.get(s, s) for s in split_order], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_type_performance_boxplot.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Training Method Comparison
    print("  Creating training method comparison...")
    plt.figure(figsize=(14, 8))
    
    # Compare training methods
    training_methods = ['cv_method1', 'cv_method2']
    method_names = {
        'cv_method1': 'CV Method 1 (Test during CV)',
        'cv_method2': 'CV Method 2 (Retrain on full)'
    }
    
    # Average R² by dataset and training method
    for method in training_methods:
        method_results = valid_results[valid_results['training'] == method].groupby('dataset')['r2'].agg(['mean', 'std']).reset_index()
        
        x_pos = range(len(method_results))
        plt.errorbar(x_pos, method_results['mean'], yerr=method_results['std'], 
                    marker='o', markersize=8, capsize=5, capthick=2,
                    label=method_names[method], linewidth=2, alpha=0.8)
    
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Average R² Score', fontsize=12, fontweight='bold')
    plt.title('CV Methods Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(datasets)), [dataset_names[d] for d in datasets])
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_method_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # 5. Best Model Summary Table
    print("  Creating best model summary...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best model-fingerprint combination for each dataset
    best_models = []
    for dataset in datasets:
        dataset_results = valid_results[
            (valid_results['dataset'] == dataset) & 
            (valid_results['training'] == 'cv_method1')
        ]
        
        if not dataset_results.empty:
            best_row = dataset_results.loc[dataset_results['r2'].idxmax()]
            best_models.append({
                'Dataset': dataset_names[dataset],
                'Best Model': best_row['model'],
                'Best Fingerprint': best_row['fingerprint'],
                'R² Score': f"{best_row['r2']:.3f}",
                'Split Type': best_row['split']
            })
    
    if best_models:
        best_df = pd.DataFrame(best_models)
        
        # Create table
        table_data = []
        for _, row in best_df.iterrows():
            table_data.append([row['Dataset'], row['Best Model'], row['Best Fingerprint'], row['R² Score'], row['Split Type']])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Dataset', 'Best Model', 'Best Fingerprint', 'R² Score', 'Split Type'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.axis('off')
        plt.title('Best Model-Fingerprint Combination per Dataset', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_model_summary_table.png'), bbox_inches='tight')
        plt.close()
    
    print("  Visualizations completed!")

def plot_enhanced_results_comparison(results_df, all_predictions, output_dir):
    """Create comprehensive visualizations for results comparison"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # Filter valid results
    valid_results = results_df[results_df['r2'].notna()].copy()
    
    if valid_results.empty:
        print("No valid results to plot")
        return
    
    # 1. R² Score Heatmap by Model, Dataset, and Fingerprint
    print("  Creating R² heatmap...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get actual datasets from results
    datasets = valid_results['dataset'].unique()
    
    # Use config display names or uppercase
    from config import get_dataset_display_name
    dataset_names = {d: get_dataset_display_name(d) for d in datasets}
    
    for idx, dataset in enumerate(datasets):
        dataset_results = valid_results[
            (valid_results['dataset'] == dataset) & 
            (valid_results['training'] == 'cv_method1')
        ]
        
        if not dataset_results.empty:
            pivot_data = dataset_results.pivot_table(
                index='model',
                columns='fingerprint',
                values='r2',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='RdYlGn',
                       vmin=-0.5,
                       vmax=1.0,
                       ax=axes[idx],
                       cbar_kws={'label': 'R² Score'})
            
            axes[idx].set_title(f'{dataset_names[dataset]} Dataset - R² Performance', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Fingerprint Type', fontsize=10)
            axes[idx].set_ylabel('Model Type', fontsize=10)
    
    plt.suptitle('R² Performance Heatmap by Model, Dataset, and Fingerprint', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_heatmap_by_dataset.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Model Performance Bar Chart
    print("  Creating model performance bar chart...")
    plt.figure(figsize=(14, 8))
    
    # Average R² by model and fingerprint
    model_perf = valid_results[valid_results['training'] == 'cv_method1'].groupby(['model', 'fingerprint'])['r2'].agg(['mean', 'std']).reset_index()
    
    # Create grouped bar chart
    models = model_perf['model'].unique()
    fingerprints = model_perf['fingerprint'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    for i, fp in enumerate(fingerprints):
        fp_data = model_perf[model_perf['fingerprint'] == fp]
        means = [fp_data[fp_data['model'] == m]['mean'].values[0] if len(fp_data[fp_data['model'] == m]) > 0 else 0 for m in models]
        stds = [fp_data[fp_data['model'] == m]['std'].values[0] if len(fp_data[fp_data['model'] == m]) > 0 else 0 for m in models]
        
        plt.bar(x + i*width, means, width, yerr=stds, capsize=5, label=fp, alpha=0.8)
    
    plt.xlabel('Model Type', fontsize=12, fontweight='bold')
    plt.ylabel('Average R² Score', fontsize=12, fontweight='bold')
    plt.title('Average R² Score by Model Type and Fingerprint', fontsize=14, fontweight='bold')
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend(title='Fingerprint', loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_bar_chart.png'), bbox_inches='tight')
    plt.close()
    
    # 3. Split Type Performance Comparison (Box Plot)
    print("  Creating split type performance box plot...")
    plt.figure(figsize=(16, 8))
    
    # Filter for CV method 1 training only
    split_results = valid_results[valid_results['training'] == 'cv_method1']
    
    # Create box plot
    split_order = ['rm', 'ac', 'cl', 'cs', 'en', 'pc', 'sa', 'sc', 'ti']
    split_names = {
        'rm': 'Random',
        'ac': 'Activity Cliff',
        'cl': 'Cluster',
        'cs': 'Chemical Space',
        'en': 'Ensemble',
        'pc': 'Physchem',
        'sa': 'Solubility Aware',
        'sc': 'Scaffold',
        'ti': 'Time'
    }
    
    # Rename splits for display
    split_results['split_display'] = split_results['split'].map(split_names)
    
    sns.boxplot(data=split_results, x='split', y='r2', order=split_order, palette='Set3')
    plt.xlabel('Split Type', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('R² Distribution across Split Types', fontsize=14, fontweight='bold')
    plt.xticks(range(len(split_order)), [split_names.get(s, s) for s in split_order], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'split_type_performance_boxplot.png'), bbox_inches='tight')
    plt.close()
    
    # 4. Training Method Comparison
    print("  Creating training method comparison...")
    plt.figure(figsize=(14, 8))
    
    # Compare training methods
    training_methods = ['cv_method1', 'cv_method2']
    method_names = {
        'cv_method1': 'CV Method 1 (Test during CV)',
        'cv_method2': 'CV Method 2 (Retrain on full)'
    }
    
    # Average R² by dataset and training method
    for method in training_methods:
        method_results = valid_results[valid_results['training'] == method].groupby('dataset')['r2'].agg(['mean', 'std']).reset_index()
        
        x_pos = range(len(method_results))
        plt.errorbar(x_pos, method_results['mean'], yerr=method_results['std'], 
                    marker='o', markersize=8, capsize=5, capthick=2,
                    label=method_names[method], linewidth=2, alpha=0.8)
    
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Average R² Score', fontsize=12, fontweight='bold')
    plt.title('CV Methods Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(datasets)), [dataset_names[d] for d in datasets])
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_method_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # 5. Best Model Summary Table
    print("  Creating best model summary...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best model-fingerprint combination for each dataset
    best_models = []
    for dataset in datasets:
        dataset_results = valid_results[
            (valid_results['dataset'] == dataset) & 
            (valid_results['training'] == 'cv_method1')
        ]
        
        if not dataset_results.empty:
            best_row = dataset_results.loc[dataset_results['r2'].idxmax()]
            best_models.append({
                'Dataset': dataset_names[dataset],
                'Best Model': best_row['model'],
                'Best Fingerprint': best_row['fingerprint'],
                'R² Score': f"{best_row['r2']:.3f}",
                'Split Type': best_row['split']
            })
    
    if best_models:
        best_df = pd.DataFrame(best_models)
        
        # Create table
        table_data = []
        for _, row in best_df.iterrows():
            table_data.append([row['Dataset'], row['Best Model'], row['Best Fingerprint'], row['R² Score'], row['Split Type']])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Dataset', 'Best Model', 'Best Fingerprint', 'R² Score', 'Split Type'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.axis('off')
        plt.title('Best Model-Fingerprint Combination per Dataset', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'best_model_summary_table.png'), bbox_inches='tight')
        plt.close()
    
    print("  Visualizations completed!")

def plot_original_datasets_distribution(output_dir):
    """Plot the distribution of original datasets"""
    
    print("  Creating dataset distribution plots...")
    # Skip creating empty placeholder image
    pass

def cleanup_temp_files():
    """Clean up temporary directory"""
    # Clean up temporary directory only
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DIR}")

def metric_prediction(y_true, y_pred):
    """Calculate regression metrics with sanitized predictions"""
    y_pred = sanitize_predictions(y_pred)  # Sanitize before metric calculation
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }

def train_dnn_subprocess(X_train, y_train, X_test, y_test, epochs=None, batch_size=32, lr=0.001, fold_id=0):
    """
    DNN model training function using subprocess

    Purpose:
    - Prevent memory leaks through memory isolation
    - Avoid OpenMP conflicts
    - Execute each model training in independent process

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature vectors (n_samples, n_features)
    y_train : np.ndarray
        Training target values (n_samples,)
    X_test : np.ndarray
        Test feature vectors (n_test_samples, n_features)
    y_test : np.ndarray
        Test target values (n_test_samples,)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate

    Returns:
    --------
    y_pred : np.ndarray
        Predictions
    metrics : dict
        Performance metrics (r2, rmse, mae, mse)
    """
    # Handle None epochs - use global EPOCHS or get from config
    if epochs is None:
        if EPOCHS is not None:
            epochs = EPOCHS
        else:
            epochs = get_epochs_for_module('2')

    try:
        # Save data to temporary files with fold-specific names
        os.makedirs('save_model', exist_ok=True)  # create temp directory
        temp_X_train = f"save_model/temp_X_train_{fold_id}.npy"
        temp_y_train = f"save_model/temp_y_train_{fold_id}.npy"
        temp_X_test = f"save_model/temp_X_test_{fold_id}.npy"
        temp_y_test = f"save_model/temp_y_test_{fold_id}.npy"
        temp_model = f"save_model/full_model_{fold_id}.pt"

        np.save(temp_X_train, X_train)
        np.save(temp_y_train, y_train)
        np.save(temp_X_test, X_test)
        np.save(temp_y_test, y_test)
        
        # Call subprocess (OS-optimized)
        cmd = [
            sys.executable,
            "extra_code/learning_process_pytorch_torchscript.py",
            str(batch_size),
            str(epochs),
            str(lr),
            temp_X_train, temp_y_train,
            temp_X_test, temp_y_test,
            temp_model
        ]
        
        # OS-specific environment variable settings
        env = os.environ.copy()
        if OS_TYPE == "Windows":
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            env['MKL_NUM_THREADS'] = '1'
            env['OMP_NUM_THREADS'] = '1'
        elif OS_TYPE == "Darwin":  # macOS
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            env['OMP_NUM_THREADS'] = '1'
        else:  # Linux
            env['OMP_NUM_THREADS'] = str(min(4, os.cpu_count()))
            env['MKL_NUM_THREADS'] = str(min(4, os.cpu_count()))
        
        # OS-specific subprocess settings
        try:
            if OS_TYPE == "Windows":
                # Windows: create new process group with creationflags
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
            print(f"      DNN Subprocess Timeout: Process took longer than 600 seconds")
            return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}
        
        if result.returncode != 0:
            print(f"      DNN Subprocess Error: {result.stderr}")
            return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}
        
        # Parse output and load predictions from subprocess
        pred_file = temp_model.replace('.pt', '_pred.npy').replace('.pth', '_pred.npy')

        # First check if subprocess saved predictions
        if os.path.exists(pred_file):
            # Load predictions saved by subprocess
            y_pred = np.load(pred_file)
            print(f"      Loaded predictions from subprocess")

            # Parse metrics from subprocess output
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

                            metrics = {
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'mse': mse
                            }
                            print(f"      DNN Subprocess Results (from output) - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                            break
                        except ValueError:
                            continue
            else:
                # Calculate metrics from predictions if not found in output
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                metrics = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'mse': mse
                }
                print(f"      DNN Subprocess Results (calculated) - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        else:
            # No prediction file - use mean as fallback
            print(f"      Warning: No predictions file found from subprocess")
            y_pred = np.full_like(y_test, np.mean(y_train))

            # Try to parse metrics from output
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

                            metrics = {
                                'r2': r2,
                                'rmse': rmse,
                                'mae': mae,
                                'mse': mse
                            }
                            print(f"      DNN Subprocess Results (from output, no pred file) - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                            break
                        except ValueError:
                            continue
            else:
                print("      DNN Subprocess: No valid metrics found")
                metrics = {'r2': -1.0, 'rmse': float('inf'), 'mae': float('inf'), 'mse': float('inf')}
        
        # Clean up temporary files with unique names (including .pth version)
        files_to_clean = [temp_X_train, temp_y_train, temp_X_test, temp_y_test, temp_model]
        # Also clean .pth version if it exists
        if temp_model.endswith('.pt'):
            files_to_clean.append(temp_model.replace('.pt', '.pth'))
        # Also clean prediction file
        if pred_file and os.path.exists(pred_file):
            files_to_clean.append(pred_file)

        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        y_pred = sanitize_predictions(y_pred); return y_pred, metrics

    except Exception as e:
        print(f"Error in DNN subprocess training: {e}")
        return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

def cleanup_temp_files():
    """Clean up temporary DNN files to prevent conflicts between TYPE1 and TYPE2"""
    import glob
    temp_files = glob.glob("save_model/temp_*.npy")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass
    print(f"  🧹 Cleaned up {len(temp_files)} temp files for memory optimization")

def train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test, epochs=None, batch_size=32, lr=0.001, fold_id=0):
    """
    Alias for train_dnn_subprocess for compatibility with dual CV system
    """
    if epochs is None:
        epochs = MODEL_CONFIG['epochs']
    return train_dnn_subprocess(X_train, y_train, X_test, y_test, epochs, batch_size, lr, fold_id=fold_id)

def train_model(model_type, X_train, y_train, X_test, y_test):
    """
    Train traditional ML models (non-DNN)

    Parameters:
    -----------
    model_type : str
        Model type ('Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM')
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets

    Returns:
    --------
    y_pred : np.ndarray
        Predictions on test set
    metrics : dict
        Performance metrics
    """
    try:
        # Ensure arrays are numpy arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).flatten()
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test).flatten()

        # Initialize and train model based on type
        if model_type == 'Ridge':
            model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        elif model_type == 'SVR':
            model = SVR(C=1.0, gamma='scale', kernel='rbf')
        elif model_type == 'RandomForest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=1  # Single thread to avoid conflicts
            )
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=1,  # Single thread
                verbosity=0  # Suppress output
            )
        elif model_type == 'LightGBM':
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=1,  # Single thread
                verbosity=-1  # Suppress output
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Sanitize predictions first to avoid NaN/Inf in metrics
        y_pred = sanitize_predictions(y_pred)

        # Calculate metrics with sanitized predictions
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse
        }

        return y_pred, metrics

    except Exception as e:
        print(f"      Error training {model_type}: {e}")
        return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

def train_dnn_pytorch_optimized(X_train, y_train, X_test, y_test, epochs=None, batch_size=32, lr=0.001, device=None):
    """
    Optimized PyTorch DNN model training function (fallback method)

    Purpose:
    - Train directly with PyTorch when subprocess fails
    - Apply OS-specific optimizations
    - Improve performance with early stopping and learning rate scheduling

    Optimization techniques:
    1. Use ReLU activation function
    2. Adam optimizer
    3. ReduceLROnPlateau scheduler
    4. Early stopping (patience=20)
    5. Gradient clipping
    6. Batch normalization
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    epochs : Number of training epochs
    batch_size : Batch size
    lr : Initial learning rate
    device : PyTorch device (auto-selected if None)

    Returns:
    --------
    y_pred : Predictions
    metrics : Performance metrics
    """
    # Handle None epochs - use global EPOCHS or get from config
    if epochs is None:
        if EPOCHS is not None:
            epochs = EPOCHS
        else:
            epochs = get_epochs_for_module('2')

    try:
        # Use global device if not specified
        if device is None:
            device = globals().get('device', torch.device('cpu'))
        
        # Safe mode if parallelization fails
        if isinstance(device, str):
            device = torch.device(device)
            
        # Convert to PyTorch tensors with error handling
        try:
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).reshape(-1, 1)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        except Exception as e:
            print(f"      Warning: Tensor conversion error: {e}")
            # Fallback to numpy arrays
            X_train = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
            y_train = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1)
            X_test = torch.from_numpy(np.asarray(X_test, dtype=np.float32))
            y_test = torch.from_numpy(np.asarray(y_test, dtype=np.float32)).reshape(-1, 1)
        
        # Debug info
        print(f"      DNN Debug - X_train shape: {X_train.shape}, y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"      DNN Debug - X_test shape: {X_test.shape}, y_test range: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # Create model
        input_dim = X_train.shape[1]
        model = DNNModel(input_dim).to(device)
        
        # Model will be saved after training in eval mode
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
        
        # Create data loaders with OS-specific settings
        train_dataset = TensorDataset(X_train, y_train)
        
        # DataLoader worker settings (OS-specific optimization)
        if OS_TYPE == "Windows":
            # Windows: disable worker processes (prevent OpenMP conflicts)
            num_workers = 0
        elif OS_TYPE == "Darwin":  # macOS
            # macOS: use limited workers
            num_workers = min(2, os.cpu_count() // 2)
        else:  # Linux
            # Linux: use aggressive workers
            num_workers = min(4, os.cpu_count() // 2)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and device.type == 'cuda'
        )
        
        # Training loop with early stopping
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"      Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model using TorchScript
        model.eval()  # Important: switch to eval mode
        os.makedirs("save_model", exist_ok=True)
        
        # Save as TorchScript
        model_cpu = model.cpu()
        model_cpu.eval()  # Ensure eval mode for BatchNorm
        
        # For models with track_running_stats=False, we need to use scripting instead of tracing
        # Or use a batch_size that matches training (32) for more accurate tracing
        input_dim = X_train.shape[1]
        # Use actual batch size from training for accurate model behavior
        custom_input = torch.randn(32, input_dim)  # Use training batch_size for proper BatchNorm
        try:
            traced_model = torch.jit.trace(model_cpu, custom_input, check_trace=False)
            traced_model.save("save_model/full_model.pt")
            print("      Saved TorchScript model to save_model/full_model.pt")
        except Exception as e:
            print(f"      Warning: Could not save TorchScript model: {e}")
            # Alternative: save as regular PyTorch model
            torch.save(model_cpu.state_dict(), "save_model/full_model.pth")
            print("      Saved as regular PyTorch model instead")
        
        # Move model back to original device for evaluation
        model = model.to(device)
        
        # Evaluation
        with torch.inference_mode():
            X_test_device = X_test.to(device)
            y_pred = model(X_test_device).cpu().numpy()
            y_test_np = y_test.numpy()
            
            # Debug predictions
            print(f"      DNN Debug - y_pred range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
            print(f"      DNN Debug - y_test range: [{y_test_np.min():.3f}, {y_test_np.max():.3f}]")
            
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
            
            print(f"      DNN Results - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return y_pred.flatten(), metrics
        
    except Exception as e:
        print(f"Error in PyTorch DNN training: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros_like(y_test), {'r2': 0.0, 'rmse': float('inf'), 'mae': 0.0, 'mse': float('inf')}

def train_model_simple(model_type, X_train, y_train, X_test, y_test):
    """Train model with simple train/test split - DNN uses subprocess"""

    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Clean data: remove NaN/Inf values
    print(f"  [CLEAN] Original data: {len(X_train)} train, {len(X_test)} test samples")
    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)
    print(f"  [CLEAN] Cleaned data: {len(X_train)} train, {len(X_test)} test samples")
    
    # Debug info for all models
    print(f"        Debug - {model_type} input shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"        Debug - {model_type} target ranges: y_train=[{y_train.min():.3f}, {y_train.max():.3f}], y_test=[{y_test.min():.3f}, {y_test.max():.3f}]")
    
    if model_type == 'DNN':
        # Try subprocess implementation first for memory isolation
        try:
            y_pred, metrics = train_dnn_subprocess(
                X_train, y_train, X_test, y_test,
                epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr, fold_id=0
            )
            # Check if subprocess failed
            if metrics['r2'] == 0.0 and metrics['rmse'] == float('inf'):
                print("      Subprocess failed, trying direct PyTorch training...")
                y_pred, metrics = train_dnn_pytorch_optimized(
                    X_train, y_train, X_test, y_test,
                    epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr
                )
        except Exception as e:
            print(f"      Subprocess error: {e}, falling back to direct PyTorch training...")
            try:
                y_pred, metrics = train_dnn_pytorch_optimized(
                    X_train, y_train, X_test, y_test,
                    epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr
                )
            except Exception as e2:
                print(f"      Direct PyTorch training also failed: {e2}")
                y_pred = np.zeros_like(y_test)
        
        y_pred = sanitize_predictions(y_pred); return y_pred, None
    else:
        # OS-specific parallelization settings
        if OS_TYPE == "Windows":
            # Windows: Use fewer threads to avoid conflicts
            n_jobs = min(2, os.cpu_count())
        else:
            # Linux/Mac: Use more threads
            n_jobs = min(4, os.cpu_count())
        
        # Traditional ML models with OS-optimized settings
        models = {
            'Ridge': Ridge(alpha=1.0),  # specify alpha
            'SVR': SVR(),
            'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs),
            'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(
                random_state=RANDOM_STATE, 
                n_jobs=n_jobs, 
                verbose=-1,
                enable_categorical=False,
                force_row_wise=True
            )
        }        
        model = models[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Debug predictions
        print(f"        Debug - {model_type} predictions range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
        
        y_pred = sanitize_predictions(y_pred); return y_pred, None

def train_model_cv_type1(model_type, X_data, y_data, n_folds=5):
    """
    Type 1: Research Pipeline - CV methodology
    Splits data into K-folds and uses each fold as test set once

    Args:
        model_type: Type of model to train
        X_data: Feature matrix
        y_data: Target values
        n_folds: Number of CV folds

    Returns:
        Dictionary with CV statistics
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds}")

    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    # Clean data
    X_data, y_data = clean_data(X_data, y_data)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mse_scores = []
    cv_mae_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        if model_type == 'DNN':
            y_pred, metrics = train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test)
        else:
            y_pred, metrics = train_model(model_type, X_train, y_train, X_test, y_test)

        if metrics:
            cv_r2_scores.append(metrics['r2'])
            cv_rmse_scores.append(metrics['rmse'])
            cv_mse_scores.append(metrics['mse'])
            cv_mae_scores.append(metrics['mae'])
            print(f"      [TYPE1] Fold {fold+1}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Calculate statistics
    if len(cv_r2_scores) == 0:
        return {'cv_stats': {'mean_r2': 0, 'std_r2': 0, 'mean_rmse': 0, 'std_rmse': 0, 'mean_mse': 0, 'std_mse': 0, 'mean_mae': 0, 'std_mae': 0}}

    mean_r2 = np.mean(cv_r2_scores)
    std_r2 = np.std(cv_r2_scores)
    mean_rmse = np.mean(cv_rmse_scores)
    std_rmse = np.std(cv_rmse_scores)
    mean_mse = np.mean(cv_mse_scores)
    std_mse = np.std(cv_mse_scores)
    mean_mae = np.mean(cv_mae_scores)
    std_mae = np.std(cv_mae_scores)

    best_fold_idx = np.argmax(cv_r2_scores)
    best_r2 = cv_r2_scores[best_fold_idx]
    best_rmse = cv_rmse_scores[best_fold_idx]
    best_mse = cv_mse_scores[best_fold_idx]
    best_mae = cv_mae_scores[best_fold_idx]

    print(f"    [TYPE1-Research] CV Results: R2={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}, MSE={mean_mse:.4f}±{std_mse:.4f}, MAE={mean_mae:.4f}±{std_mae:.4f}")

    return {
        'cv_stats': {
            'mean_r2': mean_r2, 'std_r2': std_r2, 'best_r2': best_r2,
            'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 'best_rmse': best_rmse,
            'mean_mse': mean_mse, 'std_mse': std_mse, 'best_mse': best_mse,
            'mean_mae': mean_mae, 'std_mae': std_mae, 'best_mae': best_mae
        }
    }

def train_model_cv_type2(X_train, y_train, X_test, y_test, model_type, n_folds=5):
    """
    Type 2: Production Pipeline - Train/Test Split + CV
    Performs CV on pre-split training data and evaluates on independent test set

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_type: Type of model to train
        n_folds: Number of CV folds

    Returns:
        Tuple of (cv_stats, final_test_metrics)
    """
    print(f"    [TYPE2-Production] Production Pipeline - CV-{n_folds} on Train + Final Test")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # Clean data
    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)

    # Step 1: CV on training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = []
    cv_rmse_scores = []
    cv_mse_scores = []
    cv_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        if model_type == 'DNN':
            y_pred, metrics = train_pytorch_dnn_subprocess(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold_id=fold)
        else:
            y_pred, metrics = train_model(model_type, X_train_fold, y_train_fold, X_val_fold, y_val_fold)

        if metrics:
            cv_r2_scores.append(metrics['r2'])
            cv_rmse_scores.append(metrics['rmse'])
            cv_mse_scores.append(metrics['mse'])
            cv_mae_scores.append(metrics['mae'])
            print(f"      [TYPE2] CV Fold {fold+1}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Calculate CV statistics
    mean_r2 = np.mean(cv_r2_scores) if cv_r2_scores else 0
    std_r2 = np.std(cv_r2_scores) if cv_r2_scores else 0
    mean_rmse = np.mean(cv_rmse_scores) if cv_rmse_scores else 0
    std_rmse = np.std(cv_rmse_scores) if cv_rmse_scores else 0
    mean_mse = np.mean(cv_mse_scores) if cv_mse_scores else 0
    std_mse = np.std(cv_mse_scores) if cv_mse_scores else 0
    mean_mae = np.mean(cv_mae_scores) if cv_mae_scores else 0
    std_mae = np.std(cv_mae_scores) if cv_mae_scores else 0

    cv_stats = {
        'mean_r2': mean_r2, 'std_r2': std_r2,
        'mean_rmse': mean_rmse, 'std_rmse': std_rmse,
        'mean_mse': mean_mse, 'std_mse': std_mse,
        'mean_mae': mean_mae, 'std_mae': std_mae
    }

    print(f"    [TYPE2-Production] CV Results: R2={mean_r2:.4f}±{std_r2:.4f}, RMSE={mean_rmse:.4f}±{std_rmse:.4f}, MSE={mean_mse:.4f}±{std_mse:.4f}, MAE={mean_mae:.4f}±{std_mae:.4f}")

    # Step 2: Train final model on full training data and test on test set
    print(f"    [TYPE2-Production] Training final model on full training data...")

    if model_type == 'DNN':
        y_pred_final, test_metrics = train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test)
    else:
        y_pred_final, test_metrics = train_model(model_type, X_train, y_train, X_test, y_test)

    if test_metrics:
        print(f"    [TYPE2-Production] Final Test: R2={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}")
        final_test_metrics = test_metrics
    else:
        final_test_metrics = {'r2': 0, 'rmse': 0, 'mse': 0, 'mae': 0}

    return cv_stats, final_test_metrics

def train_model_cv_both_types(model_type, X_train, y_train, X_test, y_test, n_folds=5):
    """
    Dual CV methodology (prevents data leakage)
    - Type1: K-fold CV on training data only + test prediction per fold
    - Type2: Train/val split + single independent test prediction

    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_folds: Number of CV folds

    Returns:
        Dictionary with Type1 and Type2 results
    """
    print(f"\n=== Running Dual CV Types for {model_type} ===")

    results = {}

    # Type 1: Research Pipeline - K-fold CV on training data + test prediction per fold
    print(f"\n--- Type 1: Research Pipeline ---")
    type1_results = train_model_cv_type1(model_type, X_train, y_train, X_test, y_test, n_folds)
    results['type1'] = {'cv_stats': type1_results}

    # Type 2: Production Pipeline - Train/Val split + test prediction
    print(f"\n--- Type 2: Production Pipeline ---")
    type2_cv_stats, type2_final_metrics = train_model_cv_type2(model_type, X_train, y_train, X_test, y_test, n_folds)
    results['type2'] = {'cv_stats': type2_cv_stats, 'final_metrics': type2_final_metrics}

    return results

def train_model_cv_type1(model_type, X_train, y_train, X_test, y_test, n_folds=5):
    """
    Type1: Research Pipeline - CV methodology
    Performs K-fold CV using training data only and predicts independent test set per fold

    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_folds: Number of CV folds

    Returns:
        CV statistics dictionary
    """
    print(f"    [TYPE1-Research] Research Pipeline - CV-{n_folds} on Train + Test prediction per fold")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_val_r2_scores = []  # Validation scores (train CV)
    cv_val_rmse_scores = []
    cv_val_mae_scores = []

    test_r2_scores = []  # Test scores (test prediction per fold)
    test_rmse_scores = []
    test_mae_scores = []
    test_predictions = []  # Store actual predictions from each fold

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"      Fold {fold+1}/{n_folds}", end='\r')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train model on this fold and predict test set (Type1 approach)
        if model_type == 'DNN':
            # Test prediction only (Type1: each fold predicts independent test set)
            y_test_pred, test_metrics = train_pytorch_dnn_subprocess(X_tr, y_tr, X_test, y_test)
            if test_metrics:
                test_r2_scores.append(test_metrics['r2'])
                test_rmse_scores.append(test_metrics['rmse'])
                test_mae_scores.append(test_metrics['mae'])
                # Use test metrics for CV validation too
                cv_val_r2_scores.append(test_metrics['r2'])
                cv_val_rmse_scores.append(test_metrics['rmse'])
                cv_val_mae_scores.append(test_metrics['mae'])
                # Store actual predictions
                test_predictions.append(y_test_pred)
        else:
            # Test prediction only (Type1: each fold predicts independent test set)
            y_test_pred, test_metrics = train_model(model_type, X_tr, y_tr, X_test, y_test)
            if test_metrics:
                test_r2_scores.append(test_metrics['r2'])
                test_rmse_scores.append(test_metrics['rmse'])
                test_mae_scores.append(test_metrics['mae'])
                # Use test metrics for CV validation too
                cv_val_r2_scores.append(test_metrics['r2'])
                cv_val_rmse_scores.append(test_metrics['rmse'])
                cv_val_mae_scores.append(test_metrics['mae'])
                # Store actual predictions
                test_predictions.append(y_test_pred)

    # Calculate CV validation statistics
    val_r2_mean = np.mean(cv_val_r2_scores) if cv_val_r2_scores else 0.0
    val_r2_std = np.std(cv_val_r2_scores) if cv_val_r2_scores else 0.0
    val_rmse_mean = np.mean(cv_val_rmse_scores) if cv_val_rmse_scores else 0.0
    val_rmse_std = np.std(cv_val_rmse_scores) if cv_val_rmse_scores else 0.0
    val_mae_mean = np.mean(cv_val_mae_scores) if cv_val_mae_scores else 0.0
    val_mae_std = np.std(cv_val_mae_scores) if cv_val_mae_scores else 0.0

    # Calculate test statistics (average of K test predictions)
    test_r2_mean = np.mean(test_r2_scores) if test_r2_scores else 0.0
    test_r2_std = np.std(test_r2_scores) if test_r2_scores else 0.0
    test_rmse_mean = np.mean(test_rmse_scores) if test_rmse_scores else 0.0
    test_rmse_std = np.std(test_rmse_scores) if test_rmse_scores else 0.0
    test_mae_mean = np.mean(test_mae_scores) if test_mae_scores else 0.0
    test_mae_std = np.std(test_mae_scores) if test_mae_scores else 0.0

    print(f"\n    [TYPE1-Research] CV Val: R²={val_r2_mean:.4f}±{val_r2_std:.4f}, RMSE={val_rmse_mean:.4f}±{val_rmse_std:.4f}, MAE={val_mae_mean:.4f}±{val_mae_std:.4f}")
    print(f"    [TYPE1-Research] Test Avg: R²={test_r2_mean:.4f}±{test_r2_std:.4f}, RMSE={test_rmse_mean:.4f}±{test_rmse_std:.4f}, MAE={test_mae_mean:.4f}±{test_mae_std:.4f}")

    # Calculate MSE statistics
    test_mse_scores = [rmse**2 for rmse in test_rmse_scores]
    test_mse_mean = np.mean(test_mse_scores) if test_mse_scores else 0.0
    test_mse_std = np.std(test_mse_scores) if test_mse_scores else 0.0

    # Calculate average predictions from all folds
    if test_predictions:
        # Average predictions from all folds
        y_pred_avg = np.mean(test_predictions, axis=0)
    else:
        # Fallback: train final model if no predictions available
        print("      Warning: No test predictions, training final model...")
        if model_type == 'DNN':
            y_pred_avg, _ = train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test, fold_id=999)
        else:
            y_pred_avg, _ = train_model(model_type, X_train, y_train, X_test, y_test)

    return {
        'mean_r2': test_r2_mean,  # Use test performance as main metric
        'std_r2': test_r2_std,
        'mean_rmse': test_rmse_mean,
        'std_rmse': test_rmse_std,
        'mean_mae': test_mae_mean,
        'std_mae': test_mae_std,
        'mean_mse': test_mse_mean,
        'std_mse': test_mse_std,
        'val_r2_mean': val_r2_mean,
        'val_r2_std': val_r2_std,
        'test_r2_mean': test_r2_mean,
        'test_r2_std': test_r2_std,
        'test_r2_scores': test_r2_scores,  # Add individual fold scores
        'test_rmse_scores': test_rmse_scores,
        'test_mae_scores': test_mae_scores,
        'cv_val_r2_scores': cv_val_r2_scores,
        'test_predictions': test_predictions,  # Add actual predictions
        'y_pred_avg': y_pred_avg  # Add average predictions
    }

def train_model_cv_type2(model_type, X_train, y_train, X_test, y_test, n_folds=5):
    """
    Type2: Production Pipeline - Train/Test Split + CV
    Performs CV on pre-split training data and evaluates on independent test set

    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_folds: Number of CV folds

    Returns:
        Tuple of (cv_stats, final_test_metrics)
    """
    print(f"    [TYPE2-Production] Production Pipeline - Train/Val split + Final test")

    # Split training data into train/val
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    # K-fold CV on training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_fold_tr, X_fold_val = X_tr[train_idx], X_tr[val_idx]
        y_fold_tr, y_fold_val = y_tr[train_idx], y_tr[val_idx]

        if model_type == 'DNN':
            y_pred, metrics = train_pytorch_dnn_subprocess(X_fold_tr, y_fold_tr, X_fold_val, y_fold_val, fold_id=fold)
        else:
            y_pred, metrics = train_model(model_type, X_fold_tr, y_fold_tr, X_fold_val, y_fold_val)

        if metrics:
            cv_scores.append(metrics['r2'])
            cv_rmse_scores.append(metrics['rmse'])
            cv_mae_scores.append(metrics['mae'])

    cv_stats = {
        'mean_r2': np.mean(cv_scores) if cv_scores else 0.0,
        'std_r2': np.std(cv_scores) if cv_scores else 0.0,
        'mean_rmse': np.mean(cv_rmse_scores) if cv_rmse_scores else 0.0,
        'std_rmse': np.std(cv_rmse_scores) if cv_rmse_scores else 0.0,
        'mean_mae': np.mean(cv_mae_scores) if cv_mae_scores else 0.0,
        'std_mae': np.std(cv_mae_scores) if cv_mae_scores else 0.0
    }

    # Final test prediction (once only)
    if model_type == 'DNN':
        y_test_pred, final_metrics = train_pytorch_dnn_subprocess(X_train, y_train, X_test, y_test)
    else:
        y_test_pred, final_metrics = train_model(model_type, X_train, y_train, X_test, y_test)

    final_test_metrics = {
        'test_r2': final_metrics['r2'] if final_metrics else 0.0,
        'test_rmse': final_metrics['rmse'] if final_metrics else 0.0,
        'test_mae': final_metrics['mae'] if final_metrics else 0.0,
        'y_test_pred': y_test_pred  # Add actual predictions
    }

    print(f"    [TYPE2-Production] CV: R²={cv_stats['mean_r2']:.4f}±{cv_stats['std_r2']:.4f}, RMSE={cv_stats['mean_rmse']:.4f}±{cv_stats['std_rmse']:.4f}, MAE={cv_stats['mean_mae']:.4f}±{cv_stats['std_mae']:.4f}")
    print(f"    [TYPE2-Production] Test: R²={final_test_metrics['test_r2']:.4f}, RMSE={final_test_metrics['test_rmse']:.4f}, MAE={final_test_metrics['test_mae']:.4f}")

    return cv_stats, final_test_metrics

def train_model_cv(model_type, X_train, y_train, X_test, y_test, n_folds=5, cv_type='traditional'):
    """Train model with specified CV type

    Args:
        cv_type: 'traditional' (for traditional ML), 'type1' (research), or 'type2' (production)
    """
    # Ensure arrays are numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Clean data: remove NaN/Inf values
    print(f"  [CLEAN] Original CV data: {len(X_train)} train, {len(X_test)} test samples")

    X_train, y_train = clean_data(X_train, y_train)
    X_test, y_test = clean_data(X_test, y_test)
    print(f"  [CLEAN] Cleaned CV data: {len(X_train)} train, {len(X_test)} test samples")

    # For DNN models, use specified CV type
    if model_type == 'DNN':
        if cv_type == 'type1':
            # Run Type1 (Research) CV only
            type1_results = train_model_cv_type1(model_type, X_train, y_train, X_test, y_test, n_folds)

            # Return results in original format
            y_pred_cv = type1_results.get('y_pred_avg', np.full_like(y_test, np.mean(y_train)))  # Use actual average predictions
            cv_val_scores = type1_results['cv_val_r2_scores']  # Use actual CV scores
            cv_val_preds = type1_results.get('test_predictions', [y_pred_cv] * n_folds)
            test_preds = type1_results.get('test_predictions', [y_pred_cv] * n_folds)
            test_scores = type1_results['test_r2_scores']  # Use actual test scores

            return y_pred_cv, cv_val_scores, cv_val_preds, test_preds, test_scores

        elif cv_type == 'type2':
            # Run Type2 (Production) CV only
            cv_stats, final_metrics = train_model_cv_type2(model_type, X_train, y_train, X_test, y_test, n_folds)

            # Return results in original format
            y_pred_cv = final_metrics.get('y_test_pred', np.full_like(y_test, np.mean(y_train)))  # Use actual predictions from type2
            cv_val_scores = [cv_stats['mean_r2']] * n_folds
            cv_val_preds = [y_pred_cv] * n_folds
            test_preds = [y_pred_cv] * n_folds
            test_scores = [final_metrics['test_r2']] * n_folds

            return y_pred_cv, cv_val_scores, cv_val_preds, test_preds, test_scores

    # Keep existing logic for traditional ML models
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_val_predictions = []  # Validation predictions for each fold
    cv_val_scores = []  # Validation scores for each fold
    test_predictions = []  # Test predictions for each fold (Method 1)
    test_scores = []  # Test scores for each fold (Method 1)

    # Check if we should evaluate test during CV
    evaluate_test_during_cv = True  # Default behavior

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        if model_type == 'DNN':
            # Try subprocess implementation first for memory isolation
            try:
                # Train model for this fold
                val_pred, val_metrics = train_dnn_subprocess(
                    X_tr, y_tr, X_val, y_val,
                    epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr, fold_id=fold
                )
                
                # Store validation results
                cv_val_predictions.append(val_pred)
                cv_val_scores.append(val_metrics['r2'])
                
                # Method 1: Also evaluate on test set during CV
                if evaluate_test_during_cv:
                    test_pred, test_metrics = train_dnn_subprocess(
                        X_tr, y_tr, X_test, y_test,
                        epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr, fold_id=fold
                    )
                    # Check if subprocess failed
                    if test_metrics['r2'] == 0.0 and test_metrics['rmse'] == float('inf'):
                        print(f"      Fold {fold+1} subprocess failed, trying direct PyTorch...")
                        test_pred, _ = train_dnn_pytorch_optimized(
                            X_tr, y_tr, X_test, y_test,
                            epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr
                        )
                    test_predictions.append(test_pred)
                    test_scores.append(r2_score(y_test, test_pred))
                    
            except Exception as e:
                print(f"      Fold {fold+1} error: {e}, using direct PyTorch...")
                try:
                    val_pred, _ = train_dnn_pytorch_optimized(
                        X_tr, y_tr, X_val, y_val,
                        epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr
                    )
                    cv_val_predictions.append(val_pred)
                    cv_val_scores.append(r2_score(y_val, val_pred))
                    
                    if evaluate_test_during_cv:
                        test_pred, _ = train_dnn_pytorch_optimized(
                            X_tr, y_tr, X_test, y_test,
                            epochs=EPOCHS, batch_size=BATCHSIZE, lr=lr
                        )
                        test_predictions.append(test_pred)
                        test_scores.append(r2_score(y_test, test_pred))
                except Exception as e2:
                    print(f"      Fold {fold+1} PyTorch also failed: {e2}")
                    cv_val_predictions.append(np.zeros_like(y_val))
                    cv_val_scores.append(0.0)
                    if evaluate_test_during_cv:
                        test_predictions.append(np.zeros_like(y_test))
                        test_scores.append(0.0)
            
        else:
            # Traditional ML models with try-catch for error handling
            try:
                # OS-specific parallelization settings
                if OS_TYPE == "Windows":
                    # Windows: Use fewer threads to avoid conflicts
                    n_jobs = min(2, os.cpu_count())
                else:
                    # Linux/Mac: Use more threads
                    n_jobs = min(4, os.cpu_count())

                # Traditional ML models with OS-optimized settings
                models = {
                    'Ridge': Ridge(),
                    'SVR': SVR(),
                    'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs),
                    'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0),
                    'LightGBM': lgb.LGBMRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                        verbose=-1,
                        enable_categorical=False,
                        force_row_wise=True
                    )
                }

                model = models[model_type]
                model.fit(X_tr, y_tr)

                # Validation predictions
                val_pred = model.predict(X_val)
                val_pred = sanitize_predictions(val_pred)  # Sanitize validation predictions
                cv_val_predictions.append(val_pred)

                r2_val = r2_score(y_val, val_pred)
                cv_val_scores.append(r2_val)

                # Method 1: Predict on test set during CV
                if evaluate_test_during_cv:
                    test_pred = model.predict(X_test)
                    test_pred = sanitize_predictions(test_pred)  # Sanitize test predictions
                    test_predictions.append(test_pred)

                    r2_test = r2_score(y_test, test_pred)
                    test_scores.append(r2_test)

            except Exception as e:
                print(f"      [ERROR] Error in CV (traditional) fold {fold+1}: {e}")
                import traceback
                print(f"      [ERROR] Full traceback:")
                traceback.print_exc()

                print(f"      [ERROR] Adding fallback values...")
                cv_val_predictions.append(np.zeros_like(y_val))
                cv_val_scores.append(0.0)
                if evaluate_test_during_cv:
                    test_predictions.append(np.zeros_like(y_test))
                    test_scores.append(0.0)

    # Method 2: Retrain on full data and evaluate test set
    # Average predictions across folds
    if test_predictions:
        y_pred = np.mean(test_predictions, axis=0)
    else:
        # If no predictions were made (all folds failed), return zeros like y_test
        y_pred = np.zeros_like(y_test)

    return y_pred, cv_val_scores, cv_val_predictions, test_predictions, test_scores

def run_full_experiment_with_predictions(fp_map, y_map, args=None, output_dir="result/2_standard_comp"):
    """Run experiments and save predictions for visualization"""
    
    # Initialize resource monitoring
    experiment_start_time = time.time()
    experiment_start_memory = get_memory_usage()
    experiment_start_cpu = get_cpu_usage()
    
    print(f"\n=== EXPERIMENT RESOURCE MONITORING ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start memory: {experiment_start_memory:.1f} MB")
    print(f"Start CPU: {experiment_start_cpu:.1f}%")
    print(f"========================================\n")
    
    # ===== RESTART/RESUME LOGIC =====
    def load_existing_results():
        """Load existing results if available"""
        results_file = Path(output_dir) / "all_results_with_predictions.csv"
        if results_file.exists():
            try:
                existing_df = pd.read_csv(results_file)
                print(f"Found existing results file with {len(existing_df)} combinations")
                return existing_df.to_dict('records')
            except Exception as e:
                print(f"Error loading existing results: {e}")
        return []

    def should_skip_combination(split_type, dataset, fingerprint, model_type, existing_results, failed_only=False):
        """Check if combination should be skipped based on restart config and filters"""
        if RESTART_CONFIG['mode'] == 'restart' or 2 in RESTART_CONFIG.get('force_restart_modules', []):
            return False

        if RESTART_CONFIG['mode'] == 'resume' or failed_only:
            # Check if combination exists in results
            for result in existing_results:
                if (result.get('split') == split_type and
                    result.get('dataset') == dataset and
                    result.get('fingerprint') == fingerprint and
                    result.get('model') == model_type):
                    # Check if result is complete and successful
                    r2_value = result.get('r2')
                    if pd.notna(r2_value) and r2_value not in [-999, 999]:
                        if failed_only:
                            # For --failed-only mode: only run failed experiments
                            if r2_value <= 0.0:
                                print(f"    ↻ Running {split_type}-{dataset}-{fingerprint}-{model_type} (failed: R²={r2_value:.3f})")
                                return False
                            else:
                                # Skip successful experiments in failed-only mode
                                return True
                        else:
                            # Normal resume mode
                            if r2_value <= 0.0:
                                print(f"    ↻ Retrying {split_type}-{dataset}-{fingerprint}-{model_type} (failed result: R²={r2_value:.3f})")
                                return False
                            else:
                                print(f"    ✓ Skipping {split_type}-{dataset}-{fingerprint}-{model_type} (successful: R²={r2_value:.3f})")
                                return True
                    elif not RESTART_CONFIG.get('resume_from_partial', True) and not failed_only:
                        print(f"    ↻ Redoing {split_type}-{dataset}-{fingerprint}-{model_type} (incomplete result)")
                        return False
                    elif failed_only:
                        # In failed-only mode, skip incomplete results
                        return True

            # If no existing result found
            if failed_only:
                # In failed-only mode, skip combinations with no existing results
                return True

        return False

    # Load existing results based on config
    if RESTART_CONFIG['mode'] in ['resume', 'smart']:
        existing_results = load_existing_results()
        all_results = existing_results.copy()
        print(f"Loaded {len(all_results)} existing results")

        # Analyze existing results for failed experiments
        if existing_results:
            successful_results = [r for r in existing_results if pd.notna(r.get('r2')) and r.get('r2') > 0.0]
            failed_results = [r for r in existing_results if pd.notna(r.get('r2')) and r.get('r2') <= 0.0]
            incomplete_results = [r for r in existing_results if pd.isna(r.get('r2')) or r.get('r2') in [-999, 999]]

            print(f"  ✓ Successful results: {len(successful_results)}")
            print(f"  ⚠️ Failed results (R²≤0): {len(failed_results)}")
            print(f"  ? Incomplete results: {len(incomplete_results)}")

            if failed_results:
                print(f"  ↻ Will retry {len(failed_results)} failed experiments")

        if RESTART_CONFIG['mode'] == 'smart' and RESTART_CONFIG.get('check_input_changes', True):
            # Here you could add logic to compare current config with saved config
            print("Smart mode: checking for input changes...")
    else:
        existing_results = []
        all_results = []
        print("Starting fresh experiment (restart mode)")

    all_predictions = {}  # Store predictions for later visualization
    
    # Model types to test
    model_types = ['Ridge', 'SVR', 'RandomForest', 'XGBoost', 'LightGBM', 'DNN']
    
    # Fingerprint types from config (Code 2 specific)
    fp_types = get_code_fingerprints(2)
    
    # Get split types from config
    available_splits = ACTIVE_SPLIT_TYPES
    
    total_combinations = 0
    completed_combinations = 0
    
    # Count total combinations first
    for split_type in available_splits:
        datasets = set()
        for key in fp_map[split_type].keys():
            dataset = key.split('_')[0]
            # Debug print
            print(f"  Debug - Found dataset: {dataset}")
            # Check if dataset matches any active dataset
            # Check if dataset is in active datasets for code 2
            if dataset in get_code_datasets(2):
                datasets.add(dataset)
                print(f"    -> Added {dataset} (active for code 2)")
        
        for dataset in datasets:
            train_key = f"{dataset}_train"
            test_key = f"{dataset}_test"
            if train_key in fp_map[split_type] and test_key in fp_map[split_type]:
                total_combinations += len(fp_types) * len(model_types)
    
    print(f"Total combinations to run: {total_combinations}")
    
    # Create previous_records folder at the beginning
    previous_records_dir = Path(output_dir) / "previous_records"
    previous_records_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified previous_records folder: {previous_records_dir}")
    
    # Apply split filter if specified
    if args.split:
        available_splits = [s for s in available_splits if s == args.split]
        print(f"\nFiltered to split: {available_splits}")

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
            # Debug print
            print(f"  Debug - Found dataset: {dataset}")
            # Check if dataset matches any active dataset
            # Check if dataset is in active datasets for code 2
            if dataset in get_code_datasets(2):
                datasets.add(dataset)
                print(f"    -> Added {dataset} (active for code 2)")
        
        for dataset in datasets:
            # Apply dataset filter if specified
            if args.dataset and dataset != args.dataset:
                continue

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
                # Apply fingerprint filter if specified
                if args.fingerprint and fp_type != args.fingerprint:
                    continue

                print(f"  Fingerprint: {fp_type}")
                
                if fp_type not in all_predictions[split_type][dataset]:
                    all_predictions[split_type][dataset][fp_type] = {}
                
                # Get fingerprints
                if fp_type == 'all':
                    # Combine all fingerprints
                    X_train_list = []
                    X_test_list = []
                    fp_names = []
                    
                    if 'morgan' in fp_map[split_type][train_key]:
                        X_train_list.append(fp_map[split_type][train_key]['morgan'])
                        X_test_list.append(fp_map[split_type][test_key]['morgan'])
                        fp_names.append('morgan')
                    if 'maccs' in fp_map[split_type][train_key]:
                        X_train_list.append(fp_map[split_type][train_key]['maccs'])
                        X_test_list.append(fp_map[split_type][test_key]['maccs'])
                        fp_names.append('maccs')
                    if 'avalon' in fp_map[split_type][train_key]:
                        X_train_list.append(fp_map[split_type][train_key]['avalon'])
                        X_test_list.append(fp_map[split_type][test_key]['avalon'])
                        fp_names.append('avalon')
                    
                    if X_train_list:
                        X_train = np.hstack(X_train_list)
                        X_test = np.hstack(X_test_list)
                    else:
                        # If no fingerprints found, try 'morgan' as default
                        X_train = fp_map[split_type][train_key].get('morgan', np.array([]))
                        X_test = fp_map[split_type][test_key].get('morgan', np.array([]))
                else:
                    X_train = fp_map[split_type][train_key][fp_type]
                    X_test = fp_map[split_type][test_key][fp_type]
                
                # Fingerprint data validation
                print(f"      Debug - {fp_type} fingerprint shapes: X_train={X_train.shape}, X_test={X_test.shape}")
                print(f"      Debug - {fp_type} fingerprint range: [{X_train.min()}, {X_train.max()}]")
                print(f"      Debug - {fp_type} fingerprint sparsity: {(X_train == 0).mean():.3f}")
                
                # Data quality checks
                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    print(f"      ERROR: Empty fingerprint data for {fp_type}")
                    continue
                
                if X_train.shape[1] == 0:
                    print(f"      ERROR: Zero features in fingerprint {fp_type}")
                    continue
                
                # Check if all values are zero
                if np.all(X_train == 0):
                    print(f"      WARNING: All zeros in training fingerprint {fp_type}")
                    print(f"      This may indicate fingerprint generation failed")
                
                # Check if all values are identical
                if len(np.unique(X_train)) == 1:
                    print(f"      WARNING: All values identical in training fingerprint {fp_type}")
                
                # Check data consistency
                if X_train.shape[1] != X_test.shape[1]:
                    print(f"      ERROR: Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")
                    continue
                
                # Check for NaN or infinite values
                if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                    print(f"      ERROR: NaN or Inf values in training data")
                    continue
                
                if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
                    print(f"      ERROR: NaN or Inf values in test data")
                    continue
                
                # Check target data consistency
                if len(y_train) != X_train.shape[0]:
                    print(f"      ERROR: Target size mismatch: y_train={len(y_train)}, X_train={X_train.shape[0]}")
                    continue
                
                if len(y_test) != X_test.shape[0]:
                    print(f"      ERROR: Target size mismatch: y_test={len(y_test)}, X_test={X_test.shape[0]}")
                    continue
                
                # Skip StandardScaler - use raw features
                # scaler = StandardScaler()
                # X_train_scaled = scaler.fit_transform(X_train)
                # X_test_scaled = scaler.transform(X_test)
                X_train_scaled = X_train
                X_test_scaled = X_test
                
                for model_type in model_types:
                    # Apply model filter if specified
                    if args.model and model_type != args.model:
                        completed_combinations += 1
                        continue

                    # Check if this combination should be skipped
                    if should_skip_combination(split_type, dataset, fp_type, model_type, existing_results, args.failed_only):
                        completed_combinations += 1
                        continue

                    completed_combinations += 1
                    progress = (completed_combinations / total_combinations) * 100
                    print(f"    Model: {model_type} ({completed_combinations}/{total_combinations} - {progress:.1f}%)")
                    
                    # Choose scaled or unscaled features based on model
                    # Only SVR and Ridge benefit from scaling (continuous features)
                    # DNN works well with binary fingerprints (0/1) without scaling
                    if model_type in ['SVR', 'Ridge']:
                        X_tr, X_te = X_train_scaled, X_test_scaled
                    else:
                        X_tr, X_te = X_train, X_test
                    
                    # For DNN, run TYPE1 and TYPE2 separately
                    cv_types_to_run = ['type1', 'type2'] if model_type == 'DNN' else ['traditional']

                    for cv_type_idx, cv_type in enumerate(cv_types_to_run):
                        # Clean up temp files before TYPE2 (memory optimization)
                        if model_type == 'DNN' and cv_type == 'type2':
                            cleanup_temp_files()
                        # CV training - Method 1 (evaluate test during CV)
                        try:
                            # Resource monitoring for CV training
                            cv_start_time = time.time()
                            cv_start_memory = get_memory_usage()
                            cv_start_cpu = get_cpu_usage()

                            y_pred_cv, cv_val_scores, cv_val_preds, test_preds, test_scores = train_model_cv(
                                model_type, X_tr, y_train, X_te, y_test, n_folds=CV, cv_type=cv_type
                            )

                            cv_end_time = time.time()
                            cv_end_memory = get_memory_usage()
                            cv_end_cpu = get_cpu_usage()

                            cv_execution_time = cv_end_time - cv_start_time
                            cv_memory_used = cv_end_memory - cv_start_memory
                            cv_cpu_used = cv_end_cpu - cv_start_cpu
                        
                            # Calculate CV statistics
                            if test_preds and test_scores:
                                cv_test_r2_mean = np.mean(test_scores)
                                cv_test_r2_std = np.std(test_scores)
                                cv_val_r2_mean = np.mean(cv_val_scores)
                                cv_val_r2_std = np.std(cv_val_scores)

                                # Calculate other metrics
                                test_pred_array = np.array(test_preds)
                                y_pred_std = np.std(test_pred_array, axis=0)
                                mean_std = np.mean(y_pred_std)

                                # For Type1: use val metrics std from CV (different val sets)
                                # For Type2: calculate from fold predictions on same test set
                                if cv_type == 'type1':
                                    # Type1: Calculate std from test predictions across folds
                                    if test_preds and len(test_preds) > 1:
                                        fold_metrics = []
                                        for fold_pred in test_preds:
                                            fold_metric = metric_prediction(y_test, fold_pred)
                                            fold_metrics.append(fold_metric)
                                        rmse_std = np.std([m['rmse'] for m in fold_metrics])
                                        mae_std = np.std([m['mae'] for m in fold_metrics])
                                        mse_std = np.std([m['mse'] for m in fold_metrics])
                                    else:
                                        # Fallback to 0 if no fold predictions
                                        rmse_std = mae_std = mse_std = 0.0
                                else:
                                    # Type2 or fallback: Calculate from fold predictions
                                    fold_metrics = []
                                    for fold_pred in test_preds:
                                        fold_metric = metric_prediction(y_test, fold_pred)
                                        fold_metrics.append(fold_metric)

                                    rmse_std = np.std([m['rmse'] for m in fold_metrics])
                                    mae_std = np.std([m['mae'] for m in fold_metrics])
                                    mse_std = np.std([m['mse'] for m in fold_metrics])
                            else:
                                cv_test_r2_mean = cv_test_r2_std = 0.0
                                cv_val_r2_mean = cv_val_r2_std = 0.0
                                mean_std = rmse_std = mae_std = mse_std = 0.0
                                y_pred_std = None
                        
                            # Store CV predictions
                            cv_suffix = f"_{cv_type}" if model_type == 'DNN' else ""
                            all_predictions[split_type][dataset][fp_type][f"{model_type}_cv{cv_suffix}"] = {
                                'y_true': y_test,
                                'y_pred': y_pred_cv,
                                'cv_preds': test_preds,
                                'y_pred_std': y_pred_std,
                                'cv_test_r2_values': test_scores,
                                'cv_val_r2_values': cv_val_scores
                            }

                            # Save predictions for DNN models (both NPY and CSV)
                            if model_type == 'DNN':
                                # Create directory structure
                                pred_dir = Path(output_dir) / dataset / f"split_{split_type}" / f"fp_{fp_type}" / f"cv_{cv_type}" / "predictions"
                                pred_dir.mkdir(parents=True, exist_ok=True)

                                # Save as NPY (fast loading)
                                npy_file = pred_dir / f"DNN_{cv_type}_predictions.npy"
                                np.save(npy_file, {
                                    'y_true': y_test,
                                    'y_pred': y_pred_cv,
                                    'cv_preds': test_preds if test_preds else [],
                                    'metadata': {
                                        'split': split_type,
                                        'dataset': dataset,
                                        'fingerprint': fp_type,
                                        'cv_type': cv_type,
                                        'n_folds': CV if cv_type == 'type1' else 1
                                    }
                                })
                                print(f"      Saved predictions (NPY): {npy_file.name}")

                                # Save as single CSV file with all predictions
                                csv_file = pred_dir / f"DNN_{cv_type}_predictions.csv"

                                if cv_type == 'type1' and test_preds:
                                    # Type1: Save y_true and all fold predictions in one CSV
                                    pred_data = {'y_true': y_test}
                                    for i, fold_pred in enumerate(test_preds):
                                        pred_data[f'fold_{i+1}_pred'] = fold_pred
                                    # Also add the average prediction
                                    pred_data['y_pred_avg'] = y_pred_cv
                                    pred_df = pd.DataFrame(pred_data)
                                    pred_df.to_csv(csv_file, index=False, float_format='%.6f')
                                    print(f"      Saved predictions (CSV): {csv_file.name}")
                                else:
                                    # Type2: Save y_true and y_pred
                                    pred_df = pd.DataFrame({
                                        'y_true': y_test,
                                        'y_pred': y_pred_cv
                                    })
                                    pred_df.to_csv(csv_file, index=False, float_format='%.6f')
                                    print(f"      Saved predictions (CSV): {csv_file.name}")

                            metrics_cv = metric_prediction(y_test, y_pred_cv)

                            result_cv = {
                                'split': split_type,
                                'dataset': dataset,
                                'fingerprint': fp_type,
                                'model': model_type,
                                'training': f'cv{cv_suffix}',
                                'cv_type': cv_type if model_type == 'DNN' else 'traditional',
                                'train_time': cv_execution_time,
                                'memory_used_mb': cv_memory_used,
                                'cpu_used_percent': cv_cpu_used,
                                'start_memory_mb': cv_start_memory,
                                'end_memory_mb': cv_end_memory,
                                'start_cpu_percent': cv_start_cpu,
                                'end_cpu_percent': cv_end_cpu,
                                'r2': metrics_cv['r2'],
                                'r2_std': cv_test_r2_std,
                                'cv_val_r2_mean': cv_val_r2_mean,
                                'cv_val_r2_std': cv_val_r2_std,
                                'cv_test_r2_mean': cv_test_r2_mean,
                                'cv_test_r2_std': cv_test_r2_std,
                                'rmse': metrics_cv['rmse'],
                                'rmse_std': rmse_std,
                                'mae': metrics_cv['mae'],
                                'mae_std': mae_std,
                                'mse': metrics_cv['mse'],
                                'mse_std': mse_std,
                                'pred_std': mean_std,
                            }
                            all_results.append(result_cv)

                            # Display results with CV type
                            if model_type == 'DNN':
                                cv_type_display = "[TYPE1-Research]" if cv_type == 'type1' else "[TYPE2-Production]"
                                if cv_type == 'type1':
                                    # TYPE1: Show standard deviations for both test and val
                                    print(f"      {cv_type_display} Test R²: {cv_test_r2_mean:.3f}±{cv_test_r2_std:.3f}, Val R²: {cv_val_r2_mean:.3f}±{cv_val_r2_std:.3f}, Time: {format_time(cv_execution_time)}")
                                    print(f"      {cv_type_display} RMSE: {metrics_cv['rmse']:.3f}±{rmse_std:.3f}, MAE: {metrics_cv['mae']:.3f}±{mae_std:.3f}, MSE: {metrics_cv['mse']:.3f}±{mse_std:.3f}")
                                else:
                                    # TYPE2: No standard deviation for test (single test), but show for val
                                    print(f"      {cv_type_display} Test R²: {cv_test_r2_mean:.3f}, Val R²: {cv_val_r2_mean:.3f}±{cv_val_r2_std:.3f}, Time: {format_time(cv_execution_time)}")
                                    print(f"      {cv_type_display} RMSE: {metrics_cv['rmse']:.3f}, MAE: {metrics_cv['mae']:.3f}, MSE: {metrics_cv['mse']:.3f}")
                            else:
                                print(f"      CV - Test R²: {cv_test_r2_mean:.3f}±{cv_test_r2_std:.3f}, Val R²: {cv_val_r2_mean:.3f}±{cv_val_r2_std:.3f}, Time: {format_time(cv_execution_time)}")
                                print(f"      CV - RMSE: {metrics_cv['rmse']:.3f}±{rmse_std:.3f}, MAE: {metrics_cv['mae']:.3f}±{mae_std:.3f}, MSE: {metrics_cv['mse']:.3f}±{mse_std:.3f}")

                        except Exception as e:
                            print(f"      Error in CV ({cv_type}): {e}")
                            result_cv = {
                                'split': split_type,
                                'dataset': dataset,
                                'fingerprint': fp_type,
                                'model': model_type,
                                'training': f'cv_{cv_type}' if model_type == 'DNN' else 'cv',
                                'cv_type': cv_type if model_type == 'DNN' else 'traditional',
                                'r2': np.nan,
                                'error': str(e)
                            }
                            all_results.append(result_cv)
                    
                    # Garbage collection after each model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # After all models for this fingerprint are done, save results and plots
                print(f"\n    Saving results for {dataset}-{split_type}-{fp_type}...")

                # Collect results for this fingerprint
                fp_results = [r for r in all_results if
                             r.get('dataset') == dataset and
                             r.get('split') == split_type and
                             r.get('fingerprint') == fp_type]

                # Get predictions for this fingerprint
                fp_predictions = all_predictions[split_type][dataset][fp_type] if fp_type in all_predictions[split_type][dataset] else {}

                # Save fingerprint results and create plots
                save_fingerprint_results(dataset, split_type, fp_type, fp_results, fp_predictions, output_dir)

            # Save intermediate results every dataset to previous_records folder
            if all_results:
                interim_df = pd.DataFrame(all_results)
                interim_file = previous_records_dir / f"interim_results_{completed_combinations}.csv"
                interim_df.to_csv(interim_file, index=False)
        
        # After all datasets and fingerprints for this split are done
        print(f"\n  Creating split-level comparison plots for {split_type}...")

        # Collect all results for this split
        split_results = [r for r in all_results if r.get('split') == split_type]

        # Get all predictions for this split
        split_predictions = all_predictions.get(split_type, {})

        # Save split-level comparison plots
        save_split_comparison(split_type, split_results, split_predictions, output_dir)

        # Major garbage collection after each split
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate final experiment statistics
    experiment_end_time = time.time()
    experiment_end_memory = get_memory_usage()
    experiment_end_cpu = get_cpu_usage()
    
    experiment_total_time = experiment_end_time - experiment_start_time
    experiment_total_memory = experiment_end_memory - experiment_start_memory
    experiment_total_cpu = experiment_end_cpu - experiment_start_cpu
    
    print(f"\n=== EXPERIMENT COMPLETED ===")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {format_time(experiment_total_time)}")
    print(f"Total memory change: {experiment_total_memory:+.1f} MB")
    print(f"Final memory usage: {experiment_end_memory:.1f} MB")
    print(f"Final CPU usage: {experiment_end_cpu:.1f}%")
    print(f"==============================\n")
    
    # Calculate detailed statistics
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Add experiment-level statistics
        experiment_stats = {
            'experiment_total_time': experiment_total_time,
            'experiment_total_memory_mb': experiment_total_memory,
            'experiment_start_memory_mb': experiment_start_memory,
            'experiment_end_memory_mb': experiment_end_memory,
            'experiment_start_cpu_percent': experiment_start_cpu,
            'experiment_end_cpu_percent': experiment_end_cpu,
            'total_combinations': len(all_results),
            'successful_combinations': len(results_df[results_df['r2'].notna()]),
            'failed_combinations': len(results_df[results_df['r2'].isna()])
        }
        
        # Save detailed results
        results_df.to_csv(f"{output_dir}/all_results_with_predictions.csv", index=False)
        
        # Save experiment statistics
        import json
        with open(f"{output_dir}/experiment_statistics.json", 'w') as f:
            json.dump(experiment_stats, f, indent=2, default=str)
        
        # Save predictions in chunks to manage memory
        import pickle
        with open(f"{output_dir}/all_predictions.pkl", 'wb') as f:
            pickle.dump(all_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print summary statistics
        print(f"=== SUMMARY STATISTICS ===")
        print(f"Total combinations: {experiment_stats['total_combinations']}")
        print(f"Successful: {experiment_stats['successful_combinations']}")
        print(f"Failed: {experiment_stats['failed_combinations']}")
        print(f"Success rate: {experiment_stats['successful_combinations']/experiment_stats['total_combinations']*100:.1f}%")
        
        # Model performance summary
        if not results_df.empty and 'r2' in results_df.columns:
            valid_results = results_df[results_df['r2'].notna()]
            if not valid_results.empty:
                print(f"\nBest R² score: {valid_results['r2'].max():.4f}")
                print(f"Average R² score: {valid_results['r2'].mean():.4f}")
                print(f"Worst R² score: {valid_results['r2'].min():.4f}")
        
        # Resource usage summary
        if 'train_time' in results_df.columns:
            valid_times = results_df[results_df['train_time'].notna()]
            if not valid_times.empty:
                total_training_time = valid_times['train_time'].sum()
                avg_training_time = valid_times['train_time'].mean()
                print(f"\nTotal training time: {format_time(total_training_time)}")
                print(f"Average training time per model: {format_time(avg_training_time)}")
        
        if 'memory_used_mb' in results_df.columns:
            valid_memory = results_df[results_df['memory_used_mb'].notna()]
            if not valid_memory.empty:
                total_memory_used = valid_memory['memory_used_mb'].sum()
                avg_memory_used = valid_memory['memory_used_mb'].mean()
                print(f"Total memory used: {total_memory_used:.1f} MB")
                print(f"Average memory per model: {avg_memory_used:.1f} MB")
        
        print(f"==========================\n")
        
        return results_df, all_predictions
    else:
        print("No results generated!")
        return pd.DataFrame(), {}

def extract_xy_from_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Dict, Dict]:
    """
    Extract SMILES (X) and target values (y) from loaded data
    """
    x_map = {}
    y_map = {}
    
    for split_type, datasets in data_dict.items():
        x_map[split_type] = {}
        y_map[split_type] = {}
        
        for dataset_key, df in datasets.items():
            # Find SMILES column
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            
            # Find target column
            target_col = None
            for col in ['logS', 'logS0', 'measured log solubility in mols per litre']:
                if col in df.columns:
                    target_col = col
                    break
            
            if smiles_col and target_col:
                x_map[split_type][dataset_key] = df[smiles_col].tolist()
                y_map[split_type][dataset_key] = np.array(df[target_col].values, dtype=np.float32)
    
    return x_map, y_map

def build_fingerprints_with_cache_OLD(data_dict: Dict, fingerprint_types: List[str]) -> Dict:
    """
    Build fingerprints using the cache system from mol_fps_maker.py
    Module 2: Only fingerprints (no descriptors)
    """
    fp_map = {}
    
    for split_type in ACTIVE_SPLIT_TYPES:
        if split_type not in data_dict:
            continue
            
        fp_map[split_type] = {}
        
        for dataset_name in get_code_datasets(2):  # Use code-specific datasets
            # Use dataset name directly as key (ws, de, lo, hu)
            dataset_key = dataset_name
            
            # Load train and test data
            for data_type in ['train', 'test']:
                key = f"{dataset_key}_{data_type}"
                
                if key not in data_dict[split_type]:
                    print(f"  Skipping {dataset_key}/{split_type}/{data_type} - no data found")
                    continue
                
                df = data_dict[split_type][key]
                
                # Find SMILES column
                smiles_col = None
                for col in ['SMILES', 'smiles', 'Smiles', 'isomeric_smiles']:
                    if col in df.columns:
                        smiles_col = col
                        break
                
                if not smiles_col:
                    print(f"  Warning: No SMILES column found for {key}")
                    continue
                
                smiles_list = df[smiles_col].tolist()
                
                # Check if cache exists to avoid unnecessary Mol conversion
                from pathlib import Path
                from config import RESULT_PATH, CACHE_CONFIG
                
                # Check cache file existence for this dataset
                fp_dir = Path(RESULT_PATH) / f'fingerprint/{dataset_key.lower()}/{split_type}'
                cache_file = fp_dir / f"{dataset_key.lower()}_{split_type}_{data_type}.npz"
                remake_fingerprint = CACHE_CONFIG.get('remake_fingerprint', False)
                
                # Generate fingerprints for each type
                fp_map[split_type][key] = {}
                
                if not cache_file.exists() or remake_fingerprint:
                    # No cache - need to create mols and generate fingerprints
                    from rdkit import Chem
                    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
                    mols = [mol for mol in mols if mol is not None]  # Filter out failed conversions
                    print(f"  Converting SMILES to molecules for {dataset_key}/{split_type}/{data_type} (cache not found)")
                    
                    for fp_type in fingerprint_types:
                        print(f"  Getting {fp_type} fingerprints for {dataset_key}/{split_type}/{data_type}...")
                        
                        fps = get_fingerprints_combined(
                            mols,
                            dataset_key,
                            split_type,
                            data_type,
                            fingerprint_type=fp_type
                        )
                        
                        if fps is not None and len(fps) > 0:
                            fp_map[split_type][key][fp_type] = fps
                            print(f"    Loaded {fp_type}: shape {fps.shape}")
                else:
                    # Cache exists - load directly without creating mols
                    print(f"  Using cached fingerprints for {dataset_key}/{split_type}/{data_type}")
                    
                    import numpy as np
                    try:
                        data = np.load(cache_file)
                        if 'morgan' in data and 'maccs' in data and 'avalon' in data:
                            morgan = data['morgan']
                            maccs = data['maccs']
                            avalon = data['avalon']
                            
                            for fp_type in fingerprint_types:
                                print(f"  Getting {fp_type} fingerprints for {dataset_key}/{split_type}/{data_type}...")
                                
                                # Generate the specific fingerprint type
                                if fp_type == 'morgan':
                                    fps = morgan
                                elif fp_type == 'maccs':
                                    fps = maccs
                                elif fp_type == 'avalon':
                                    fps = avalon
                                elif fp_type == 'all':
                                    fps = np.hstack([morgan, maccs, avalon])
                                elif fp_type == 'morgan+maccs':
                                    fps = np.hstack([morgan, maccs])
                                elif fp_type == 'morgan+avalon':
                                    fps = np.hstack([morgan, avalon])
                                elif fp_type == 'maccs+avalon':
                                    fps = np.hstack([maccs, avalon])
                                else:
                                    print(f"    Warning: Unknown fingerprint type '{fp_type}', using 'all'")
                                    fps = np.hstack([morgan, maccs, avalon])
                                
                                if fps is not None and len(fps) > 0:
                                    fp_map[split_type][key][fp_type] = fps
                                    print(f"    Loaded {fp_type}: shape {fps.shape}")
                        else:
                            print(f"    Error: Cache file missing fingerprint data")
                    except Exception as e:
                        print(f"    Error loading cache: {e}")
    
    return fp_map

def main():
    """Main execution function"""
    # Set up imports first
    from pathlib import Path
    from config import MODULE_NAMES
    from datetime import datetime
    import sys
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Standard ML Comparison with PyTorch Optimization')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to run (ws, de, lo, hu). Default: all datasets')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Multiple datasets to process (default: from CODE_SPECIFIC_DATASETS). Options: ws, de, lo, hu')
    parser.add_argument('--split', type=str, default=None,
                       help='Specific split type to run (rm, ac, cl, cs, en, pc, sa, sc, ti). Default: all splits')
    parser.add_argument('--fingerprint', type=str, default=None,
                       help='Specific fingerprint to run (morgan, maccs, avalon, morgan+maccs, morgan+avalon, maccs+avalon, all). Default: all fingerprints')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to run (Ridge, SVR, RandomForest, XGBoost, LightGBM, DNN). Default: all models')
    parser.add_argument('--failed-only', action='store_true',
                       help='Only retry failed experiments (R² ≤ 0)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs for DNN training')
    args, _ = parser.parse_known_args()

    print("="*80)
    print("[MODULE 2] STANDARD COMPARISON PYTORCH OPTIMIZED")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Module Focus: Standard ML Comparison with PyTorch")
    print("="*80)

    # Initialize performance monitoring
    if USE_MONITORING:
        monitor = PerformanceMonitor("Module_2", output_dir="logs/performance")
        monitor.start("Standard ML Comparison")
    else:
        monitor = None

    # Show selected filters
    if args.dataset or args.split or args.fingerprint or args.model or args.failed_only:
        print("\nFilters applied:")
        if args.dataset: print(f"  Dataset: {args.dataset}")
        if args.split: print(f"  Split: {args.split}")
        if args.fingerprint: print(f"  Fingerprint: {args.fingerprint}")
        if args.model: print(f"  Model: {args.model}")
        if args.failed_only: print(f"  Mode: Failed experiments only")
        if args.epochs: print(f"  Epochs: {args.epochs}")
        print("="*80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get module name from config
    module_name = MODULE_NAMES.get('2', '2_standard_comp_pytorch_optimized')

    # Get epochs with proper priority
    global EPOCHS  # Make EPOCHS available globally
    EPOCHS = get_epochs_for_module('2', args)
    print(f"\nUsing epochs: {EPOCHS}")

    # Create logs directory
    log_dir = Path(f"logs/{module_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module_name}_{timestamp}.log"

    # Create a custom logger class
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            self.log.write(f"{'='*60}\n")
            self.log.write(f"Module 2 (Standard Comparison) Execution Started: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n\n")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    # Redirect stdout to both terminal and file
    logger = Logger(log_file)
    sys.stdout = logger

    print("Starting Optimized PyTorch-based Standard Comparison Experiment")
    print(f"Device: {device}")
    print(f"Configuration: EPOCHS={EPOCHS}, BATCH_SIZE={BATCHSIZE}, LR={lr}, CV_FOLDS={CV}")

    # Get code-specific configurations with priority: argparse > CODE_SPECIFIC_DATASETS > data directory
    from config import CODE_SPECIFIC_DATASETS

    if args.datasets:
        code_datasets = args.datasets
        print(f"Datasets from argparse: {code_datasets}")
    elif args.dataset:
        code_datasets = [args.dataset]
        print(f"Dataset from argparse (single): {code_datasets}")
    elif '2' in CODE_SPECIFIC_DATASETS:
        code_datasets = CODE_SPECIFIC_DATASETS['2']
        print(f"Datasets from CODE_SPECIFIC_DATASETS: {code_datasets}")
    else:
        # Fallback: scan data directory
        code_datasets = []
        for split_type in ACTIVE_SPLIT_TYPES:
            split_dir = Path(DATA_PATH) / 'train' / split_type
            if split_dir.exists():
                for csv_file in split_dir.glob('*_train.csv'):
                    dataset = csv_file.stem.split('_')[1]  # Extract dataset from filename
                    if dataset not in code_datasets:
                        code_datasets.append(dataset)
        print(f"Datasets from data directory scan: {code_datasets}")

    code_fingerprints = get_code_fingerprints(2)  # Code 2

    print(f"Active Datasets (Code 2): {code_datasets}")
    print(f"Active Split Types: {ACTIVE_SPLIT_TYPES}")
    print(f"Active Fingerprints (Code 2): {code_fingerprints}")
    print(f"Optimizations: ReLU activation, Adam optimizer, ReduceLROnPlateau scheduler")
    print(f"Early stopping with patience=20, Gradient clipping for stability")
    print(f"Fingerprint rebuild: {'FORCED' if FORCE_REBUILD_FINGERPRINTS else 'REUSE EXISTING'}")

    # Check renew setting from config
    from config import MODEL_CONFIG
    renew = MODEL_CONFIG.get('renew', False)

    # Resume functionality based on renew setting
    completed_tasks = set()
    previous_records_dir = Path("previous_records")

    if renew:
        print("🆕 RENEW=True: Starting fresh experiment (ignoring previous results)")
        # Optionally clean up previous results
        if previous_records_dir.exists():
            print(f"⚠️  Previous results exist but will be ignored due to renew=True")
    else:
        print("🔄 RENEW=False: Checking for previous results to resume...")
        if previous_records_dir.exists():
            interim_files = list(previous_records_dir.glob("interim_results_*.csv"))
            if interim_files:
                latest_interim = max(interim_files, key=lambda p: p.stat().st_mtime)
                try:
                    previous_df = pd.read_csv(latest_interim)
                    for _, row in previous_df.iterrows():
                        task_key = (row.get('dataset', ''), row.get('split_type', ''), row.get('fingerprint', ''))
                        completed_tasks.add(task_key)
                    print(f"📋 Found {len(completed_tasks)} completed tasks from {latest_interim.name}")
                    print("✅ Will skip completed tasks and resume from interruption point")
                except Exception as e:
                    print(f"⚠️  Could not load previous results: {e}")
                    print("🆕 No valid previous results found, starting fresh")
            else:
                print("📄 No previous interim results found, starting fresh")

    total_tasks = len(code_datasets) * len(ACTIVE_SPLIT_TYPES) * len(code_fingerprints)
    remaining_tasks = total_tasks - len(completed_tasks)
    print(f"Total tasks: {total_tasks}, Completed: {len(completed_tasks)}, Remaining: {remaining_tasks}")
    print(f"Renew setting: {renew} ({'Fresh start' if renew else 'Resume mode'})")
    
    try:
        # Load original datasets directly  
        print("\nLoading datasets...")
        
        # Load and split datasets
        all_data = {}
        x_map = {split: {} for split in ACTIVE_SPLIT_TYPES}  # Initialize x_map for SMILES data
        y_map = {split: {} for split in ACTIVE_SPLIT_TYPES}  # Initialize y_map for target values
        
        from pathlib import Path
        
        for dataset_name in code_datasets:
            print(f"  Loading {dataset_name}...")
            dataset_full = DATASETS.get(dataset_name, dataset_name)
            
            for split_type in ACTIVE_SPLIT_TYPES:
                train_path_short = Path(f"data/train/{split_type}/{split_type}_{dataset_name}_train.csv")
                test_path_short = Path(f"data/test/{split_type}/{split_type}_{dataset_name}_test.csv")
                
                train_path_full = Path(f"data/train/{split_type}/{split_type}_{dataset_full}_train.csv")
                test_path_full = Path(f"data/test/{split_type}/{split_type}_{dataset_full}_test.csv")
            
                # Use whichever exists
                if train_path_short.exists() and test_path_short.exists():
                    train_df = pd.read_csv(train_path_short)
                    test_df = pd.read_csv(test_path_short)
                elif train_path_full.exists() and test_path_full.exists():
                    train_df = pd.read_csv(train_path_full)
                    test_df = pd.read_csv(test_path_full)
                else:
                    train_df = None
                    test_df = None
                
                if train_df is not None and test_df is not None:
                    all_data[f"{dataset_name}_{split_type}_train"] = train_df
                    all_data[f"{dataset_name}_{split_type}_test"] = test_df
                    
                    # Extract SMILES and target values
                    smiles_col = 'smiles' if 'smiles' in train_df.columns else 'SMILES'
                    target_col = None
                    for col in ['logS', 'logS0', 'measured log solubility in mols per litre', 'Solubility', 'target']:
                        if col in train_df.columns:
                            target_col = col
                            break
                    
                    if smiles_col in train_df.columns and target_col:
                        x_map[split_type][f"{dataset_name}_train"] = train_df[smiles_col].tolist()
                        x_map[split_type][f"{dataset_name}_test"] = test_df[smiles_col].tolist()
                        import numpy as np
                        y_map[split_type][f"{dataset_name}_train"] = np.array(train_df[target_col].values, dtype=np.float32)
                        y_map[split_type][f"{dataset_name}_test"] = np.array(test_df[target_col].values, dtype=np.float32)
        
        # Data quality check
        print("\n=== DATA QUALITY CHECK ===")
        for split_type in x_map.keys():
            print(f"\nSplit: {split_type}")
            for key in x_map[split_type].keys():
                if key in y_map[split_type]:
                    x_data = x_map[split_type][key]
                    y_data = y_map[split_type][key]
                    print(f"  {key}: {len(x_data)} samples, y_range: [{y_data.min():.3f}, {y_data.max():.3f}]")
                    
                    # Check for data quality issues
                    if len(x_data) < 10:
                        print(f"    [WARNING]  WARNING: Very small dataset ({len(x_data)} samples)")
                    if y_data.std() < 0.1:
                        print(f"    [WARNING]  WARNING: Low target variance (std={y_data.std():.3f})")
        print("=========================")
        
        # Build fingerprints using cache
        print("\nBuilding fingerprints using cache system...")
        fp_map = {split: {} for split in ACTIVE_SPLIT_TYPES}
        
        for key, df in all_data.items():
            parts = key.split('_')
            # key format: "{dataset_name}_{split_type}_{data_type}"
            dataset_name = parts[0]
            split_type = parts[1]
            data_type = parts[2]
            
            # Get SMILES
            smiles_col = None
            for col in ['SMILES', 'smiles', 'Smiles', 'isomeric_smiles']:
                if col in df.columns:
                    smiles_col = col
                    break
            
            if smiles_col:
                smiles_list = df[smiles_col].tolist()
                
                # Check if cache exists to avoid unnecessary Mol conversion
                from pathlib import Path
                from config import RESULT_PATH, CACHE_CONFIG
                
                # Check cache file existence for this dataset
                fp_dir = Path(RESULT_PATH) / f'fingerprint/{dataset_name.lower()}/{split_type}'
                cache_file = fp_dir / f"{dataset_name.lower()}_{split_type}_{data_type}.npz"
                remake_fingerprint = CACHE_CONFIG.get('remake_fingerprint', False)
                
                # Use consistent key format for fp_map: "{dataset_name}_{data_type}"
                fp_key = f"{dataset_name}_{data_type}"
                fp_map[split_type][fp_key] = {}
                
                # Try to load fingerprints from cache without creating mols (for Module 2 efficiency)
                print(f"  Using cached fingerprints for {dataset_name}/{data_type}")
                cache_success = True
                
                for fp_type in code_fingerprints:
                    print(f"  Getting {fp_type} fingerprints for {dataset_name}/{split_type}/{data_type}...")
                    
                    # Use the new cache-only function from mol_fps_maker
                    from extra_code.mol_fps_maker import get_fingerprints_from_cache_only
                    fps = get_fingerprints_from_cache_only(
                        dataset_name,
                        split_type,
                        data_type,
                        fingerprint_type=fp_type
                    )
                    
                    if fps is not None:
                        fp_map[split_type][fp_key][fp_type] = fps
                        print(f"    Loaded: shape {fps.shape}")
                    else:
                        print(f"    Cache not found, need to generate fingerprints")
                        cache_success = False
                        break
                
                # If cache loading failed, fall back to traditional method with mols
                if not cache_success:
                    from rdkit import Chem
                    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
                    mols = [mol for mol in mols if mol is not None]
                    print(f"  Converting SMILES to molecules for {dataset_name}/{data_type} (cache not found)")
                    
                    for fp_type in code_fingerprints:
                        print(f"  Getting {fp_type} fingerprints for {dataset_name}/{split_type}/{data_type}...")
                        fps = get_fingerprints_combined(
                            mols,
                            dataset_name,
                            split_type,
                            data_type,
                            fingerprint_type=fp_type
                        )
                        if fps is not None:
                            fp_map[split_type][fp_key][fp_type] = fps
                            print(f"    Loaded: shape {fps.shape}")
        
        # Run experiments
        print("\nRunning experiments...")
        results_df, all_predictions = run_full_experiment_with_predictions(fp_map, y_map, args)
        
        print("\nExperiment completed!")
        print(f"Results saved to: {target_path}")
        
        # Print summary statistics
        if not results_df.empty:
            print("\n" + "="*60)
            print("SUMMARY STATISTICS")
            print("="*60)
            
            # Best models by dataset
            for training in ['cv_method1']:
                print(f"\n{training.upper()} Training Results:")
                print("-" * 40)
                
                training_df = results_df[results_df['training'] == training]
                
                for dataset in ['ws', 'de', 'lo', 'hu']:
                    dataset_df = training_df[training_df['dataset'] == dataset]
                    if not dataset_df.empty and 'r2' in dataset_df.columns:
                        best_row = dataset_df.loc[dataset_df['r2'].idxmax()]
                        print(f"\n{get_dataset_display_name(dataset)} Dataset:")
                        print(f"  Best: {best_row['model']} - {best_row['fingerprint']}")
                        print(f"  R² = {best_row['r2']:.3f}")
                        if training == 'cv_method1' and 'r2_std' in best_row:
                            print(f"  R² std = {best_row['r2_std']:.3f}")
                            if 'cv_test_r2_mean' in best_row:
                                print(f"  CV Test R² = {best_row['cv_test_r2_mean']:.3f}±{best_row['cv_test_r2_std']:.3f}")
                                print(f"  CV Val R² = {best_row['cv_val_r2_mean']:.3f}±{best_row['cv_val_r2_std']:.3f}")
                            # Output standard deviation of additional metrics
                            if 'rmse' in best_row and 'rmse_std' in best_row:
                                print(f"  RMSE = {best_row['rmse']:.3f}±{best_row['rmse_std']:.3f}")
                            if 'mae' in best_row and 'mae_std' in best_row:
                                print(f"  MAE = {best_row['mae']:.3f}±{best_row['mae_std']:.3f}")
                            if 'mse' in best_row and 'mse_std' in best_row:
                                print(f"  MSE = {best_row['mse']:.3f}±{best_row['mse_std']:.3f}")
            
            # Create enhanced visualizations
            print("\nCreating enhanced visualizations...")
            plot_enhanced_results_comparison(results_df, all_predictions, target_path)
            
            # Create prediction scatter plots
            print("\nCreating prediction scatter plots...")
            create_prediction_scatter_plots(all_predictions, results_df, target_path)
            
            # Create CV comparison scatter plots
            print("\nCreating CV comparison scatter plots...")
            plot_cv_comparison_scatter(all_predictions, results_df, target_path)
            
            # Plot original datasets distribution
            print("\nPlotting original datasets distribution...")
            plot_original_datasets_distribution(target_path)
    
    finally:
        # Clean up temporary files
        cleanup_temp_files()

        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # End performance monitoring
        if USE_MONITORING and monitor:
            monitor.end()

        # Close log file
        logger.log.write(f"\n{'='*60}\n")
        logger.log.write(f"Module 2 (Standard Comparison) Execution Finished: {datetime.now()}\n")
        logger.log.write(f"{'='*60}\n")
        sys.stdout = logger.terminal
        logger.close()

if __name__ == "__main__":
    main()