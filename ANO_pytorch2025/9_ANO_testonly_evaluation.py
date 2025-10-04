#!/usr/bin/env python3
"""
ANO Test-Only Evaluation - ANO Framework Module 9
=================================================

PURPOSE:
Module 9 performs independent evaluation of the best models from Module 8
on held-out test datasets to assess generalization performance.

KEY FEATURES:
1. **Independent Testing**: Uses test-only datasets never seen during training
2. **Model Loading**: Loads best models from Module 8
3. **Feature Consistency**: Ensures same feature extraction as training
4. **Generalization Metrics**: Evaluates on unseen molecular structures
5. **Cross-dataset Testing**: Tests models on different datasets
6. **Production Validation**: Final check before deployment

RECENT UPDATES (2024):
- Enhanced model loading for both .pt and .pth files
- Added StandardScaler for test data consistency
- Added comprehensive metrics (Pearson, Spearman, percentile errors)
- Fixed feature extraction to match training preprocessing
- Improved plot generation with scatter plots

TEST-ONLY DATASETS:
- FreeSolv: Free solvation database
- Lipophilicity: Lipophilicity measurements
- AqSolDB: Aqueous solubility database
- BigSolDB: Large solubility database

EVALUATION PROTOCOL:
1. Load best model from Module 8
2. Extract same features as training
3. Apply same preprocessing (StandardScaler if used)
4. Generate predictions
5. Calculate comprehensive metrics
6. Create visualization plots

OUTPUT STRUCTURE:
result/9_test_only/
├── {test_dataset}/
│   ├── {model}_{train_dataset}_{split}/
│   │   ├── predictions.csv
│   │   ├── metrics.json
│   │   └── scatter_plot.png
│   └── summary.csv
├── plots/
│   └── cross_dataset_performance.png
└── evaluation_summary.json

METRICS:
- R² Score: Coefficient of determination
- RMSE: Root mean squared error
- MAE: Mean absolute error
- Pearson: Linear correlation
- Spearman: Rank correlation
- Percentile Errors: Error distribution analysis

USAGE:
python 9_ANO_testonly_evaluation.py [options]
  --model-dir: Path to Module 8 models (default: result/8_final_model)
  --test-dataset: Specific test dataset (FreeSolv/Lipophilicity/AqSolDB/BigSolDB)
  --train-dataset: Training dataset to evaluate (ws/de/lo/hu)
  --split: Split type to evaluate (rm/ac/cl/cs/en/pc/sa/sc/ti)
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from config import MODULE_NAMES
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import performance monitoring
try:
    from extra_code.performance_monitor import PerformanceMonitor, get_device_with_monitoring
    USE_MONITORING = True
except ImportError:
    USE_MONITORING = False
    print("Note: Performance monitoring not available")

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Import configuration
from config import (
    MODEL_CONFIG,
    CODE_SPECIFIC_DATASETS,
    ACTIVE_SPLIT_TYPES,
    DATASET_DISPLAY_NAMES,
    DATA_PATH,
    RESULT_PATH,
    DATASETS,
    get_code_datasets, get_code_fingerprints
)

# Import functions from extra_code
from extra_code.mol_fps_maker import get_fingerprints_cached
from extra_code.ano_feature_selection import (
    SimpleDNN,
    FlexibleDNNModel,
    selection_data_descriptor_compress,
    convert_params_to_selection
)

# Import MOFO feature generation
try:
    from extra_code.ano_feature_search import search_data_descriptor_compress
except:
    search_data_descriptor_compress = None

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Paths
TEST_DATA_DIR = Path(DATA_PATH) / 'test' / 'to'
MODEL_DIR = Path(RESULT_PATH) / '8_ANO_final_model_training'
OUTPUT_DIR = Path(RESULT_PATH) / '9_ANO_testonly_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset name mapping
CODE_DATASETS = get_code_datasets(9)  # Code 9
CODE_FINGERPRINTS = get_code_fingerprints(9)  # Code 9
DATASET_MAPPING = {k: DATASETS[k] for k in CODE_DATASETS}

def has_problematic_elements(mol):
    """
    Check if molecule contains elements that might cause descriptor calculation issues

    Args:
        mol: RDKit molecule object

    Returns:
        bool: True if molecule has problematic elements, False otherwise
    """
    if mol is None:
        return True

    # Elements that may cause issues with certain descriptors
    problematic_elements = {'Si', 'B', 'Se', 'Te', 'As', 'Sb', 'Bi', 'Al'}

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in problematic_elements:
            return True

    return False

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
    try:
        # Load preprocessed data
        train_smiles, train_y, test_smiles, test_y = load_preprocessed_data(dataset_short, split_type)

        print(f"  Loaded {dataset_short.upper()}-{split_type}: {len(train_smiles)} train, {len(test_smiles)} test")

        # Convert SMILES to molecules
        train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
        test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]

        # Filter out None molecules and molecules with problematic elements
        train_valid = []
        train_excluded = 0
        for mol, y in zip(train_mols, train_y):
            if mol is not None:
                if not has_problematic_elements(mol):
                    train_valid.append((mol, y))
                else:
                    train_excluded += 1

        test_valid = []
        test_excluded = 0
        for mol, y in zip(test_mols, test_y):
            if mol is not None:
                if not has_problematic_elements(mol):
                    test_valid.append((mol, y))
                else:
                    test_excluded += 1

        if train_excluded > 0 or test_excluded > 0:
            print(f"    ⚠️ Excluded molecules with problematic elements: {train_excluded} train, {test_excluded} test")

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
            train_mols_filtered, dataset_short.upper(), split_type, 'train', module_name='9_ANO_testonly_evaluation'
        )
        test_morgan, test_maccs, test_avalon = get_fingerprints_cached(
            test_mols_filtered, dataset_short.upper(), split_type, 'test', module_name='9_ANO_testonly_evaluation'
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

def load_test_datasets(max_samples=None):
    """Load test-only datasets from data/test/to folder
    
    Args:
        max_samples: Maximum number of samples to load per dataset (None for all)
    """
    datasets = {}
    
    if not TEST_DATA_DIR.exists():
        print(f"Error: Test directory {TEST_DATA_DIR} does not exist")
        return datasets
    
    print(f"\nLoading test datasets from {TEST_DATA_DIR}...")
    if max_samples:
        print(f"  Limiting each dataset to {max_samples} samples")
    
    # Dynamically find all CSV files in test directory
    csv_files = list(TEST_DATA_DIR.glob("*.csv"))
    print(f"  Found {len(csv_files)} CSV files in test directory")

    # Map files to abbreviated keys
    test_files = {}
    for csv_file in csv_files:
        filename = csv_file.name
        # Generate key from filename (first 2-3 letters of main name)
        if filename.lower().startswith('aqsoldb'):
            key = 'aq'
        elif filename.lower().startswith('bigsoldb'):
            key = 'bi'
        elif filename.lower().startswith('freesolv'):
            key = 'fr'
        elif filename.lower().startswith('lipophilicity'):
            key = 'li'
        else:
            # Generic key from filename
            base_name = filename.replace('.csv', '').replace('_test', '').replace('_Test', '')
            key = base_name[:3].lower()

        test_files[key] = filename
        print(f"    {filename} -> {key}")

    for key, filename in test_files.items():
        filepath = TEST_DATA_DIR / filename
        if filepath.exists():
            if max_samples:
                df = pd.read_csv(filepath, nrows=max_samples)
                total_rows = sum(1 for _ in open(filepath)) - 1  # Count total rows (minus header)
                print(f"  Loaded {filename}: {len(df)}/{total_rows} molecules (limited)")
            else:
                df = pd.read_csv(filepath)
                print(f"  Loaded {filename}: {len(df)} molecules")
            datasets[key] = df
        else:
            print(f"  Warning: {filename} not found")
    
    return datasets

def create_scatter_plot(y_test, y_pred, model_name, test_dataset, r2_value, output_dir):
    """
    Create scatter plot of predicted vs actual values with x=y diagonal line.

    Args:
        y_test (np.ndarray): Actual test values
        y_pred (np.ndarray): Predicted values
        model_name (str): Model identifier (format: dataset_split_module)
        test_dataset (str): Test dataset name
        r2_value (float): R² score to display on plot
        output_dir (Path): Output directory for saving plots

    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(8, 8))
    
    # Create scatter plot
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    
    # Add x=y diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='x=y')
    
    # Labels and title
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name} on {test_dataset}\nR² = {r2_value:.4f}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    
    # Add legend
    plt.legend()
    
    # Parse model name to get dataset, split, and module
    # model_name format: dataset_split_module (e.g., de_rm_FO)
    parts = model_name.split('_')
    dataset = parts[0]
    split_type = parts[1]
    
    # Save plot with organized folder structure
    plot_dir = output_dir / 'plots' / dataset / split_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / f'{model_name}_prediction_{test_dataset}.png'
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    return str(plot_file)

def evaluate_model_from_module8(model_path, X_test, y_test):
    """Evaluate a Module 8 trained model with enhanced metrics"""

    if not Path(model_path).exists():
        # Check for .pth fallback
        pth_path = Path(str(model_path).replace('.pt', '.pth'))
        if pth_path.exists():
            model_path = pth_path
        else:
            return None

    try:
        # Check if there's a model info file
        info_path = Path(str(model_path).replace('.pt', '.json').replace('.pth', '.json'))
        model_info = {}
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)

        # Load model based on file type
        if str(model_path).endswith('.pth'):
            # Load regular PyTorch model with FlexibleDNNModel support
            checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)

            # Extract model from checkpoint
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Need to recreate model from info
                if model_info.get('is_flexible', False):
                    # Recreate FlexibleDNNModel
                    params = model_info['model_params']
                    model = FlexibleDNNModel(
                        input_dim=model_info['input_dim'],
                        n_layers=params['n_layers'],
                        hidden_dims=params['hidden_dims'],
                        activation=params.get('activation', 'relu'),
                        dropout_rate=params.get('dropout_rate', 0.2),
                        use_batch_norm=params.get('use_batch_norm', False)
                    )
                else:
                    # Recreate SimpleDNN
                    params = model_info['model_params']
                    model = SimpleDNN(
                        input_dim=model_info['input_dim'],
                        hidden_dims=params['hidden_dims'],
                        dropout_rate=params.get('dropout_rate', 0.2)
                    )
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(DEVICE)
            else:
                print(f"    Warning: Unknown checkpoint format")
                return None
        else:
            # Load TorchScript model
            model = torch.jit.load(str(model_path), map_location=DEVICE)

        model.eval()

        # Apply StandardScaler to test data
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        # Predict
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        with torch.inference_mode():
            y_pred = model(X_test_tensor).cpu().numpy()

        # Flatten if needed
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()

        # Calculate comprehensive metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Additional metrics
        residuals = y_test - y_pred
        max_error = np.max(np.abs(residuals))
        median_ae = np.median(np.abs(residuals))
        std_error = np.std(residuals)

        # Correlation coefficients
        pearson_r = np.corrcoef(y_test, y_pred)[0, 1]
        from scipy import stats
        spearman_r, _ = stats.spearmanr(y_test, y_pred)

        # Percentile errors
        percentiles = [25, 50, 75, 90, 95]
        percentile_errors = {}
        for p in percentiles:
            percentile_errors[f'p{p}_error'] = np.percentile(np.abs(residuals), p)

        return {
            'r2': r2,
            'rmse': rmse,
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            'median_ae': median_ae,
            'std_error': std_error,
            'pearson_r': pearson_r,
            'spearman_r': spearman_r,
            'n_samples': len(y_test),
            'y_pred': y_pred,
            'y_test': y_test,
            **percentile_errors  # Add percentile errors
        }

    except Exception as e:
        print(f"    Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def has_problematic_elements(mol):
    """
    Check if molecule contains problematic elements for 3D/charge calculations.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object

    Returns:
        bool: True if molecule contains problematic elements, False otherwise
    """
    if mol is None:
        return False

    problematic_elements = {
        'Zn', 'Cd', 'Hg',  # Transition metals
        'As', 'Sb', 'Bi',  # Metalloids
        'Se', 'Te', 'Po',  # Chalcogens
        'Be', 'Mg',        # Alkaline earth metals
        'Al', 'Ga', 'In', 'Tl',  # Post-transition metals
        'Sn', 'Pb',        # Group 14 metals
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Lanthanides
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'  # Actinides
    }

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in problematic_elements:
            return True
    return False

# Removed: calculate_descriptors_from_module8_cache()
# No longer needed - we now use ChemDescriptorCalculator + selection_data_descriptor_compress
# which ensures exact same descriptor calculation as Modules 3-8

def prepare_features_for_module(test_df, test_key, train_dataset, split_type, module):
    """Prepare features for specific module based on Module 8's training

    This function ensures that testonly data uses the EXACT same descriptor calculation
    as Modules 3-8 by:
    1. Using ChemDescriptorCalculator to generate full descriptor cache
    2. Using selection_data_descriptor_compress to apply same selection as training
    """

    # Get SMILES and target values
    smiles_col = 'SMILES' if 'SMILES' in test_df.columns else 'smiles'

    # Check for target column with various names
    if 'target' in test_df.columns:
        target_col = 'target'
    elif 'logS' in test_df.columns:
        target_col = 'logS'
    elif 'LogS' in test_df.columns:
        target_col = 'LogS'
    else:
        print(f"    Error: Target column not found. Available columns: {test_df.columns.tolist()}")
        return None, None

    if smiles_col not in test_df.columns:
        print(f"    Error: SMILES column not found. Available columns: {test_df.columns.tolist()}")
        return None, None

    smiles_list = test_df[smiles_col].tolist()
    y_test = test_df[target_col].values

    # Convert to molecules and filter out problematic elements
    mols = []
    valid_indices = []
    excluded_count = 0
    problematic_smiles = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Check for problematic elements
            if has_problematic_elements(mol):
                excluded_count += 1
                problematic_smiles.append(smi)
                continue  # Skip molecules with problematic elements
            mols.append(mol)
            valid_indices.append(i)

    if excluded_count > 0:
        print(f"    ⚠️ Excluded {excluded_count} molecules with problematic elements (Zn, Cd, Hg, etc.)")
        if len(problematic_smiles) <= 5:
            print(f"       Examples: {problematic_smiles}")
        else:
            print(f"       First 5 examples: {problematic_smiles[:5]}")

    if len(mols) == 0:
        print(f"    Error: No valid molecules found after filtering")
        return None, None

    y_test = y_test[valid_indices]
    print(f"    Processing {len(mols)} valid molecules (after filtering)")

    # ========================================
    # Step 1: Generate fingerprints (same as all modules)
    # ========================================
    morgan_fps, maccs_fps, avalon_fps = get_fingerprints_cached(
        mols=mols,
        dataset_name='testonly',
        split_type=test_key,
        data_type='test',
        module_name='9_ANO_testonly_evaluation'
    )
    base_features = np.hstack([morgan_fps, maccs_fps, avalon_fps])

    # ========================================
    # Step 2: Generate full descriptor cache (same as Module 3/4)
    # This ensures selection_data_descriptor_compress can load from cache
    # Testonly cache structure: result/chemical_descriptors/testonly/{dataset}/
    # ========================================
    from extra_code.chem_descriptor_maker import ChemDescriptorCalculator
    calculator = ChemDescriptorCalculator(cache_dir='result/chemical_descriptors')

    print(f"    Calculating full descriptors for testonly data...")
    full_descriptors = calculator.calculate_selected_descriptors(
        mols,
        dataset_name=test_key,     # Dataset name (aq, bi, fr, li)
        split_type='testonly',     # Split type for testonly
        subset='test',
        mols_3d=None
    )
    print(f"    Full descriptors calculated: {full_descriptors.shape}")

    # ========================================
    # Step 3: Apply module-specific selection
    # ========================================
    if module == 'MO':
        # MO uses only fingerprints
        return base_features, y_test

    elif module in ['FO', 'FOMO', 'MOFO']:
        # All three use selection_data_descriptor_compress (same as Module 4/6/7)
        model_info_file = Path(RESULT_PATH) / f'8_ANO_final_model_training/{train_dataset}/{split_type}/{train_dataset}_{split_type}_{module}_model.json'

        if not model_info_file.exists():
            print(f"    WARNING: Model info not found for {module}")
            return base_features, y_test

        with open(model_info_file, 'r') as f:
            model_info = json.load(f)

        expected_dim = model_info.get('input_dim')
        print(f"    Model expects {expected_dim} features")

        # Get best_params (Module 8 stores this for all modules)
        descriptor_info = model_info.get('descriptor_info')
        if not descriptor_info or descriptor_info.get('type') != 'best_params':
            print(f"    WARNING: No best_params found for {module}, using base features only")
            return base_features, y_test

        best_params = descriptor_info.get('best_params', {})

        # Convert to selection array
        selection_dict = convert_params_to_selection(best_params)

        # Apply same selection as Module 4/6/7 using cached descriptors
        # Cache name format: testonly-aq (matches ChemDescriptorCalculator structure)
        cache_name = f'testonly-{test_key}'
        X_test, _ = selection_data_descriptor_compress(
            selection_dict,
            base_features,
            mols,
            cache_name,
            target_path=str(OUTPUT_DIR / 'descriptor_cache'),
            save_res="np",
            mols_3d=None
        )
        print(f"    Features after selection: {X_test.shape}")

        # Verify dimensions
        if X_test.shape[1] != expected_dim:
            diff = expected_dim - X_test.shape[1]
            print(f"    ERROR: Dimension mismatch: {X_test.shape[1]} != {expected_dim} (diff: {diff})")
            print(f"    This indicates descriptor calculation is not matching Module 3-8")
            return None, None

        print(f"    ✓ Final features shape: {X_test.shape}")
        return X_test, y_test

    # For all other cases, return base features
    return base_features, y_test

def pre_generate_descriptors(test_datasets, cache_dir='result/chemical_descriptors'):
    """
    Pre-generate descriptors for all test-only datasets

    Args:
        test_datasets: Dictionary of test datasets
        cache_dir: Directory to cache descriptors
    """
    print("\n" + "="*60)
    print("PRE-GENERATING DESCRIPTORS FOR TEST DATASETS")
    print("="*60)

    from extra_code.chem_descriptor_maker import ChemDescriptorCalculator
    calculator = ChemDescriptorCalculator(cache_dir=cache_dir)

    generated_count = 0
    skipped_count = 0

    for test_key, test_data in test_datasets.items():
        descriptor_dir = f"{cache_dir}/testonly_{test_key}"
        test_desc_file = f"{descriptor_dir}/testonly_{test_key}_test_descriptors.npz"

        # Check if already exists
        if os.path.exists(test_desc_file):
            print(f"✓ {test_key}: Descriptors already exist, skipping...")
            skipped_count += 1
            continue

        print(f"\n  📊 Generating descriptors for {test_key}...")
        print(f"   Dataset size: {len(test_data)}")

        try:
            # Get SMILES and molecules
            smiles_list = test_data['smiles'].tolist()
            all_mols = [Chem.MolFromSmiles(s) for s in smiles_list]

            # Filter out None molecules and molecules with problematic elements
            mols = []
            excluded_problematic = 0
            invalid_smiles = 0

            for mol in all_mols:
                if mol is None:
                    invalid_smiles += 1
                elif has_problematic_elements(mol):
                    excluded_problematic += 1
                else:
                    mols.append(mol)

            if invalid_smiles > 0:
                print(f"   ⚠️ Warning: {invalid_smiles} invalid SMILES skipped")
            if excluded_problematic > 0:
                print(f"   ⚠️ Warning: {excluded_problematic} molecules with problematic elements excluded")

            # Calculate all descriptors
            print(f"  📊 Calculating 2048 RDKit descriptors...")
            start_time = time.time()

            calculator.calculate_selected_descriptors(
                mols,
                dataset_name=f'testonly_{test_key}',
                split_type='test',
                subset='test'
            )

            elapsed = time.time() - start_time
            print(f"   ✅ Complete! Time: {elapsed:.2f} seconds")
            print(f"   Saved to: {test_desc_file}")
            generated_count += 1

        except Exception as e:
            print(f"   ❌ Error generating descriptors: {e}")
            continue

    print("\n" + "="*60)
    print(f"DESCRIPTOR GENERATION COMPLETE")
    print(f"  Generated: {generated_count} datasets")
    print(f"  Skipped (existing): {skipped_count} datasets")
    print("="*60 + "\n")

    return generated_count > 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ANO Test-Only Evaluation - Module 9')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Maximum number of samples per test dataset (default: 500)')
    parser.add_argument('--skip-cached', action='store_true',
                        help='Skip evaluation if cached descriptors exist')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific training dataset to evaluate (default: all)')
    parser.add_argument('--split', type=str, default=None,
                        help='Specific split type to evaluate (default: all)')
    parser.add_argument('--pre-generate-descriptors', action='store_true',
                        help='Pre-generate all descriptors for test datasets before evaluation')
    parser.add_argument('--descriptors-only', action='store_true',
                        help='Only generate descriptors without running evaluation')
    args = parser.parse_args()
    
    print("="*80)
    print("[MODULE 9] ANO TEST-ONLY EVALUATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Module Focus: Test-Only Model Evaluation & Validation")
    print("="*80)

    # Check renew setting from config (informational only for evaluation module)
    from config import MODEL_CONFIG
    renew = MODEL_CONFIG.get('renew', False)
    print(f"⚙️  Renew setting: {renew} (Note: Evaluation module doesn't use renew functionality)")
    print("Evaluating Module 8 trained models only")
    if args.max_samples:
        print(f"Limiting test datasets to {args.max_samples} samples each")
    if args.skip_cached:
        print("Skipping datasets with cached descriptors")
    print("=" * 60)
    
    # Get configurations
    if args.dataset:
        training_datasets = [args.dataset]
    else:
        training_datasets = CODE_SPECIFIC_DATASETS.get(8, ['ws', 'de', 'lo', 'hu'])

    if args.split:
        split_types = [args.split]
    else:
        split_types = ACTIVE_SPLIT_TYPES

    modules = ['FO', 'MO', 'FOMO', 'MOFO']

    print(f"\nConfiguration:")
    print(f"  Training datasets: {training_datasets}")
    print(f"  Split types: {split_types}")
    print(f"  Modules: {modules}")
    print(f"  Model directory: {MODEL_DIR}")
    
    # Load test datasets with optional limit
    test_datasets = load_test_datasets(max_samples=args.max_samples)
    if not test_datasets:
        print("\nERROR: No test datasets found!")
        return

    print(f"\nFound {len(test_datasets)} test datasets: {list(test_datasets.keys())}")

    # Skip pre-generating descriptors - too slow and unnecessary
    # Descriptors will be calculated on-demand for selected features only
    if args.pre_generate_descriptors or args.descriptors_only:
        print("⚠️ Skipping descriptor pre-generation for performance (will calculate on-demand)")
        if args.descriptors_only:
            print("Descriptor-only mode disabled for performance.")
            return

    # Results storage
    all_results = {}

    # Process each test dataset
    for test_key, test_df in test_datasets.items():
        print(f"\n{'='*40}")
        print(f"Evaluating on {test_key.upper()} test dataset ({len(test_df)} samples)")
        print(f"{'='*40}")
        
        # Check for cached descriptors if skip_cached is enabled
        if args.skip_cached:
            cache_path = Path(f'result/chemical_descriptors/testonly_{test_key}/testonly_{test_key}_test.npz')
            if cache_path.exists():
                print(f"  Skipping {test_key} - cached descriptors found at {cache_path}")
                continue
        
        test_results = {}
        
        for train_dataset in training_datasets:
            for split_type in split_types:
                for module in modules:
                    model_name = f'{train_dataset}_{split_type}_{module}'
                    model_path = MODEL_DIR / train_dataset / split_type / f'{model_name}_model.pt'
                    
                    if not model_path.exists():
                        print(f"  ✗ Model not found: {model_name}")
                        test_results[model_name] = {
                            'r2': -999,
                            'rmse': 999,
                            'mse': 999999,
                            'mae': 999,
                            'n_samples': 0,
                            'error': 'Model file not found'
                        }
                        continue
                    
                    print(f"\n  Processing {model_name}...")
                    
                    # Prepare features
                    try:
                        X_test, y_test = prepare_features_for_module(test_df, test_key, train_dataset, split_type, module)
                        
                        if X_test is None or y_test is None:
                            print(f"    ✗ Feature preparation failed")
                            test_results[model_name] = {
                                'r2': -999,
                                'rmse': 999,
                                'mse': 999999,
                                'mae': 999,
                                'n_samples': 0,
                                'error': 'Feature preparation failed'
                            }
                            continue
                        
                        print(f"    Features shape: {X_test.shape}")
                        
                        # Evaluate model
                        metrics = evaluate_model_from_module8(model_path, X_test, y_test)
                        
                        if metrics:
                            test_results[model_name] = metrics
                            print(f"    ✓ R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MSE={metrics['mse']:.4f}")
                            
                            # Create scatter plot if we have predictions
                            if 'y_pred' in metrics and 'y_test' in metrics:
                                plot_file = create_scatter_plot(
                                    metrics['y_test'], 
                                    metrics['y_pred'],
                                    model_name,
                                    test_key,
                                    metrics['r2'],
                                    OUTPUT_DIR
                                )
                                print(f"    ✓ Scatter plot saved: {plot_file}")
                        else:
                            test_results[model_name] = {
                                'r2': -999,
                                'rmse': 999,
                                'mse': 999999,
                                'mae': 999,
                                'n_samples': 0,
                                'error': 'Model evaluation failed'
                            }
                            print(f"    ✗ Evaluation failed")
                            
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        test_results[model_name] = {
                            'r2': -999,
                            'rmse': 999,
                            'mse': 999999,
                            'mae': 999,
                            'n_samples': 0,
                            'error': str(e)
                        }
        
        all_results[test_key] = test_results
    
    # Save results
    results_file = OUTPUT_DIR / 'testonly_evaluation_results.json'
    
    # Remove y_pred and y_test from results before saving (to avoid JSON serialization issues)
    clean_results = {}
    for test_key, test_results in all_results.items():
        clean_results[test_key] = {}
        for model_name, metrics in test_results.items():
            clean_metrics = {k: v for k, v in metrics.items() if k not in ['y_pred', 'y_test']}
            clean_results[test_key][model_name] = clean_metrics
    
    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    # Save as CSV
    csv_data = []
    for test_dataset, test_results in clean_results.items():
        for model_name, metrics in test_results.items():
            parts = model_name.split('_')
            row = {
                'test_dataset': test_dataset,
                'train_dataset': parts[0] if len(parts) > 0 else '',
                'split_type': parts[1] if len(parts) > 1 else '',
                'module': parts[2] if len(parts) > 2 else '',
                'r2': metrics.get('r2', -999),
                'rmse': metrics.get('rmse', 999),
                'mse': metrics.get('mse', 999999),
                'mae': metrics.get('mae', 999),
                'n_samples': metrics.get('n_samples', 0),
                'error': metrics.get('error', '')
            }
            csv_data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / 'testonly_evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    # Print summary
    for test_dataset, results in all_results.items():
        if results:
            print(f"\n{test_dataset.upper()} Test Dataset:")
            print("-" * 40)
            
            # Filter out failed evaluations
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                # Sort by R² score
                sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['r2'], reverse=True)
                
                # Show top 5 models
                print("Top 5 models:")
                for i, (model_name, metrics) in enumerate(sorted_results[:5], 1):
                    print(f"  {i}. {model_name}:")
                    print(f"     R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}, MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}")
                
                # Best model
                if sorted_results:
                    best = sorted_results[0]
                    print(f"\nBest model: {best[0]} (R²={best[1]['r2']:.4f})")
            else:
                print("  No successful evaluations!")
            
            # Report failures
            failed = [k for k, v in results.items() if 'error' in v]
            if failed:
                print(f"\nFailed evaluations: {len(failed)}/{len(results)}")
                for model_name in failed[:5]:  # Show first 5 failures
                    print(f"  - {model_name}: {results[model_name]['error']}")
    
    print(f"\nResults saved to: {results_file}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    # Set up logging
    module_name = MODULE_NAMES.get('9', '9_ANO_testonly_evaluation')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get dataset from command line arguments or use all datasets
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
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
    
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
        def close(self):
            self.log.close()
    
    logger = Logger(str(log_file))
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        print(f"Log file: {log_file}")
        main()
    finally:
        logger.close()
        sys.stdout = logger.terminal
        sys.stderr = sys.__stderr__