#!/usr/bin/env python3
"""
Complete Molecular Data Processing Pipeline
==========================================

Molecular fingerprint generation and combination pipeline with
fully configurable parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional
import warnings
import gc
import os
from tqdm import tqdm
from itertools import combinations

# RDKit imports
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools

warnings.filterwarnings('ignore')

# ===========================
# Default Configuration
# ===========================

# Default fingerprint types
DEFAULT_FINGERPRINT_TYPES = ["morgan", "maccs", "avalon"]

# Default datasets
DEFAULT_DATASETS = ["ws", "de", "lo", "hu"]

# Default splits
DEFAULT_SPLITS = ["rm", "sc", "cs", "cl", "pc", "ac", "sa", "ti", "en"]

# Default dataset mapping
DEFAULT_DATASET_MAP = {
    "delaney-processed": "de",
    "Lovric2020_logS0": "lo",
    "ws496_logS": "ws",
    "huusk": "hu"
}

# Default split mapping
DEFAULT_SPLIT_MAP = {
    'rm': 'random',
    'sc': 'scaffold',
    'cs': 'chemical_space_coverage',
    'cl': 'cluster',
    'pc': 'physchem',
    'ac': 'activity_cliff',
    'sa': 'solubility_aware',
    'ti': 'time_series',
    'en': 'ensemble',
}

# Default fingerprint sizes
DEFAULT_FP_SIZES = {
    'morgan': 2048,
    'maccs': 167,
    'avalon': 512
}

# Column names to search for
DEFAULT_SMILES_COLUMNS = ['SMILES', 'smiles', 'Smiles', 'SMILE', 'smile', 'molecule']
DEFAULT_TARGET_COLUMNS = [
    'target', 'Target', 'logS', 'LogS', 'logs', 'LogS0',
    'measured log solubility in mols per litre',
    'y', 'Y', 'label', 'Label', 'activity', 'Activity'
]

# ===========================
# Helper Functions
# ===========================

def generate_fp_combos(fingerprint_types: List[str]) -> List[Tuple[str, ...]]:
    """
    Generate all possible combinations of fingerprint types
    
    Parameters:
    -----------
    fingerprint_types : list
        List of fingerprint types
    
    Returns:
    --------
    fp_combos : list of tuples
        All possible combinations
    """
    fp_combos = []
    for r in range(1, len(fingerprint_types) + 1):
        fp_combos.extend(combinations(fingerprint_types, r))
    return fp_combos

# ===========================
# Data Loading Functions
# ===========================

def load_split_data(base_dir: str, 
                   dataset_map: Dict[str, str] = None,
                   split_map: Dict[str, str] = None,
                   splits: List[str] = None,
                   use_parallel: bool = True) -> Dict:
    """
    Load train/test data from preprocessed split directories
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing preprocessed data
    dataset_map : dict, optional
        Mapping of full dataset names to abbreviations (uses DEFAULT_DATASET_MAP if None)
    split_map : dict, optional
        Mapping of split abbreviations to full names (uses DEFAULT_SPLIT_MAP if None)
    splits : list, optional
        List of splits to load (uses all splits in split_map if None)
    use_parallel : bool
        Whether to use parallel loading
    
    Returns:
    --------
    data_dict : dict
        Nested dictionary containing loaded DataFrames
    """
    if dataset_map is None:
        dataset_map = DEFAULT_DATASET_MAP
    if split_map is None:
        split_map = DEFAULT_SPLIT_MAP
    if splits is None:
        splits = list(split_map.keys())
    
    BASE = Path(base_dir)
    data_dict = {}
    
    # Filter split_map to only include requested splits
    active_split_map = {k: v for k, v in split_map.items() if k in splits}
    
    if use_parallel:
        max_workers = min(mp.cpu_count(), 8)
        file_tasks = []
        
        for split_abbr, split_name in active_split_map.items():
            data_dict[split_abbr] = {}
            
            for phase in ["train", "test"]:
                phase_dir = BASE / phase / split_abbr
                if not phase_dir.exists():
                    continue
                    
                data_dict[split_abbr][phase] = {}
                csv_files = list(phase_dir.glob("*.csv"))
                
                for csv_file in csv_files:
                    file_tasks.append((split_abbr, phase, csv_file))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for split_abbr, phase, csv_file in file_tasks:
                future = executor.submit(_load_csv_file, csv_file, dataset_map)
                futures.append((future, split_abbr, phase))
            
            for future, split_abbr, phase in futures:
                result = future.result()
                if result is not None:
                    abbr, df = result
                    data_dict[split_abbr][phase][abbr] = df
    else:
        # Sequential loading
        for split_abbr, split_name in active_split_map.items():
            data_dict[split_abbr] = {}
            
            for phase in ["train", "test"]:
                phase_dir = BASE / phase / split_abbr
                if not phase_dir.exists():
                    continue
                    
                data_dict[split_abbr][phase] = {}
                csv_files = list(phase_dir.glob("*.csv"))
                
                for csv_file in csv_files:
                    result = _load_csv_file(csv_file, dataset_map)
                    if result is not None:
                        abbr, df = result
                        data_dict[split_abbr][phase][abbr] = df
    
    return data_dict

def _load_csv_file(csv_file: Path, dataset_map: Dict[str, str]) -> Optional[Tuple[str, pd.DataFrame]]:
    """Helper function to load a single CSV file"""
    filename = csv_file.stem
    
    for full_name, abbr in dataset_map.items():
        if full_name in filename:
            try:
                df = pd.read_csv(csv_file)
                return abbr, df
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                return None
    return None

# ===========================
# Data Extraction Functions
# ===========================

def extract_xy_from_data(data_dict: Dict, 
                        datasets: List[str] = None,
                        smiles_columns: List[str] = None,
                        target_columns: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Extract SMILES (X) and target values (y) from loaded data
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing loaded DataFrames
    datasets : list, optional
        List of dataset abbreviations to extract (extracts all if None)
    smiles_columns : list, optional
        List of column names to search for SMILES (uses DEFAULT_SMILES_COLUMNS if None)
    target_columns : list, optional
        List of column names to search for targets (uses DEFAULT_TARGET_COLUMNS if None)
    
    Returns:
    --------
    x_map, y_map : tuple of dicts
        Dictionaries containing SMILES and target values
    """
    if smiles_columns is None:
        smiles_columns = DEFAULT_SMILES_COLUMNS
    if target_columns is None:
        target_columns = DEFAULT_TARGET_COLUMNS
    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    x_map = {}
    y_map = {}
    
    for split_abbr, phases in data_dict.items():
        x_map[split_abbr] = {}
        y_map[split_abbr] = {}
        
        for phase, phase_datasets in phases.items():
            for dataset_abbr, df in phase_datasets.items():
                # Skip if dataset not in requested list
                if datasets and dataset_abbr not in datasets:
                    continue
                    
                key = f"{dataset_abbr}_{phase}"
                
                # Extract SMILES column
                smiles_col = None
                for col in smiles_columns:
                    if col in df.columns:
                        smiles_col = col
                        break
                
                if smiles_col:
                    x_map[split_abbr][key] = df[smiles_col].tolist()
                else:
                    print(f"Warning: No SMILES column found in {split_abbr}/{phase}/{dataset_abbr}")
                    print(f"  Available columns: {list(df.columns)}")
                    continue
                
                # Extract target column
                target_col = None
                for col in target_columns:
                    if col in df.columns:
                        target_col = col
                        break
                
                if target_col:
                    y_map[split_abbr][key] = df[target_col].astype(float).tolist()
                else:
                    # Try to find any column with 'log' in name as fallback
                    log_cols = [col for col in df.columns if 'log' in col.lower()]
                    if log_cols:
                        y_map[split_abbr][key] = df[log_cols[0]].astype(float).tolist()
                        print(f"Info: Using '{log_cols[0]}' as target column for {split_abbr}/{phase}/{dataset_abbr}")
                    else:
                        print(f"Warning: No target column found in {split_abbr}/{phase}/{dataset_abbr}")
                        print(f"  Available columns: {list(df.columns)}")
    
    return x_map, y_map

# ===========================
# Fingerprint Generation Functions
# ===========================

def get_fingerprints(mol, fp_sizes: Dict[str, int] = None, fingerprint_types: List[str] = None):
    """
    Generate fingerprints for a single molecule
    
    Parameters:
    -----------
    mol : RDKit molecule
        Molecule object
    fp_sizes : dict, optional
        Dictionary of fingerprint sizes
    fingerprint_types : list, optional
        List of fingerprint types to generate
    
    Returns:
    --------
    fingerprints : dict
        Dictionary of fingerprint arrays
    """
    if fp_sizes is None:
        fp_sizes = DEFAULT_FP_SIZES
    if fingerprint_types is None:
        fingerprint_types = DEFAULT_FINGERPRINT_TYPES
    
    fingerprints = {}
    
    if 'morgan' in fingerprint_types:
        gen = GetMorganGenerator(radius=2, fpSize=fp_sizes.get('morgan', 2048))
        bv = gen.GetFingerprint(mol)
        morgan = np.zeros(fp_sizes.get('morgan', 2048), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bv, morgan)
        fingerprints['morgan'] = morgan
    
    if 'maccs' in fingerprint_types:
        maccs_bits = MACCSkeys.GenMACCSKeys(mol)
        maccs = np.zeros(fp_sizes.get('maccs', 167), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(maccs_bits, maccs)
        fingerprints['maccs'] = maccs
    
    if 'avalon' in fingerprint_types:
        av_bits = pyAvalonTools.GetAvalonFP(mol, nBits=fp_sizes.get('avalon', 512))
        avalon = np.zeros(fp_sizes.get('avalon', 512), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(av_bits, avalon)
        fingerprints['avalon'] = avalon
    
    return fingerprints

def build_fingerprints_for_splits(x_map: Dict,
                                out_root: Path,
                                fingerprint_types: List[str] = None,
                                fp_sizes: Dict[str, int] = None,
                                n_jobs: int = 16,
                                force_rebuild: bool = False,
                                use_cache: bool = True) -> Dict:
    """
    Build fingerprints for all splits with caching
    
    Parameters:
    -----------
    x_map : dict
        Dictionary containing SMILES data
    out_root : Path
        Output directory for fingerprint cache files
    fingerprint_types : list, optional
        List of fingerprint types to generate (uses DEFAULT_FINGERPRINT_TYPES if None)
    fp_sizes : dict, optional
        Dictionary of fingerprint sizes (uses DEFAULT_FP_SIZES if None)
    n_jobs : int
        Number of parallel jobs
    force_rebuild : bool
        If True, ignore existing cache and rebuild
    use_cache : bool
        If True, use caching system
    
    Returns:
    --------
    fp_map : dict
        Dictionary containing fingerprints for each dataset
    """
    if fingerprint_types is None:
        fingerprint_types = DEFAULT_FINGERPRINT_TYPES
    if fp_sizes is None:
        fp_sizes = DEFAULT_FP_SIZES
    
    if not isinstance(out_root, Path):
        out_root = Path(out_root)
    
    cache_dir = out_root
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    fp_map = {}
    total_tasks = sum(len(datasets) for datasets in x_map.values())
    
    with tqdm(total=total_tasks, desc="Generating fingerprints") as pbar:
        for split_abbr, datasets in x_map.items():
            fp_map[split_abbr] = {}
            
            for key, smiles_list in datasets.items():
                cache_file = cache_dir / f"{split_abbr}_{key}.npz"
                
                # Check cache
                if use_cache and not force_rebuild and cache_file.exists():
                    try:
                        data = np.load(cache_file)
                        
                        # Check if all requested fingerprint types are in cache
                        if all(fp_type in data for fp_type in fingerprint_types):
                            if len(data[fingerprint_types[0]]) == len(smiles_list):
                                fp_map[split_abbr][key] = {
                                    fp_type: data[fp_type] for fp_type in fingerprint_types
                                }
                                pbar.update(1)
                                continue
                    except Exception as e:
                        print(f"Cache load failed: {cache_file.name}, error: {e}")
                
                # Generate fingerprints
                try:
                    # Initialize arrays
                    fp_data = {
                        fp_type: np.zeros((len(smiles_list), fp_sizes.get(fp_type, DEFAULT_FP_SIZES[fp_type])), dtype=np.uint8)
                        for fp_type in fingerprint_types
                    }
                    
                    # Process molecules
                    for i, smi in enumerate(smiles_list):
                        try:
                            mol = Chem.MolFromSmiles(smi)
                            if mol is not None:
                                fps = get_fingerprints(mol, fp_sizes, fingerprint_types)
                                for fp_type, fp_array in fps.items():
                                    fp_data[fp_type][i] = fp_array
                        except Exception as e:
                            print(f"Error processing SMILES at index {i}: {e}")
                    
                    # Save cache
                    if use_cache:
                        try:
                            np.savez_compressed(cache_file, **fp_data)
                        except Exception as e:
                            print(f"Cache save failed: {e}")
                    
                    fp_map[split_abbr][key] = fp_data
                    
                except Exception as e:
                    print(f"Fingerprint generation failed ({split_abbr}/{key}): {e}")
                    # Generate empty fingerprints
                    fp_map[split_abbr][key] = {
                        fp_type: np.zeros((len(smiles_list), fp_sizes.get(fp_type, DEFAULT_FP_SIZES[fp_type])), dtype=np.uint8)
                        for fp_type in fingerprint_types
                    }
                
                pbar.update(1)
                gc.collect()
    
    return fp_map

# ===========================
# Fingerprint Combination Functions
# ===========================

def create_all_fp_combinations(fp_map: Dict,
                             fingerprint_types: List[str] = None,
                             fp_combos: List[Tuple[str, ...]] = None) -> Dict:
    """
    Create all possible fingerprint combinations
    
    Parameters:
    -----------
    fp_map : dict
        Original fingerprint map
    fingerprint_types : list, optional
        List of fingerprint types to combine
    fp_combos : list, optional
        List of specific combinations to create (generates all if None)
    
    Returns:
    --------
    all_combinations : dict
        Dictionary with all fingerprint combinations
    """
    if fingerprint_types is None:
        # Detect available fingerprint types from first entry
        first_split = list(fp_map.keys())[0]
        first_key = list(fp_map[first_split].keys())[0]
        fingerprint_types = list(fp_map[first_split][first_key].keys())
    
    # Generate all possible combinations if not provided
    if fp_combos is None:
        fp_combos = generate_fp_combos(fingerprint_types)
    
    all_combinations = {}
    
    for combo in fp_combos:
        combo_name = '+'.join(combo)
        print(f"Creating combination: {combo_name}")
        
        try:
            combined_map = {}
            
            for split_abbr, datasets in fp_map.items():
                combined_map[split_abbr] = {}
                
                for key, fp_data in datasets.items():
                    # Combine fingerprints
                    fps_to_combine = []
                    for fp_type in combo:
                        if fp_type in fp_data:
                            fps_to_combine.append(fp_data[fp_type])
                    
                    if len(fps_to_combine) == 1:
                        combined_fp = fps_to_combine[0].copy()
                    else:
                        combined_fp = np.concatenate(fps_to_combine, axis=1)
                    
                    combined_map[split_abbr][key] = {
                        'combined': combined_fp,
                        'fp_types': combo,
                        'shape': combined_fp.shape
                    }
            
            all_combinations[combo_name] = combined_map
            
        except Exception as e:
            print(f"  Error creating {combo_name}: {e}")
    
    return all_combinations

def get_train_test_data(all_combinations: Dict,
                       y_map: Dict,
                       combination_name: str,
                       split_abbr: str,
                       dataset_abbr: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """
    Get train and test data for a specific combination
    
    Parameters:
    -----------
    all_combinations : dict
        Dictionary with all fingerprint combinations
    y_map : dict
        Target values map
    combination_name : str
        Name of the combination (e.g., 'morgan+maccs')
    split_abbr : str
        Split abbreviation (e.g., 'rm')
    dataset_abbr : str
        Dataset abbreviation (e.g., 'de')
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Training and test data
    """
    train_key = f"{dataset_abbr}_train"
    test_key = f"{dataset_abbr}_test"
    
    if combination_name not in all_combinations:
        raise KeyError(f"Combination '{combination_name}' not found. Available: {list(all_combinations.keys())}")
    
    if split_abbr not in all_combinations[combination_name]:
        raise KeyError(f"Split '{split_abbr}' not found in combination '{combination_name}'. Available: {list(all_combinations[combination_name].keys())}")
    
    if train_key not in all_combinations[combination_name][split_abbr]:
        raise KeyError(f"Train dataset '{train_key}' not found. Available: {list(all_combinations[combination_name][split_abbr].keys())}")
    
    if test_key not in all_combinations[combination_name][split_abbr]:
        raise KeyError(f"Test dataset '{test_key}' not found. Available: {list(all_combinations[combination_name][split_abbr].keys())}")
    
    X_train = all_combinations[combination_name][split_abbr][train_key]['combined']
    X_test = all_combinations[combination_name][split_abbr][test_key]['combined']
    
    y_train = y_map[split_abbr][train_key]
    y_test = y_map[split_abbr][test_key]
    
    return X_train, X_test, y_train, y_test

def create_summary_dataframe(all_combinations: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame of all fingerprint combinations
    
    Parameters:
    -----------
    all_combinations : dict
        Dictionary with all fingerprint combinations
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary DataFrame
    """
    summary_data = []
    
    for combo_name, combined_map in all_combinations.items():
        for split_abbr in combined_map:
            for key in combined_map[split_abbr]:
                data = combined_map[split_abbr][key]
                
                # Parse dataset and phase from key
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    dataset, phase = parts
                else:
                    dataset, phase = key, 'unknown'
                
                summary_data.append({
                    'combination': combo_name,
                    'split': split_abbr,
                    'dataset': dataset,
                    'phase': phase,
                    'dataset_key': key,
                    'n_molecules': data['shape'][0],
                    'n_features': data['shape'][1],
                    'fp_types': ', '.join(data['fp_types'])
                })
    
    return pd.DataFrame(summary_data)

# ===========================
# Main Pipeline Function
# ===========================

def run_pipeline(base_dir: str = "result/1_preprocess",
                out_root: str = "result/2_fingerprint",
                dataset_map: Dict[str, str] = None,
                split_map: Dict[str, str] = None,
                datasets: List[str] = None,
                splits: List[str] = None,
                fingerprint_types: List[str] = None,
                fp_combos: List[Tuple[str, ...]] = None,
                fp_sizes: Dict[str, int] = None,
                smiles_columns: List[str] = None,
                target_columns: List[str] = None,
                n_jobs: int = 16,
                force_rebuild: bool = False,
                use_cache: bool = True,
                create_combinations: bool = True) -> Tuple:
    """
    Run the complete molecular data processing pipeline
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing preprocessed data
    out_root : str
        Output directory for fingerprints
    dataset_map : dict, optional
        Mapping of dataset names to abbreviations
    split_map : dict, optional
        Mapping of split abbreviations to names
    datasets : list, optional
        List of dataset abbreviations to process
    splits : list, optional
        List of split abbreviations to process
    fingerprint_types : list, optional
        List of fingerprint types to generate
    fp_combos : list, optional
        List of specific fingerprint combinations to create
    fp_sizes : dict, optional
        Dictionary of fingerprint sizes
    smiles_columns : list, optional
        Column names to search for SMILES
    target_columns : list, optional
        Column names to search for targets
    n_jobs : int
        Number of parallel jobs
    force_rebuild : bool
        If True, rebuild cache even if it exists
    use_cache : bool
        If True, use caching system
    create_combinations : bool
        If True, create all fingerprint combinations
    
    Returns:
    --------
    data_dict, x_map, y_map, fp_map, all_combinations : tuple
        All pipeline outputs
    """
    # Use defaults if not provided
    if dataset_map is None:
        dataset_map = DEFAULT_DATASET_MAP
    if split_map is None:
        split_map = DEFAULT_SPLIT_MAP
    if datasets is None:
        datasets = DEFAULT_DATASETS
    if splits is None:
        splits = DEFAULT_SPLITS
    if fingerprint_types is None:
        fingerprint_types = DEFAULT_FINGERPRINT_TYPES
    if fp_sizes is None:
        fp_sizes = DEFAULT_FP_SIZES
    if smiles_columns is None:
        smiles_columns = DEFAULT_SMILES_COLUMNS
    if target_columns is None:
        target_columns = DEFAULT_TARGET_COLUMNS
    
    print("=== Molecular Data Processing Pipeline ===\n")
    print(f"Configuration:")
    print(f"  Datasets: {datasets}")
    print(f"  Splits: {splits}")
    print(f"  Fingerprint types: {fingerprint_types}")
    print(f"  Fingerprint sizes: {fp_sizes}")
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    data_dict = load_split_data(base_dir, dataset_map, split_map, splits)
    print(f"Loaded {len(data_dict)} splits")
    
    # Step 2: Extract X and y
    print("\nStep 2: Extracting SMILES and targets...")
    x_map, y_map = extract_xy_from_data(data_dict, datasets, smiles_columns, target_columns)
    
    # Count extracted data
    total_datasets = sum(len(split_data) for split_data in x_map.values())
    print(f"Extracted {total_datasets} dataset-phase combinations")
    
    # Step 3: Generate fingerprints
    print("\nStep 3: Generating fingerprints...")
    fp_map = build_fingerprints_for_splits(
        x_map, Path(out_root), fingerprint_types, fp_sizes,
        n_jobs, force_rebuild, use_cache
    )
    
    # Step 4: Create combinations
    all_combinations = None
    if create_combinations:
        print("\nStep 4: Creating fingerprint combinations...")
        
        # Generate FP_COMBOS if not provided
        if fp_combos is None:
            fp_combos = generate_fp_combos(fingerprint_types)
            print(f"Generated {len(fp_combos)} combinations from {fingerprint_types}")
        
        all_combinations = create_all_fp_combinations(fp_map, fingerprint_types, fp_combos)
        
        # Create summary
        summary_df = create_summary_dataframe(all_combinations)
        print(f"\nCreated {len(all_combinations)} combinations")
        
        # Show feature dimensions
        print("\nFeature dimensions by combination:")
        combo_dims = summary_df[summary_df['phase'] == 'train'].groupby('combination')['n_features'].first()
        for combo, dims in combo_dims.items():
            print(f"  {combo}: {dims} features")
    
    print("\n=== Pipeline Complete ===")
    
    return data_dict, x_map, y_map, fp_map, all_combinations

# ===========================
# Convenience Functions
# ===========================

def get_fp_combos(fingerprint_types: List[str] = None) -> List[Tuple[str, ...]]:
    """
    Get all fingerprint combinations for given types
    
    Parameters:
    -----------
    fingerprint_types : list, optional
        List of fingerprint types (uses DEFAULT_FINGERPRINT_TYPES if None)
    
    Returns:
    --------
    fp_combos : list of tuples
        All possible combinations
    """
    if fingerprint_types is None:
        fingerprint_types = DEFAULT_FINGERPRINT_TYPES
    return generate_fp_combos(fingerprint_types)

def print_configuration():
    """Print default configuration"""
    print("=== Default Configuration ===")
    print(f"\nFingerprint Types: {DEFAULT_FINGERPRINT_TYPES}")
    print(f"\nDatasets: {DEFAULT_DATASETS}")
    print(f"Dataset Mapping: {DEFAULT_DATASET_MAP}")
    print(f"\nSplits: {DEFAULT_SPLITS}")
    print(f"Split Mapping: {DEFAULT_SPLIT_MAP}")
    print(f"\nFingerprint Sizes: {DEFAULT_FP_SIZES}")
    print(f"\nSMILES Columns: {DEFAULT_SMILES_COLUMNS}")
    print(f"\nTarget Columns: {DEFAULT_TARGET_COLUMNS[:5]} ...")
    
    fp_combos = get_fp_combos()
    print(f"\nFingerprint Combinations ({len(fp_combos)}):")
    for combo in fp_combos:
        print(f"  {'+'.join(combo)}")

# ===========================
# Example Usage
# ===========================

if __name__ == "__main__":
    # Print configuration
    print_configuration()
    
    # Example: Run with custom parameters
    print("\n\n=== Example: Running with custom parameters ===")
    
    # Custom configuration
    custom_fingerprints = ["morgan", "maccs"]
    custom_datasets = ["de", "ws"]
    custom_splits = ["rm", "sc"]
    
    # Run pipeline
    results = run_pipeline(
        base_dir="result/1_preprocess",
        out_root="result/2_fingerprint",
        datasets=custom_datasets,
        splits=custom_splits,
        fingerprint_types=custom_fingerprints,
        create_combinations=True
    )
    
    data_dict, x_map, y_map, fp_map, all_combinations = results
    
    # Show results
    if all_combinations:
        print("\n=== Results ===")
        summary_df = create_summary_dataframe(all_combinations)
        print(f"Total entries: {len(summary_df)}")
        print("\nFirst 5 entries:")
        print(summary_df.head())