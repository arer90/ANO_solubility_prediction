#!/usr/bin/env python3
"""
Developer: Lee, Seungjin (arer90)

Complete Molecular Data Processing Pipeline
==========================================

PURPOSE:
This module provides comprehensive molecular data processing capabilities including
fingerprint generation, data loading, and molecular descriptor calculation for
solubility prediction tasks.

KEY FEATURES:
- Multiple molecular fingerprint types (Morgan, MACCS, Avalon)
- Efficient parallel processing for large datasets
- 3D conformer generation and filtering
- Support for various data splitting strategies
- Memory-efficient batch processing

FINGERPRINT TYPES:
1. Morgan (ECFP): 2048-bit circular fingerprints capturing local environments
2. MACCS: 167-bit structural key-based fingerprints
3. Avalon: 512-bit fingerprints with good general purpose performance

USAGE:
- Load molecular data with automatic SMILES validation
- Generate fingerprints for machine learning models
- Prefilter molecules for 3D descriptor calculation
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
from rdkit.Chem import DataStructs, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools

warnings.filterwarnings('ignore')

def load_split_data(base_path: str = "result/1_preprocess") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load preprocessed train/test split data from the 1_preprocess directory.
    
    This function loads molecular datasets that have been split using various
    strategies (random, scaffold, chemical space, etc.) for robust model evaluation.
    Each split ensures proper separation between training and test sets to avoid
    data leakage and provide realistic performance estimates.
    
    Split Types:
    - rm: Random split (baseline)
    - sc: Scaffold split (tests generalization to new scaffolds)
    - cs: Chemical space coverage (ensures diverse test set)
    - cl: Cluster-based split (groups similar molecules)
    - pc: Physicochemical property-based split
    - ac: Activity cliff-aware split (challenging cases)
    - sa: Solubility-aware split (stratified by solubility)
    - ti: Time series split (temporal validation)
    - en: Ensemble split (combined strategies)
    
    Parameters:
    -----------
    base_path : str
        Base path to the preprocessing results directory (default: "result/1_preprocess")
    
    Returns:
    --------
    data_dict : dict
        Nested dictionary with structure:
        {split_type: {dataset_phase: DataFrame}}
        Example: {'rm': {'de_train': df, 'de_test': df, 'ws_train': df, ...}}
        
        Dataset abbreviations:
        - de: delaney-processed
        - lo: Lovric2020_logS0
        - ws: ws496_logS
        - hu: huusk
    """
    base_path = Path(base_path)
    train_dir = base_path / "train"
    test_dir = base_path / "test"
    
    data_dict = {}
    
    # Define expected splits with numbering (only those with actual data)
    split_types = [
        'rm',  # 1. random
        'sc',  # 2. scaffold  
        'cs',  # 3. chemical_space_coverage
        'cl',  # 4. cluster
        'pc',  # 5. physchem
        'ac',  # 6. activity_cliff
        'sa',  # 7. solubility_aware
        'ti',  # 8. time_series
        'en'   # 9. ensemble
    ]
    
    # Dataset mapping from full names to abbreviations
    dataset_map = {
        "delaney-processed": "de",
        "Lovric2020_logS0": "lo", 
        "ws496_logS": "ws",
        "huusk": "hu"
    }
    
    for split_type in split_types:
        data_dict[split_type] = {}
        
        # Check if split directory exists
        split_train_dir = train_dir / split_type
        split_test_dir = test_dir / split_type
        
        if split_train_dir.exists():
            # Load all CSV files in the split directory
            for csv_file in split_train_dir.glob("*.csv"):
                filename = csv_file.stem
                # Extract dataset name from filename (e.g., "rm_delaney-processed_train")
                parts = filename.split('_')
                if len(parts) >= 3 and parts[-1] == 'train':
                    # Reconstruct the dataset name (handle names with underscores)
                    dataset_full = '_'.join(parts[1:-1])
                    
                    # Map to abbreviation
                    dataset_abbr = dataset_map.get(dataset_full)
                    if dataset_abbr:
                        try:
                            df = pd.read_csv(csv_file)
                            data_dict[split_type][f"{dataset_abbr}_train"] = df
                        except Exception as e:
                            print(f"Error loading {csv_file}: {e}")
        
        if split_test_dir.exists():
            # Load all CSV files in the test directory
            for csv_file in split_test_dir.glob("*.csv"):
                filename = csv_file.stem
                # Extract dataset name from filename
                parts = filename.split('_')
                if len(parts) >= 3 and parts[-1] == 'test':
                    # Reconstruct the dataset name
                    dataset_full = '_'.join(parts[1:-1])
                    
                    # Map to abbreviation
                    dataset_abbr = dataset_map.get(dataset_full)
                    if dataset_abbr:
                        try:
                            df = pd.read_csv(csv_file)
                            data_dict[split_type][f"{dataset_abbr}_test"] = df
                        except Exception as e:
                            print(f"Error loading {csv_file}: {e}")
    
    # Print summary (minimal)
    total_files = 0
    for split_type, datasets in data_dict.items():
        if datasets:
            total_files += len(datasets)
    print(f"Loaded {total_files} files from {len([s for s in split_types if data_dict[s]])} splits")
    
    return data_dict

def extract_xy_from_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[Dict, Dict]:
    """
    Extract SMILES (X) and target values (y) from loaded data
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames from load_split_data
    
    Returns:
    --------
    x_map, y_map : tuple of dicts
        x_map: {split_type: {dataset_phase: smiles_list}}
        y_map: {split_type: {dataset_phase: target_array}}
    """
    x_map = {}
    y_map = {}
    
    # Possible column names for SMILES
    smiles_cols = ['SMILES', 'smiles', 'Smiles', 'target_x', 'molecule']
    # Possible column names for target values  
    target_cols = ['target', 'target_y', 'logS', 'LogS', 'logs', 'LogS0', 
                   'measured log solubility in mols per litre', 'y', 'Y', 
                   'label', 'Label', 'activity', 'Activity']
    
    # Debug info
    total_extracted = 0
    
    for split_type, datasets in data_dict.items():
        x_map[split_type] = {}
        y_map[split_type] = {}
        
        for key, df in datasets.items():
            if not isinstance(df, pd.DataFrame):
                print(f"Warning: {split_type}/{key} is not a DataFrame, skipping")
                continue
                
            # Find SMILES column
            smiles_col = None
            for col in smiles_cols:
                if col in df.columns:
                    smiles_col = col
                    break
            
            # Find target column
            target_col = None
            for col in target_cols:
                if col in df.columns:
                    target_col = col
                    break
            
            if smiles_col and target_col:
                # Extract data and handle any issues
                try:
                    smiles_list = df[smiles_col].astype(str).tolist()
                    target_values = df[target_col].astype(float).values
                    
                    # Remove any NaN values
                    valid_indices = ~pd.isna(target_values)
                    smiles_list = [s for i, s in enumerate(smiles_list) if valid_indices[i]]
                    target_values = target_values[valid_indices]
                    
                    x_map[split_type][key] = smiles_list
                    y_map[split_type][key] = target_values
                    total_extracted += len(smiles_list)
                    
                except Exception as e:
                    print(f"Error extracting data from {split_type}/{key}: {e}")
                    continue
            else:
                print(f"Warning: Could not find required columns in {split_type}/{key}")
                print(f"  Available columns: {list(df.columns)}")
                print(f"  Looking for SMILES in: {smiles_cols}")
                print(f"  Looking for target in: {target_cols}")
    
    print(f"Extracted {total_extracted} samples from {len(x_map)} splits")
    
    return x_map, y_map

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

# The complex version was removed to avoid conflicts with the simpler load_split_data defined above

# ===========================
# Fingerprint Generation Functions
# ===========================

def prefilter_3d_conformers(smiles_list, y_list):
    """Pre-filter molecules that can generate 3D conformers"""
    print("Pre-filtering molecules for 3D conformer generation...")
    
    # Convert SMILES to mol objects
    mols = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
        else:
            print(f"  Warning: Invalid SMILES at index {i}: {smiles}")
    
    print(f"Valid molecules: {len(mols)} out of {len(smiles_list)}")
    
    # Generate 3D conformers
    mols_3d = process_molecules_parallel(mols, max_workers=8)
    
    if mols_3d:
        print(f"Attempted 3D conformer generation for {len(mols_3d)} molecules")
        
        # Separate molecules that can and cannot generate 3D conformers
        successful_indices = []
        failed_indices = []
        successful_mols = []
        successful_mols_3d = []
        successful_smiles = []
        successful_y = []
        
        for i, mol in enumerate(mols_3d):
            original_index = valid_indices[i]
            if mol is not None and mol.GetNumConformers() > 0:
                successful_indices.append(original_index)
                successful_mols.append(mols[i])
                successful_mols_3d.append(mol)
                successful_smiles.append(smiles_list[original_index])
                successful_y.append(y_list[original_index])
            else:
                failed_indices.append(original_index)
        
        print(f"3D conformer generation successful: {len(successful_mols)} molecules")
        print(f"3D conformer generation failed: {len(failed_indices)} molecules")
        
        return successful_smiles, successful_y, successful_mols, successful_mols_3d
    else:
        print("No 3D conformers generated, returning original data")
        return smiles_list, y_list, mols, []

def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    """Process molecules in parallel for 3D conformer generation"""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    
    def mol3d_safe(mol, timeout_seconds=30):
        """Generate 3D conformer with timeout to prevent infinite loops"""
        result = {'mol': None, 'error': None}
        
        def generate_conformer():
            try:
                # Check for unsupported elements
                unsupported_elements = {
                    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th',
                    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                    'Se', 'Te', 'Rn', 'Xe', 'Kr', 'He', 'Ne', 'Ar'
                }
                
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in unsupported_elements:
                        result['error'] = f"Unsupported elements"
                        return
                
                # Generate 3D conformer
                mol_copy = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_copy)
                result['mol'] = mol_copy
                
            except Exception as e:
                result['error'] = str(e)
        
        # Run with timeout
        thread = threading.Thread(target=generate_conformer)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            result['error'] = "Timeout"
        
        return result['mol'], result['error']
    
    # Process molecules in parallel
    mols_3d = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(mol3d_safe, mol) for mol in mols]
        
        for future in futures:
            try:
                mol_3d, error = future.result()
                if error:
                    print(f"  Warning: 3D conformer generation failed: {error}")
                mols_3d.append(mol_3d)
            except Exception as e:
                print(f"  Warning: 3D conformer generation failed: {e}")
                mols_3d.append(None)
    
    return mols_3d

def get_fingerprints(mols, fp_sizes: Dict[str, int] = None, fingerprint_types: List[str] = None):
    """
    Generate fingerprints for a list of molecules
    
    Parameters:
    -----------
    mols : list of RDKit molecules
        List of molecule objects
    fp_sizes : dict, optional
        Dictionary of fingerprint sizes
    fingerprint_types : list, optional
        List of fingerprint types to generate
    
    Returns:
    --------
    fingerprints : tuple
        Tuple of (morgan_fps, maccs_fps, avalon_fps) as numpy arrays
    """
    if fp_sizes is None:
        fp_sizes = DEFAULT_FP_SIZES
    if fingerprint_types is None:
        fingerprint_types = DEFAULT_FINGERPRINT_TYPES
    
    morgan_fps = []
    maccs_fps = []
    avalon_fps = []
    
    for mol in mols:
        if 'morgan' in fingerprint_types:
            try:
                gen = GetMorganGenerator(radius=2, fpSize=fp_sizes.get('morgan', 2048))
                bv = gen.GetFingerprint(mol)
                morgan = np.zeros(fp_sizes.get('morgan', 2048), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(bv, morgan)
                morgan_fps.append(morgan)
            except Exception as e:
                print(f"Error generating Morgan fingerprint: {e}")
                morgan_fps.append(np.zeros(fp_sizes.get('morgan', 2048), dtype=np.uint8))
        
        if 'maccs' in fingerprint_types:
            try:
                maccs_bits = MACCSkeys.GenMACCSKeys(mol)
                maccs = np.zeros(fp_sizes.get('maccs', 167), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(maccs_bits, maccs)
                maccs_fps.append(maccs)
            except Exception as e:
                print(f"Error generating MACCS fingerprint: {e}")
                maccs_fps.append(np.zeros(fp_sizes.get('maccs', 167), dtype=np.uint8))
        
        if 'avalon' in fingerprint_types:
            try:
                av_bits = pyAvalonTools.GetAvalonFP(mol, nBits=fp_sizes.get('avalon', 512))
                avalon = np.zeros(fp_sizes.get('avalon', 512), dtype=np.uint8)
                DataStructs.ConvertToNumpyArray(av_bits, avalon)
                avalon_fps.append(avalon)
            except Exception as e:
                print(f"Error generating Avalon fingerprint: {e}")
                avalon_fps.append(np.zeros(fp_sizes.get('avalon', 512), dtype=np.uint8))
    
    # Convert to numpy arrays
    morgan_fps = np.array(morgan_fps) if morgan_fps else np.array([])
    maccs_fps = np.array(maccs_fps) if maccs_fps else np.array([])
    avalon_fps = np.array(avalon_fps) if avalon_fps else np.array([])
    
    return morgan_fps, maccs_fps, avalon_fps

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

# ===========================
# Data Loading Functions for ANO Modules
# ===========================

def load_data_ws():
    """Load WS dataset data"""
    try:
        data_dict = load_split_data()
        x_map, y_map = extract_xy_from_data(data_dict)
        
        # Get train and test data for WS
        train_key = "ws_train"
        test_key = "ws_test"
        
        # Find the split that has WS data
        for split_type in ['rm', 'sa', 'cs', 'cl', 'pc', 'ac', 'en']:
            if split_type in x_map and train_key in x_map[split_type] and test_key in x_map[split_type]:
                smiles_list = x_map[split_type][train_key] + x_map[split_type][test_key]
                targets = list(y_map[split_type][train_key]) + list(y_map[split_type][test_key])
                return smiles_list, targets
        
        raise ValueError("WS data not found in any split")
        
    except Exception as e:
        print(f"Error loading WS data: {e}")
        raise

def load_data_de():
    """Load DE dataset data"""
    try:
        data_dict = load_split_data()
        x_map, y_map = extract_xy_from_data(data_dict)
        
        # Get train and test data for DE
        train_key = "de_train"
        test_key = "de_test"
        
        # Find the split that has DE data
        for split_type in ['rm', 'sa', 'cs', 'cl', 'pc', 'ac', 'en']:
            if split_type in x_map and train_key in x_map[split_type] and test_key in x_map[split_type]:
                smiles_list = x_map[split_type][train_key] + x_map[split_type][test_key]
                targets = list(y_map[split_type][train_key]) + list(y_map[split_type][test_key])
                return smiles_list, targets
        
        raise ValueError("DE data not found in any split")
        
    except Exception as e:
        print(f"Error loading DE data: {e}")
        raise

def load_data_lo():
    """Load LO dataset data"""
    try:
        data_dict = load_split_data()
        x_map, y_map = extract_xy_from_data(data_dict)
        
        # Get train and test data for LO
        train_key = "lo_train"
        test_key = "lo_test"
        
        # Find the split that has LO data
        for split_type in ['rm', 'sa', 'cs', 'cl', 'pc', 'ac', 'en']:
            if split_type in x_map and train_key in x_map[split_type] and test_key in x_map[split_type]:
                smiles_list = x_map[split_type][train_key] + x_map[split_type][test_key]
                targets = list(y_map[split_type][train_key]) + list(y_map[split_type][test_key])
                return smiles_list, targets
        
        raise ValueError("LO data not found in any split")
        
    except Exception as e:
        print(f"Error loading LO data: {e}")
        raise

def load_data_hu():
    """Load HU dataset data"""
    try:
        data_dict = load_split_data()
        x_map, y_map = extract_xy_from_data(data_dict)
        
        # Get train and test data for HU
        train_key = "hu_train"
        test_key = "hu_test"
        
        # Find the split that has HU data
        for split_type in ['rm', 'sa', 'cs', 'cl', 'pc', 'ac', 'en']:
            if split_type in x_map and train_key in x_map[split_type] and test_key in x_map[split_type]:
                smiles_list = x_map[split_type][train_key] + x_map[split_type][test_key]
                targets = list(y_map[split_type][train_key]) + list(y_map[split_type][test_key])
                return smiles_list, targets
        
        raise ValueError("HU data not found in any split")
        
    except Exception as e:
        print(f"Error loading HU data: {e}")
        raise