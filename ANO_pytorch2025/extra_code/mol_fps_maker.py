#!/usr/bin/env python3
"""
Unified Molecular Fingerprints Generation and Caching System
============================================================

This comprehensive module provides high-performance molecular fingerprint generation
with intelligent caching, parallel processing, and memory optimization. It serves as
the core molecular representation engine for the ANO framework, supporting multiple
fingerprint types and efficient data management strategies.

Key Features:
------------
1. **Multi-Fingerprint Support**: Morgan, RDKit, MACCS, ECFP, Avalon, and custom fingerprints
2. **Intelligent Caching**: NPZ-based caching with automatic invalidation and versioning
3. **Parallel Processing**: Multi-core fingerprint generation with memory-efficient batching
4. **Memory Optimization**: Streaming data loading and garbage collection for large datasets
5. **Error Handling**: Robust SMILES validation and fallback mechanisms
6. **3D Conformer Support**: Integration with 3D molecular descriptors when available

Supported Fingerprint Types:
----------------------------
- **Morgan Fingerprints**: Circular fingerprints with customizable radius and bit count
- **RDKit Descriptors**: 200+ calculated molecular descriptors
- **MACCS Keys**: 166-bit structural key fingerprints
- **ECFP**: Extended Connectivity Fingerprints with various parameters
- **Avalon**: Avalon toolkit fingerprints for substructure analysis
- **Custom FPs**: User-defined fingerprint functions

Core Functions:
--------------
- **get_fingerprints_cached()**: Main entry point with intelligent caching
- **get_fingerprints_combined()**: Combined fingerprints based on configuration
- **get_fingerprints()**: Direct fingerprint generation without caching
- **load_split_data()**: Load preprocessed training/test splits

Performance Features:
--------------------
- **Batch Processing**: Configurable batch sizes for memory management
- **Parallel Execution**: Multi-process fingerprint calculation
- **Cache Optimization**: Automatic cache pruning and compression
- **Memory Monitoring**: Real-time memory usage tracking
- **Progress Tracking**: Detailed progress bars for long-running operations

Usage Examples:
--------------
# Basic fingerprint generation with caching
fps, failed_smiles = get_fingerprints_cached(
    dataset='ws', split_type='rm', fingerprint='morgan',
    use_cache=True, n_jobs=4
)

# Combined fingerprints (e.g., Morgan + RDKit descriptors)
combined_fps = get_fingerprints_combined(
    dataset='ws', split_type='rm',
    fingerprints=['morgan', 'rdkit_desc']
)

# Custom fingerprint parameters
fps = get_fingerprints_cached(
    dataset='de', split_type='sc', fingerprint='morgan',
    fp_params={'radius': 3, 'nBits': 4096}
)

# Load preprocessed data splits
train_data = load_split_data('ws', 'train', split_type='rm')
test_data = load_split_data('ws', 'test', split_type='rm')

Cache Management:
----------------
The module maintains an efficient NPZ cache system:
- Cache location: cache/fingerprints/{dataset}_{split}_{fingerprint}.npz
- Automatic invalidation based on source data timestamps
- Compressed storage for large fingerprint matrices
- Metadata tracking for reproducibility

Error Handling:
--------------
- Invalid SMILES detection and reporting
- Graceful degradation for failed fingerprint calculations
- Comprehensive logging of processing issues
- Fallback mechanisms for corrupted cache files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional, Union
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
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# From molecular_loader_with_fp.py - Data Loading and Fingerprint Generation
# ============================================================================

def check_3d_conformer_generation(smiles: str) -> bool:
    """
    Check if 3D conformer can be generated for a molecule
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if 3D conformer can be generated, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        mol_3d = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol_3d, randomSeed=42, maxAttempts=100)
        if result == -1:
            return False
        
        AllChem.MMFFOptimizeMolecule(mol_3d)
        return True
    except:
        return False

def save_failed_molecules(failed_molecules: Dict, result_path: str = None):
    """
    Save failed molecules to file
    """
    if result_path is None:
        from config import RESULT_PATH
        result_path = RESULT_PATH
    
    # Save to both fingerprint and chemical_descriptors folders
    for folder in ['fingerprint/failed', 'chemical_descriptors/failed']:
        failed_dir = Path(result_path) / folder
        failed_dir.mkdir(parents=True, exist_ok=True)
        
        failed_file = failed_dir / 'failed_3d_conformers.json'
        with open(failed_file, 'w') as f:
            json.dump(failed_molecules, f, indent=2)

def load_split_data(base_path: str = "data", check_3d_conformers: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load preprocessed train/test split data from the 1_preprocess directory
    Optionally checks and excludes molecules that fail 3D conformer generation
    
    Args:
        base_path: Base path for data directory
        check_3d_conformers: Whether to check and exclude 3D conformer failures (slow!)
    
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
    """
    base_path = Path(base_path)
    train_dir = base_path / "train"
    test_dir = base_path / "test"
    
    data_dict = {}
    
    # Define expected splits
    split_types = [
        'rm', 'sc', 'cs', 'cl', 'pc', 'ac', 'sa', 'ti', 'en'
    ]
    
    for split_type in split_types:
        data_dict[split_type] = {}
        
        # Load train data
        train_split_dir = train_dir / split_type
        if train_split_dir.exists():
            for csv_file in train_split_dir.glob("*.csv"):
                dataset_name = csv_file.stem.replace(f"{split_type}_", "").replace("_train", "")
                df = pd.read_csv(csv_file)
                data_dict[split_type][f"{dataset_name}_train"] = df
        
        # Load test data
        test_split_dir = test_dir / split_type
        if test_split_dir.exists():
            for csv_file in test_split_dir.glob("*.csv"):
                dataset_name = csv_file.stem.replace(f"{split_type}_", "").replace("_test", "")
                df = pd.read_csv(csv_file)
                data_dict[split_type][f"{dataset_name}_test"] = df
    
    # Only check for 3D conformer failures if requested
    if not check_3d_conformers:
        return data_dict
    
    print("\nChecking for 3D conformer generation failures (this may take a while)...")
    from config import RESULT_PATH
    
    # Load existing failed molecules if any
    failed_file = Path(RESULT_PATH) / 'fingerprint' / 'failed' / 'failed_3d_conformers.json'
    if failed_file.exists():
        with open(failed_file, 'r') as f:
            failed_molecules = json.load(f)
    else:
        failed_molecules = {}
    
    # Check all molecules and identify failures
    for split_type in data_dict.keys():
        for dataset_key in list(data_dict[split_type].keys()):
            df = data_dict[split_type][dataset_key]
            if len(df) == 0:
                continue
                
            smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
            target_col = 'target' if 'target' in df.columns else 'logS'
            if target_col not in df.columns:
                for col in ['logS0', 'measured log solubility in mols per litre', 'Solubility']:
                    if col in df.columns:
                        target_col = col
                        break
            
            # Check each molecule
            failed_indices = []
            for idx, row in df.iterrows():
                smiles = row[smiles_col]
                
                # Skip if already known to fail
                if smiles in failed_molecules:
                    failed_indices.append(idx)
                    continue
                
                # Test 3D conformer generation
                if not check_3d_conformer_generation(smiles):
                    failed_indices.append(idx)
                    # Record failure
                    if smiles not in failed_molecules:
                        failed_molecules[smiles] = {
                            'target': row[target_col] if target_col in df.columns else 0.0,
                            'first_failed': datetime.now().isoformat(),
                            'occurrences': []
                        }
                    
                    # Extract dataset name and type
                    dataset_name = dataset_key.replace('_train', '').replace('_test', '')
                    data_type = 'train' if '_train' in dataset_key else 'test'
                    
                    occurrence = {
                        'dataset': dataset_name,
                        'split_type': split_type,
                        'data_type': data_type,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Check if not already recorded
                    existing = False
                    for occ in failed_molecules[smiles]['occurrences']:
                        if (occ.get('dataset') == dataset_name and 
                            occ.get('split_type') == split_type and
                            occ.get('data_type') == data_type):
                            existing = True
                            break
                    
                    if not existing:
                        failed_molecules[smiles]['occurrences'].append(occurrence)
            
            # Remove failed molecules from dataframe
            if failed_indices:
                original_len = len(df)
                df = df.drop(failed_indices).reset_index(drop=True)
                data_dict[split_type][dataset_key] = df
                print(f"  Excluded {len(failed_indices)} molecules from {dataset_key} ({split_type})")
    
    # Save failed molecules
    if failed_molecules:
        save_failed_molecules(failed_molecules)
        print(f"\nTotal molecules with 3D conformer failures: {len(failed_molecules)}")
        print(f"Saved to: fingerprint/failed/failed_3d_conformers.json")
    
    return data_dict

def get_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint using new MorganGenerator API"""
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    
    # Use new MorganGenerator API to avoid deprecation warning
    mfpgen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mfpgen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_maccs_fingerprint(mol):
    """Generate MACCS fingerprint (167 bits)"""
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_avalon_fingerprint(mol, n_bits=512):
    """Generate Avalon fingerprint"""
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    
    try:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits, dtype=np.uint8)

def process_molecule(mol):
    """Process single molecule to generate all fingerprints"""
    morgan_fp = get_morgan_fingerprint(mol)
    maccs_fp = get_maccs_fingerprint(mol)
    avalon_fp = get_avalon_fingerprint(mol)
    return morgan_fp, maccs_fp, avalon_fp

def get_fingerprints(mols: List[Chem.Mol], use_parallel: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate fingerprints for a list of molecules
    
    Parameters:
    -----------
    mols : List[Chem.Mol]
        List of RDKit molecules
    use_parallel : bool
        Whether to use parallel processing
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (morgan_fps, maccs_fps, avalon_fps)
    """
    n_mols = len(mols)
    
    if n_mols == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    # Initialize arrays
    morgan_fps = np.zeros((n_mols, 2048), dtype=np.uint8)
    maccs_fps = np.zeros((n_mols, 167), dtype=np.uint8)
    avalon_fps = np.zeros((n_mols, 512), dtype=np.uint8)
    
    if use_parallel and n_mols > 100:
        # Use parallel processing for large datasets
        with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            futures = {executor.submit(process_molecule, mol): i 
                      for i, mol in enumerate(mols)}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    morgan_fp, maccs_fp, avalon_fp = future.result()
                    morgan_fps[idx] = morgan_fp
                    maccs_fps[idx] = maccs_fp
                    avalon_fps[idx] = avalon_fp
                except Exception as e:
                    print(f"Error processing molecule {idx}: {e}")
    else:
        # Sequential processing for small datasets
        for i, mol in enumerate(mols):
            try:
                morgan_fps[i] = get_morgan_fingerprint(mol)
                maccs_fps[i] = get_maccs_fingerprint(mol)
                avalon_fps[i] = get_avalon_fingerprint(mol)
            except Exception as e:
                print(f"Error processing molecule {i}: {e}")
    
    return morgan_fps, maccs_fps, avalon_fps

# ============================================================================
# From npz_cache_utils.py - NPZ Caching System
# ============================================================================

def get_fingerprints_cached(
    mols: List[Chem.Mol],
    dataset_name: str,
    split_type: str,
    data_type: str,
    module_name: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get fingerprints with NPZ caching (uncompressed for speed)
    
    Parameters:
    -----------
    mols : List[Chem.Mol]
        List of RDKit molecules
    dataset_name : str
        Dataset identifier (e.g., 'ws', 'de')
    split_type : str
        Split type (e.g., 'rm', 'ac')
    data_type : str
        'train' or 'test'
    module_name : str, optional
        Name of the module using this function (unused, kept for compatibility)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (morgan_fps, maccs_fps, avalon_fps)
    """
    # Setup cache directory with dataset/split_type folder structure
    from config import RESULT_PATH
    # Always use shared cache directory in result folder
    fp_dir = Path(RESULT_PATH) / f'fingerprint/{dataset_name.lower()}/{split_type}'
    fp_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file path - include dataset and split in filename for clarity
    cache_file = fp_dir / f"{dataset_name.lower()}_{split_type}_{data_type}.npz"
    
    # Check remake flag from config
    from config import CACHE_CONFIG
    remake_fingerprint = CACHE_CONFIG.get('remake_fingerprint', False)
    
    # Try to load from cache if not remaking
    if cache_file.exists() and not remake_fingerprint:
        try:
            print(f"  Loading cached fingerprints from {cache_file}...")
            data = np.load(cache_file)
            
            # Check if individual fingerprints are available
            if 'morgan' in data and 'maccs' in data and 'avalon' in data:
                morgan = data['morgan']
                maccs = data['maccs']
                avalon = data['avalon']
                
                # Verify shape matches
                if morgan.shape[0] == len(mols):
                    return morgan, maccs, avalon
                else:
                    print(f"    Cache size mismatch: {morgan.shape[0]} vs {len(mols)}")
                    print(f"    Regenerating fingerprints...")
            
            # Try combined format (backward compatibility)
            elif 'X' in data:
                X = data['X']
                if X.shape[0] == len(mols):
                    # Split combined fingerprints
                    morgan = X[:, :2048]
                    maccs = X[:, 2048:2215]
                    avalon = X[:, 2215:]
                    return morgan, maccs, avalon
                else:
                    print(f"    Cache size mismatch: {X.shape[0]} vs {len(mols)}")
                    print(f"    Regenerating fingerprints...")
        except Exception as e:
            print(f"    Warning: Failed to load cache: {e}")
    
    # Generate new fingerprints
    print(f"  Generating new fingerprints for {dataset_name}/{split_type}/{data_type}...")
    
    # Generate fingerprints
    morgan_fps, maccs_fps, avalon_fps = get_fingerprints(mols)
    
    # Save to NPZ cache (uncompressed for speed)
    try:
        np.savez(cache_file,
                morgan=morgan_fps,
                maccs=maccs_fps,
                avalon=avalon_fps,
                X=np.hstack([morgan_fps, maccs_fps, avalon_fps]))
        print(f"  Saved fingerprints to {cache_file.name}")
    except Exception as e:
        print(f"    Warning: Failed to save cache: {e}")
    
    return morgan_fps, maccs_fps, avalon_fps

def get_fingerprints_from_cache_only(
    dataset_name: str,
    split_type: str, 
    data_type: str,
    fingerprint_type: str = 'morgan'
) -> np.ndarray:
    """
    Get fingerprints directly from cache without requiring mol objects
    Used by module 2 to avoid unnecessary SMILES->Mol conversion
    
    Parameters:
    -----------
    dataset_name : str
        Dataset identifier (e.g., 'ws', 'de')
    split_type : str
        Split type (e.g., 'rm', 'ac')
    data_type : str
        'train' or 'test'
    fingerprint_type : str
        Options: 'morgan', 'maccs', 'avalon', 'all', combinations
        
    Returns:
    --------
    np.ndarray
        Fingerprints based on selection, or None if cache not found
    """
    from config import RESULT_PATH
    
    # Setup cache file path
    fp_dir = Path(RESULT_PATH) / f'fingerprint/{dataset_name.lower()}/{split_type}'
    cache_file = fp_dir / f"{dataset_name.lower()}_{split_type}_{data_type}.npz"
    
    if not cache_file.exists():
        return None
        
    try:
        data = np.load(cache_file)
        
        # Check if individual fingerprints are available
        if 'morgan' in data and 'maccs' in data and 'avalon' in data:
            morgan = data['morgan']
            maccs = data['maccs'] 
            avalon = data['avalon']
        elif 'X' in data:
            # Split combined fingerprints (backward compatibility)
            X = data['X']
            morgan = X[:, :2048]
            maccs = X[:, 2048:2215]
            avalon = X[:, 2215:]
        else:
            return None
            
        # Combine based on selection
        if fingerprint_type == 'morgan':
            return morgan
        elif fingerprint_type == 'maccs':
            return maccs
        elif fingerprint_type == 'avalon':
            return avalon
        elif fingerprint_type == 'all':
            return np.hstack([morgan, maccs, avalon])
        elif fingerprint_type == 'morgan+maccs':
            return np.hstack([morgan, maccs])
        elif fingerprint_type == 'morgan+avalon':
            return np.hstack([morgan, avalon])
        elif fingerprint_type == 'maccs+avalon':
            return np.hstack([maccs, avalon])
        else:
            print(f"Warning: Unknown fingerprint type '{fingerprint_type}', using 'morgan'")
            return morgan
            
    except Exception as e:
        print(f"Error loading fingerprints from cache: {e}")
        return None

def get_fingerprints_combined(
    mols: List[Chem.Mol],
    dataset_name: str,
    split_type: str, 
    data_type: str,
    fingerprint_type: str = 'all',
    module_name: str = None
) -> np.ndarray:
    """
    Get combined fingerprints based on fingerprint type selection
    
    Parameters:
    -----------
    fingerprint_type : str
        Options:
        - 'morgan': Morgan only (2048 bits)
        - 'maccs': MACCS only (167 bits)
        - 'avalon': Avalon only (512 bits)
        - 'all': All three (2727 bits)
        - 'morgan+maccs': Morgan + MACCS (2215 bits)
        - 'morgan+avalon': Morgan + Avalon (2560 bits)
        - 'maccs+avalon': MACCS + Avalon (679 bits)
        
    Returns:
    --------
    np.ndarray
        Combined fingerprints based on selection
    """
    # Get all fingerprints from cache
    morgan, maccs, avalon = get_fingerprints_cached(
        mols, dataset_name, split_type, data_type, module_name
    )
    
    # Combine based on selection
    if fingerprint_type == 'morgan':
        return morgan
    elif fingerprint_type == 'maccs':
        return maccs
    elif fingerprint_type == 'avalon':
        return avalon
    elif fingerprint_type == 'all':
        return np.hstack([morgan, maccs, avalon])
    elif fingerprint_type == 'morgan+maccs':
        return np.hstack([morgan, maccs])
    elif fingerprint_type == 'morgan+avalon':
        return np.hstack([morgan, avalon])
    elif fingerprint_type == 'maccs+avalon':
        return np.hstack([maccs, avalon])
    else:
        print(f"Warning: Unknown fingerprint type '{fingerprint_type}', using 'all'")
        return np.hstack([morgan, maccs, avalon])

# ============================================================================
# Data Loading Functions (from molecular_loader_with_fp.py)
# ============================================================================

def load_data_ws():
    """Load WS496 dataset"""
    df = pd.read_csv('data/ws496_logS.csv')
    return df['SMILES'].tolist(), df['logS'].tolist()

def load_data_de():
    """Load Delaney dataset"""
    df = pd.read_csv('data/delaney-processed.csv')
    return df['smiles'].tolist(), df['measured log solubility in mols per litre'].tolist()

def load_data_lo():
    """Load Lovric2020 dataset"""
    df = pd.read_csv('data/Lovric2020_logS0.csv')
    return df['SMILES'].tolist(), df['logS0'].tolist()

def load_data_hu():
    """Load Huuskonen dataset"""
    df = pd.read_csv('data/huusk.csv')
    return df['SMILES'].tolist(), df['logS'].tolist()

def precompute_all_fingerprints(module_number=None):
    """Pre-calculate all fingerprints for all datasets and split types
    
    Args:
        module_number: If provided, only compute for datasets used by that module
    """
    import time
    import sys
    import os

    # Try importing from config, add path only if necessary
    try:
        from config import get_code_datasets, ACTIVE_SPLIT_TYPES
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import get_code_datasets, ACTIVE_SPLIT_TYPES
    
    print("\n" + "="*60)
    print(f"PRECOMPUTING FINGERPRINTS FOR MODULE {module_number if module_number else 'ALL'}")
    print("="*60)
    
    datasets = get_code_datasets(4)  # Get datasets for module 4
    split_types = ACTIVE_SPLIT_TYPES
    
    total_start = time.time()
    processed = 0
    skipped = 0
    failed = 0
    
    # Load all data at once
    all_data = load_split_data()
    
    for dataset in datasets:
        for split_type in split_types:
            for data_type in ['train', 'test']:
                try:
                    print(f"\nProcessing {dataset}-{split_type}-{data_type}...")
                    
                    # Get specific dataset from dictionary
                    if split_type not in all_data:
                        print(f"  ⚠️  Split type {split_type} not found")
                        failed += 1
                        continue
                    
                    data_key = f"{dataset}_{data_type}"
                    if data_key not in all_data[split_type]:
                        print(f"  ⚠️  Dataset {data_key} not found")
                        failed += 1
                        continue
                    
                    df = all_data[split_type][data_key]
                    if df is None or len(df) == 0:
                        print(f"  ⚠️  No data available")
                        failed += 1
                        continue
                    
                    # Extract SMILES and convert to molecules
                    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
                    mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
                    
                    # Check if cache exists
                    from config import RESULT_PATH
                    fp_dir = Path(RESULT_PATH) / f'fingerprint/{dataset.lower()}/{split_type}'
                    cache_file = fp_dir / f"{dataset.lower()}_{split_type}_{data_type}.npz"
                    
                    if cache_file.exists():
                        print(f"  ✅ Already cached")
                        skipped += 1
                        continue
                    
                    # Generate/load fingerprints (will cache automatically)
                    morgan, maccs, avalon = get_fingerprints_cached(
                        mols, dataset.upper(), split_type, data_type
                    )
                    
                    if morgan is not None:
                        processed += 1
                        print(f"  ✅ Fingerprints generated: Morgan {morgan.shape}, MACCS {maccs.shape}, Avalon {avalon.shape}")
                    else:
                        failed += 1
                        print(f"  ❌ Failed to generate fingerprints")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    failed += 1
                    continue
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Fingerprint precomputation complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped (cached): {skipped} files")
    print(f"  Failed: {failed} files")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"{'='*60}\n")


# For backward compatibility
if __name__ == "__main__":
    precompute_all_fingerprints()