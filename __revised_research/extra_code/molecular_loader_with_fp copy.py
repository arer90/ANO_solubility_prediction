#!/usr/bin/env python3
"""
Molecular Data Processing Pipeline
==================================

A high-performance pipeline for processing molecular data including:
- Parallel data loading from CSV files
- SMILES extraction and target value processing
- Molecular fingerprint generation (Morgan, MACCS, Avalon)
- Efficient caching and memory management

Author: Your Name
Date: 2024
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
from datetime import datetime
import time
import argparse

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, AllChem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.Chem import MACCSkeys
    from rdkit.Avalon import pyAvalonTools
except ImportError:
    print("Error: RDKit not installed. Please install it using:")
    print("conda install -c conda-forge rdkit")
    raise

warnings.filterwarnings('ignore')

# RDKit function simplification
GenMACCSKeys = MACCSkeys.GenMACCSKeys
GetAvalonFP = pyAvalonTools.GetAvalonFP

# ===========================
# Constants - Modify as needed
# ===========================

SPLIT_ABBREVIATIONS = {
    'random': 'rm',
    'scaffold': 'sc',
    'chemical_space_coverage': 'cs',
    'cluster': 'cl',
    'physchem': 'pc',
    'activity_cliff': 'ac',
    'solubility_aware': 'sa',
    'time_series': 'ti',
    'ensemble': 'en',
}

DATASET_MAP = {
    "delaney-processed": "de",
    "Lovric2020_logS0": "lo",
    "ws496_logS": "ws",
    "huusk": "hu"
}

# ===========================
# Data Loading Functions
# ===========================

def load_csv_file(csv_file: Path, dataset_map: dict) -> Optional[Tuple[str, pd.DataFrame]]:
    """Load a single CSV file and return dataset abbreviation and DataFrame"""
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

def load_split_data_parallel(base_dir: str = "result/1_preprocess", max_workers: Optional[int] = None):
    """Parallel version of load_split_data with fallback to sequential"""
    try:
        return _load_split_data_parallel_impl(base_dir, max_workers)
    except Exception as e:
        print(f"Parallel loading failed: {e}. Falling back to sequential loading...")
        return load_split_data_sequential(base_dir)

def _load_split_data_parallel_impl(base_dir: str, max_workers: Optional[int] = None):
    """Internal parallel implementation"""
    BASE = Path(base_dir)
    data_dict = {}
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    file_tasks = []
    for split_abbr, split_name in SPLIT_MAP.items():
        data_dict[split_abbr] = {}
        
        for phase in ["train", "test"]:
            phase_dir = BASE / phase / split_abbr
            if not phase_dir.exists():
                continue
                
            data_dict[split_abbr][phase] = {}
            csv_files = list(phase_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                file_tasks.append((split_abbr, phase, csv_file))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        load_func = partial(load_csv_file, dataset_map=DATASET_MAP)
        
        future_to_task = {
            executor.submit(load_func, task[2]): task 
            for task in file_tasks
        }
        
        for future in as_completed(future_to_task):
            split_abbr, phase, csv_file = future_to_task[future]
            result = future.result()
            
            if result is not None:
                abbr, df = result
                data_dict[split_abbr][phase][abbr] = df
    
    return data_dict

def load_split_data_sequential(base_dir: str = "result/1_preprocess"):
    """Sequential version - original implementation"""
    BASE = Path(base_dir)
    data_dict = {}
    
    for split_abbr, split_name in SPLIT_MAP.items():
        data_dict[split_abbr] = {}
        
        for phase in ["train", "test"]:
            phase_dir = BASE / phase / split_abbr
            if not phase_dir.exists():
                continue
                
            data_dict[split_abbr][phase] = {}
            
            # List all CSV files in directory
            csv_files = list(phase_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                filename = csv_file.stem
                
                # Find matching dataset
                for full_name, abbr in DATASET_MAP.items():
                    if full_name in filename:
                        df = pd.read_csv(csv_file)
                        data_dict[split_abbr][phase][abbr] = df
                        # print(f"Loaded: {split_abbr}/{phase}/{abbr} - shape: {df.shape}")
                        break
    
    return data_dict

def load_split_data(base_dir: str = "result/1_preprocess", use_parallel: bool = True):
    """Load train/test data from preprocessed split directories"""
    if use_parallel:
        return load_split_data_parallel(base_dir)
    else:
        return load_split_data_sequential(base_dir)

# ===========================
# Data Extraction Functions
# ===========================

def process_single_dataset(args: Tuple[str, str, str, pd.DataFrame]) -> Dict:
    """Process a single dataset to extract X and y values"""
    split_abbr, phase, dataset_abbr, df = args
    key = f"{dataset_abbr}_{phase}"
    result = {
        'split_abbr': split_abbr,
        'key': key,
        'x': None,
        'y': None
    }
    
    smiles_col = None
    for col in df.columns:
        if col.lower() == 'smiles':
            smiles_col = col
            break
    
    if smiles_col:
        result['x'] = df[smiles_col].tolist()
    else:
        print(f"Warning: No SMILES column found in {split_abbr}/{phase}/{dataset_abbr}")
        return result
    
    target_columns = [
        'target',
        'logS',
        'measured log solubility in mols per litre'
    ]
    
    y_col = None
    for col in target_columns:
        if col in df.columns:
            y_col = col
            break
    
    if y_col:
        result['y'] = df[y_col].astype(float).tolist()
    else:
        log_cols = [col for col in df.columns if 'log' in col.lower()]
        if log_cols:
            result['y'] = df[log_cols[0]].astype(float).tolist()
        else:
            print(f"Warning: No target column found in {split_abbr}/{phase}/{dataset_abbr}")
    
    return result

def extract_xy_from_data_parallel(data_dict: Dict, max_workers: Optional[int] = None):
    """Parallel version of extract_xy_from_data with fallback"""
    try:
        return _extract_xy_from_data_parallel_impl(data_dict, max_workers)
    except Exception as e:
        print(f"Parallel extraction failed: {e}. Falling back to sequential extraction...")
        return extract_xy_from_data_sequential(data_dict)

def _extract_xy_from_data_parallel_impl(data_dict: Dict, max_workers: Optional[int] = None):
    """Internal parallel implementation"""
    x_map = {}
    y_map = {}
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)
    
    tasks = []
    for split_abbr, phases in data_dict.items():
        x_map[split_abbr] = {}
        y_map[split_abbr] = {}
        
        for phase, datasets in phases.items():
            for dataset_abbr, df in datasets.items():
                tasks.append((split_abbr, phase, dataset_abbr, df))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_dataset, task) for task in tasks]
        
        for future in as_completed(futures):
            result = future.result()
            if result['x'] is not None:
                split_abbr = result['split_abbr']
                key = result['key']
                x_map[split_abbr][key] = result['x']
                if result['y'] is not None:
                    y_map[split_abbr][key] = result['y']
    
    return x_map, y_map

def extract_xy_from_data_sequential(data_dict: Dict):
    """Sequential version - original implementation"""
    x_map = {}
    y_map = {}
    
    for split_abbr, phases in data_dict.items():
        x_map[split_abbr] = {}
        y_map[split_abbr] = {}
        
        for phase, datasets in phases.items():
            for dataset_abbr, df in datasets.items():
                key = f"{dataset_abbr}_{phase}"
                
                # Extract SMILES and target columns
                if 'SMILES' in df.columns:
                    x_map[split_abbr][key] = df['SMILES'].tolist()
                elif 'smiles' in df.columns:
                    x_map[split_abbr][key] = df['smiles'].tolist()
                else:
                    print(f"Warning: No SMILES column found in {split_abbr}/{phase}/{dataset_abbr}")
                    continue
                
                # Extract target values
                if 'target' in df.columns:
                    y_map[split_abbr][key] = df['target'].astype(float).tolist()
                elif 'logS' in df.columns:
                    y_map[split_abbr][key] = df['logS'].astype(float).tolist()
                elif 'measured log solubility in mols per litre' in df.columns:
                    y_map[split_abbr][key] = df['measured log solubility in mols per litre'].astype(float).tolist()
                else:
                    # Try to find any column with 'log' in name
                    log_cols = [col for col in df.columns if 'log' in col.lower()]
                    if log_cols:
                        y_map[split_abbr][key] = df[log_cols[0]].astype(float).tolist()
                    else:
                        print(f"Warning: No target column found in {split_abbr}/{phase}/{dataset_abbr}")
    
    return x_map, y_map

def extract_xy_from_data(data_dict: Dict, use_parallel: bool = True):
    """Extract SMILES (X) and target values (y) from loaded data"""
    if use_parallel:
        return extract_xy_from_data_parallel(data_dict)
    else:
        return extract_xy_from_data_sequential(data_dict)

# ===========================
# Fingerprint Functions
# ===========================

def get_fingerprints(mol, LEN_FF=2048, LEN_MA=167, LEN_AV=512):
    """Generate fingerprints for a single molecule"""
    gen = GetMorganGenerator(radius=2, fpSize=LEN_FF)
    bv = gen.GetFingerprint(mol)
    ff = np.zeros(LEN_FF, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, ff)

    maccs_bits = GenMACCSKeys(mol)
    ma = np.zeros(LEN_MA, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(maccs_bits, ma)

    av_bits = GetAvalonFP(mol, nBits=LEN_AV)
    av = np.zeros(LEN_AV, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(av_bits, av)

    return ff, ma, av

def process_smiles_batch(smiles_batch: List[Tuple[int, str]], 
                        LEN_FF: int = 2048, 
                        LEN_MA: int = 167, 
                        LEN_AV: int = 512) -> Dict:
    """Process a batch of SMILES strings"""
    results = {
        'morgan': np.zeros((len(smiles_batch), LEN_FF), dtype=np.uint8),
        'maccs': np.zeros((len(smiles_batch), LEN_MA), dtype=np.uint8),
        'avalon': np.zeros((len(smiles_batch), LEN_AV), dtype=np.uint8),
        'failed_indices': []
    }
    
    morgan_gen = GetMorganGenerator(radius=2, fpSize=LEN_FF)
    
    for i, (orig_idx, smi) in enumerate(smiles_batch):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                bv = morgan_gen.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(bv, results['morgan'][i])
                
                maccs_bits = GenMACCSKeys(mol)
                DataStructs.ConvertToNumpyArray(maccs_bits, results['maccs'][i])
                
                av_bits = GetAvalonFP(mol, nBits=LEN_AV)
                DataStructs.ConvertToNumpyArray(av_bits, results['avalon'][i])
            else:
                results['failed_indices'].append(orig_idx)
        except Exception as e:
            results['failed_indices'].append(orig_idx)
    
    return results

def build_fingerprints_parallel(smiles_list: List[str], 
                              n_jobs: int = None,
                              batch_size: int = 100) -> Dict:
    """Build fingerprints in parallel"""
    if n_jobs is None:
        n_jobs = min(mp.cpu_count() - 1, 16)
    
    n_mols = len(smiles_list)
    
    morgan_fps = np.zeros((n_mols, 2048), dtype=np.uint8)
    maccs_fps = np.zeros((n_mols, 167), dtype=np.uint8)
    avalon_fps = np.zeros((n_mols, 512), dtype=np.uint8)
    failed_indices = []
    
    batches = []
    for i in range(0, n_mols, batch_size):
        batch = [(j, smiles_list[j]) for j in range(i, min(i + batch_size, n_mols))]
        batches.append(batch)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_smiles_batch, batch): i 
                  for i, batch in enumerate(batches)}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Fingerprint batches", leave=False):
            batch_idx = futures[future]
            try:
                result = future.result()
                
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batches[batch_idx])
                
                morgan_fps[start_idx:end_idx] = result['morgan']
                maccs_fps[start_idx:end_idx] = result['maccs']
                avalon_fps[start_idx:end_idx] = result['avalon']
                
                failed_indices.extend(result['failed_indices'])
                
            except Exception as e:
                print(f"Batch {batch_idx} processing failed: {e}")
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_mols)
                failed_indices.extend(range(start_idx, end_idx))
    
    return {
        'morgan': morgan_fps,
        'maccs': maccs_fps,
        'avalon': avalon_fps,
        'failed_indices': failed_indices
    }

def build_fingerprints_sequential(smiles_list: List[str]) -> Dict:
    """Build fingerprints sequentially"""
    n_mols = len(smiles_list)
    
    morgan_fps = np.zeros((n_mols, 2048), dtype=np.uint8)
    maccs_fps = np.zeros((n_mols, 167), dtype=np.uint8)
    avalon_fps = np.zeros((n_mols, 512), dtype=np.uint8)
    failed_indices = []
    
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    
    for i, smi in enumerate(tqdm(smiles_list, desc="Sequential fingerprints", leave=False)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                bv = morgan_gen.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(bv, morgan_fps[i])
                
                maccs_bits = GenMACCSKeys(mol)
                DataStructs.ConvertToNumpyArray(maccs_bits, maccs_fps[i])
                
                av_bits = GetAvalonFP(mol, nBits=512)
                DataStructs.ConvertToNumpyArray(av_bits, avalon_fps[i])
            else:
                failed_indices.append(i)
        except Exception as e:
            failed_indices.append(i)
    
    return {
        'morgan': morgan_fps,
        'maccs': maccs_fps,
        'avalon': avalon_fps,
        'failed_indices': failed_indices
    }

def build_fingerprints_for_splits(x_map: Dict, 
                                out_root: Path, 
                                n_jobs: int = 16, 
                                force_rebuild: bool = False,
                                use_cache: bool = True,
                                use_parallel: bool = True,
                                batch_size: int = 100):
    """Build fingerprints for all splits with caching
    
    Parameters:
    -----------
    x_map : Dict
        Dictionary containing SMILES data
    out_root : Path
        Output directory for fingerprint cache files
    n_jobs : int
        Number of parallel jobs (default: 16)
    force_rebuild : bool
        If True, ignore existing cache and rebuild (default: False)
    use_cache : bool
        If True, use caching system (default: True)
    use_parallel : bool
        If True, use parallel processing (default: True)
    batch_size : int
        Batch size for parallel processing (default: 100)
    
    Returns:
    --------
    fp_map : Dict
        Dictionary containing fingerprints for each dataset
    """
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
                
                # Check cache (only when use_cache=True and force_rebuild=False)
                if use_cache and not force_rebuild and cache_file.exists():
                    try:
                        data = np.load(cache_file)
                        
                        if ('morgan' in data and 'maccs' in data and 'avalon' in data and
                            len(data['morgan']) == len(smiles_list)):
                            
                            fp_map[split_abbr][key] = {
                                'morgan': data['morgan'],
                                'maccs': data['maccs'],
                                'avalon': data['avalon']
                            }
                            pbar.update(1)
                            continue
                        else:
                            print(f"Cache file corrupted or size mismatch: {cache_file.name}")
                    except Exception as e:
                        print(f"Cache load failed: {cache_file.name}, error: {e}")
                
                # Generate fingerprints
                try:
                    if use_parallel and len(smiles_list) > 100:
                        result = build_fingerprints_parallel(smiles_list, n_jobs, batch_size)
                    else:
                        # Process using original method
                        morgan_fps = []
                        maccs_fps = []
                        avalon_fps = []
                        failed_indices = []
                        
                        # Batch processing
                        batch_size_seq = 1000
                        n_batches = (len(smiles_list) + batch_size_seq - 1) // batch_size_seq
                        
                        for batch_idx in range(n_batches):
                            start = batch_idx * batch_size_seq
                            end = min((batch_idx + 1) * batch_size_seq, len(smiles_list))
                            batch_smiles = smiles_list[start:end]
                            
                            # Individual processing (prioritizing stability over parallel processing)
                            for i, smi in enumerate(batch_smiles):
                                try:
                                    mol = Chem.MolFromSmiles(smi)
                                    if mol is not None:
                                        ff, ma, av = get_fingerprints(mol)
                                        morgan_fps.append(ff)
                                        maccs_fps.append(ma)
                                        avalon_fps.append(av)
                                    else:
                                        # Empty fingerprint
                                        morgan_fps.append(np.zeros(2048, dtype=int))
                                        maccs_fps.append(np.zeros(167, dtype=int))
                                        avalon_fps.append(np.zeros(512, dtype=int))
                                        failed_indices.append(start + i)
                                except Exception as e:
                                    print(f"Error processing SMILES at index {start + i}: {e}")
                                    # Add empty fingerprint
                                    morgan_fps.append(np.zeros(2048, dtype=int))
                                    maccs_fps.append(np.zeros(167, dtype=int))
                                    avalon_fps.append(np.zeros(512, dtype=int))
                                    failed_indices.append(start + i)
                        
                        if failed_indices:
                            print(f"Warning: {len(failed_indices)} SMILES processing failed")
                        
                        result = {
                            'morgan': np.array(morgan_fps, dtype=int),
                            'maccs': np.array(maccs_fps, dtype=int),
                            'avalon': np.array(avalon_fps, dtype=int),
                            'failed_indices': failed_indices
                        }
                    
                    fp_data = {
                        'morgan': result['morgan'],
                        'maccs': result['maccs'],
                        'avalon': result['avalon']
                    }
                    
                    # Save cache (only when use_cache is True)
                    if use_cache:
                        try:
                            np.savez_compressed(cache_file, **fp_data)
                            if cache_file.exists():
                                file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
                            else:
                                print(f"Warning: Cache file was not saved!")
                        except Exception as e:
                            print(f"Cache save failed: {e}")
                            print(f"Current directory: {os.getcwd()}")
                            print(f"Absolute path: {cache_file.absolute()}")
                    
                    fp_map[split_abbr][key] = fp_data
                    
                except Exception as e:
                    print(f"Fingerprint generation failed ({split_abbr}/{key}): {e}")
                    # Generate empty fingerprints
                    n_mols = len(smiles_list)
                    fp_map[split_abbr][key] = {
                        'morgan': np.zeros((n_mols, 2048), dtype=np.uint8),
                        'maccs': np.zeros((n_mols, 167), dtype=np.uint8),
                        'avalon': np.zeros((n_mols, 512), dtype=np.uint8)
                    }
                
                pbar.update(1)
                gc.collect()
    
    # Final cache status check
    cache_files = list(cache_dir.glob("*.npz"))
    if cache_files:
        for cf in sorted(cache_files)[:10]:  # Show first 10 files
            size_mb = cf.stat().st_size / (1024 * 1024)
        if len(cache_files) > 10:
            print(f"  ... and {len(cache_files) - 10} more files")
    
    return fp_map

# ===========================
# Utility Functions
# ===========================

def check_cache_status(out_root: Path):
    """Check fingerprint cache status"""
    if not isinstance(out_root, Path):
        out_root = Path(out_root)
    
    cache_dir = out_root
    
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.npz"))
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024 * 1024)
        
        print(f"\n=== Cache Status ===")
        print(f"Cache directory: {cache_dir.absolute()}")
        print(f"Number of cache files: {len(cache_files)}")
        print(f"Total cache size: {total_size:.2f} GB")
        
        if cache_files:
            print("\nRecent cache files (max 5):")
            for cf in sorted(cache_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                size_mb = cf.stat().st_size / (1024 * 1024)
                mtime = cf.stat().st_mtime
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  - {cf.name} ({size_mb:.2f} MB, modified: {mtime_str})")
    else:
        print("Cache directory does not exist.")

def rebuild_cache(x_map: Dict, out_root: Path, n_jobs: int = 16, use_parallel: bool = True):
    """Rebuild fingerprint cache
    
    Parameters:
    -----------
    x_map : Dict
        Dictionary containing SMILES data
    out_root : Path
        Output directory for fingerprint cache files
    n_jobs : int
        Number of parallel jobs (default: 16)
    use_parallel : bool
        If True, use parallel processing (default: True)
    
    Returns:
    --------
    fp_map : Dict
        Dictionary containing newly built fingerprints
    """
    return build_fingerprints_for_splits(x_map, out_root, n_jobs, 
                                       force_rebuild=True, 
                                       use_cache=True,
                                       use_parallel=use_parallel)

def benchmark_loading(base_dir: str = "result/1_preprocess"):
    """Benchmark parallel vs sequential loading"""
    print("\n=== Loading Benchmark ===")
    
    start = time.time()
    data_seq = load_split_data_sequential(base_dir)
    seq_time = time.time() - start
    
    start = time.time()
    data_par = load_split_data_parallel(base_dir)
    par_time = time.time() - start
    
    print(f"Sequential loading time: {seq_time:.2f}s")
    print(f"Parallel loading time: {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    
    return data_seq, data_par

def benchmark_fingerprints(smiles_list: List[str], n_runs: int = 3):
    """Benchmark fingerprint generation"""
    print(f"\n=== Fingerprint Benchmark ({len(smiles_list)} molecules) ===")
    
    test_size = min(1000, len(smiles_list))
    test_smiles = smiles_list[:test_size]
    
    seq_times = []
    for i in range(n_runs):
        start = time.time()
        _ = build_fingerprints_sequential(test_smiles)
        seq_times.append(time.time() - start)
    
    par_times = []
    for i in range(n_runs):
        start = time.time()
        _ = build_fingerprints_parallel(test_smiles, n_jobs=8)
        par_times.append(time.time() - start)
    
    print(f"Sequential: {np.mean(seq_times):.2f}s (±{np.std(seq_times):.2f}s)")
    print(f"Parallel: {np.mean(par_times):.2f}s (±{np.std(par_times):.2f}s)")
    print(f"Speedup: {np.mean(seq_times)/np.mean(par_times):.2f}x")

# ===========================
# Main Pipeline Function
# ===========================

def run_full_pipeline(base_dir: str = "result/1_preprocess",
                     out_root: str = "result/2_fingerprint",
                     use_parallel: bool = True,
                     use_cache: bool = True,
                     n_jobs: int = 16,
                     force_rebuild: bool = False):
    """Run the complete molecular data processing pipeline
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing preprocessed data
    out_root : str
        Output directory for fingerprints
    use_parallel : bool
        If True, use parallel processing (default: True)
    use_cache : bool
        If True, use caching system for fingerprints (default: True)
    n_jobs : int
        Number of parallel jobs (default: 16)
    force_rebuild : bool
        If True, rebuild cache even if it exists (default: False)
    
    Returns:
    --------
    data_dict, x_map, y_map, fp_map : tuple
        Loaded data, extracted SMILES/targets, and fingerprints
    """
    
    print("=== Molecular Data Processing Pipeline ===\n")
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    start_time = time.time()
    data_dict = load_split_data(base_dir, use_parallel=use_parallel)
    print(f"Data loaded in {time.time() - start_time:.2f}s")
    
    # Print summary
    total_datasets = sum(len(phases.get('train', {})) + len(phases.get('test', {})) 
                        for phases in data_dict.values())
    print(f"Loaded {len(data_dict)} splits with {total_datasets} total datasets")
    
    # Step 2: Extract X and y
    print("\nStep 2: Extracting SMILES and target values...")
    start_time = time.time()
    x_map, y_map = extract_xy_from_data(data_dict, use_parallel=use_parallel)
    print(f"Extraction completed in {time.time() - start_time:.2f}s")
    
    # Step 3: Generate fingerprints
    print("\nStep 3: Generating molecular fingerprints...")
    print(f"Cache: {'Enabled' if use_cache else 'Disabled'}")
    print(f"Force rebuild: {'Yes' if force_rebuild else 'No'}")
    start_time = time.time()
    fp_map = build_fingerprints_for_splits(
        x_map, 
        Path(out_root), 
        n_jobs=n_jobs,
        force_rebuild=force_rebuild,
        use_cache=use_cache,
        use_parallel=use_parallel
    )
    print(f"Fingerprint generation completed in {time.time() - start_time:.2f}s")
    
    # Step 4: Check cache status
    if use_cache:
        print("\nStep 4: Checking cache status...")
        check_cache_status(Path(out_root))
    
    return data_dict, x_map, y_map, fp_map

# ===========================
# Command Line Interface
# ===========================

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Molecular Data Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python molecular_data_processor.py
  
  # Run with custom directories
  python molecular_data_processor.py --base-dir data/preprocessed --out-dir data/fingerprints
  
  # Force rebuild cache
  python molecular_data_processor.py --force-rebuild
  
  # Run without cache
  python molecular_data_processor.py --no-cache
  
  # Run benchmarks only
  python molecular_data_processor.py --benchmark
  
  # Use sequential processing
  python molecular_data_processor.py --no-parallel
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='result/1_preprocess',
                       help='Base directory for preprocessed data')
    parser.add_argument('--out-dir', type=str, default='result/2_fingerprint',
                       help='Output directory for fingerprints')
    parser.add_argument('--n-jobs', type=int, default=16,
                       help='Number of parallel jobs')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild of fingerprint cache')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching system')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarks only')
    parser.add_argument('--check-cache', action='store_true',
                       help='Check cache status only')
    
    args = parser.parse_args()
    
    # Check cache status only
    if args.check_cache:
        check_cache_status(Path(args.out_dir))
        return
    
    # Run benchmarks only
    if args.benchmark:
        # Load some data for benchmarking
        print("Loading data for benchmarks...")
        data_dict = load_split_data(args.base_dir, use_parallel=False)
        x_map, _ = extract_xy_from_data(data_dict, use_parallel=False)
        
        # Get some SMILES for fingerprint benchmark
        all_smiles = []
        for datasets in x_map.values():
            for smiles_list in datasets.values():
                all_smiles.extend(smiles_list)
                if len(all_smiles) >= 1000:
                    break
            if len(all_smiles) >= 1000:
                break
        
        # Run benchmarks
        benchmark_loading(args.base_dir)
        if all_smiles:
            benchmark_fingerprints(all_smiles)
        return
    
    # Run full pipeline
    use_parallel = not args.no_parallel
    use_cache = not args.no_cache
    
    try:
        data_dict, x_map, y_map, fp_map = run_full_pipeline(
            base_dir=args.base_dir,
            out_root=args.out_dir,
            use_parallel=use_parallel,
            use_cache=use_cache,
            n_jobs=args.n_jobs,
            force_rebuild=args.force_rebuild
        )
        
        print("\n=== Pipeline completed successfully! ===")
        
        # Print summary statistics
        total_molecules = sum(len(smiles) for datasets in x_map.values() 
                            for smiles in datasets.values())
        total_fingerprints = sum(len(fps['morgan']) for split_fps in fp_map.values() 
                               for fps in split_fps.values())
        
        print(f"\nSummary:")
        print(f"- Total molecules processed: {total_molecules:,}")
        print(f"- Total fingerprints generated: {total_fingerprints:,}")
        print(f"- Output directory: {Path(args.out_dir).absolute()}")
        
    except Exception as e:
        print(f"\nError: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage when running directly
    if len(os.sys.argv) == 1:
        print("Running with example usage...")
        print("For command line options, use: python molecular_data_processor.py --help\n")
        
        # Example: Run pipeline with default settings
        data_dict, x_map, y_map, fp_map = run_full_pipeline()
        
        # Example: Access the results
        print("\n=== Example: Accessing Results ===")
        for split_name in list(x_map.keys())[:1]:  # Show first split
            print(f"\nSplit: {split_name}")
            for dataset_key in list(x_map[split_name].keys())[:2]:  # Show first 2 datasets
                n_molecules = len(x_map[split_name][dataset_key])
                print(f"  Dataset {dataset_key}: {n_molecules} molecules")
                if dataset_key in fp_map[split_name]:
                    fp_shapes = {
                        'Morgan': fp_map[split_name][dataset_key]['morgan'].shape,
                        'MACCS': fp_map[split_name][dataset_key]['maccs'].shape,
                        'Avalon': fp_map[split_name][dataset_key]['avalon'].shape
                    }
                    print(f"    Fingerprint shapes: {fp_shapes}")
    else:
        # Run with command line arguments
        exit(main())