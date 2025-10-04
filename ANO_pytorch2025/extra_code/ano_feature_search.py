#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer: Lee, Seungjin (arer90)

Chemical Descriptor Selection with Optuna - Complete Module
==========================================================

PURPOSE:
This module implements Optuna-based feature selection for molecular descriptors.
It systematically searches through 49 different descriptor categories to find
the optimal combination for molecular property prediction.

KEY FEATURES:
- Binary selection of descriptor categories (include/exclude)
- Parallel processing for efficient descriptor calculation
- 3D conformer generation with fallback mechanisms (via mol3d_generator module)
- Memory-efficient implementation for large datasets
- Integration with Optuna for Bayesian optimization

DESCRIPTOR CATEGORIES (Total: 49):
1. 2D Descriptors (0-27): 28 types
   - Basic molecular properties (MolWeight, MolLogP, MolMR)
   - Counts and connectivity indices (HBA, HBD, NumRotatableBonds)
   - Graph descriptors (BalabanJ, BertzCT, Chi indices)
   - Topological features (HallKierAlpha, Kappa shapes)
   
2. VSA Descriptors (28-32): 5 series
   - EState_VSA: Electronic state contributions to VSA
   - VSA_EState: Van der Waals surface area by E-state
   - SlogP_VSA: LogP contributions to surface area
   - SMR_VSA: Molar refractivity contributions
   - PEOE_VSA: Partial equalization of orbital electronegativity

3. rdMol Descriptors (33-35): 3 types
   - CalcAUTOCORR2D: 2D autocorrelation
   - CalcALOGP: Atom-based LogP
   - CalcAsphericity: Molecular asphericity

4. 3D Descriptors (36-48): 13 types
   - Geometric properties (PMI, NPR, RadiusOfGyration)
   - Shape descriptors (Asphericity, Eccentricity, SpherocityIndex)
   - Surface properties (InertialShapeFactor)
   - Requires valid 3D conformers

TOTAL FEATURES:
- Base fingerprints: Morgan (2048) + MACCS (167) + Avalon (512) = 2727
- Optional descriptors: Up to 49 categories (~882 individual features)
- Maximum features: ~3609 (depending on 3D availability)

USAGE:
This module is called by ANO modules (ano_4_FO.py, ano_5_MO.py, etc.) for 
feature optimization, where Optuna suggests which descriptor categories to 
include in each trial for optimal model performance.

DEPENDENCIES:
- mol3d_generator: Centralized 3D conformer generation
- RDKit: Molecular descriptor calculation
- Optuna: Hyperparameter optimization framework
"""

import os
import numpy as np
import pandas as pd
import gc
import platform
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Union, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdDistGeom, rdPartialCharges
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Descriptors import ExactMolWt

# OS-specific multiprocessing support
if platform.system() == "Windows":
    import multiprocessing
    multiprocessing.freeze_support()

# Import descriptor calculator
try:
    # Try direct import when in same directory
    from chem_descriptor_maker import ChemDescriptorCalculator
except ImportError:
    # Fallback for imports from parent directory
    from extra_code.chem_descriptor_maker import ChemDescriptorCalculator

def get_executor(use_thread=True, max_workers=None):
    """Get appropriate executor based on OS"""
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    if use_thread:
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor(max_workers=max_workers)
    else:
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor(max_workers=max_workers)

def clear_descriptor_memory(descriptor):
    """
    Clear descriptor memory for efficient memory management
    
    Parameters:
    -----------
    descriptor : any
        Descriptor object to be freed from memory
    
    Note:
    -----
    Used for memory management after large descriptor calculations
    """
    del descriptor
    gc.collect()

def Normalization(descriptor):
    """
    Normalize descriptor values using log transformation
    
    This function performs the following processing:
    1. Clip extreme values (-1e15 to 1e15)
    2. Replace near-zero values with epsilon
    3. Apply sign-preserving log transformation
    4. Handle NaN and Inf values by replacing with 0
    
    Parameters:
    -----------
    descriptor : array-like
        Descriptor values to normalize
    
    Returns:
    --------
    descriptor_log : np.ndarray
        Log-transformed normalized values
    
    Note:
    -----
    Used for stable processing of descriptors with large value ranges
    """
    descriptor = np.asarray(descriptor)
    epsilon = 1e-10
    max_value = 1e15
    descriptor = np.clip(descriptor, -max_value, max_value)
    descriptor_custom = np.where(np.abs(descriptor) < epsilon, epsilon, descriptor)
    descriptor_log = np.sign(descriptor_custom) * np.log1p(np.abs(descriptor_custom))
    descriptor_log = np.nan_to_num(descriptor_log, nan=0.0, posinf=0.0, neginf=0.0)
    del epsilon
    gc.collect()    
    return descriptor_log

def values_chi(mol, chi_type):
    """Calculate Chi descriptors for a molecule"""
    i = 0
    chi_func = Chem.GraphDescriptors.ChiNn_ if chi_type == 'n' else Chem.GraphDescriptors.ChiNv_
    while chi_func(mol, i) != 0.0:
        i += 1
    return np.array([chi_func(mol, j) for j in range(i)])

def generate_chi(mols, chi_type):
    """Generate Chi descriptors in parallel"""
    n_jobs = os.cpu_count()
    with get_executor(max_workers=n_jobs) as executor:
        futures = [executor.submit(values_chi, mol, chi_type) for mol in mols]
        descriptor = [future.result() for future in futures]
    
    max_length = max(len(x) for x in descriptor)
    padded_descriptor = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in descriptor])
    
    return padded_descriptor

def sanitize_and_compute_descriptor(mol):
    """Sanitize molecule and compute BCUT2D descriptor"""
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            return [0] * 8
        
        try:
            return rdMolDescriptors.BCUT2D(mol)
        except Exception as e:
            return [0] * 8
    except Exception as e:
        return [0] * 8

def compute_descriptors_parallel(mols, max_workers=8):
    """Compute descriptors in parallel"""
    with get_executor(max_workers=max_workers) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols]
        descriptors = [future.result() for future in futures]
    return descriptors

def generating_newfps(fps, descriptor, descriptor_name, save_res="np"):
    """Generate new fingerprints by concatenating existing ones with new descriptors"""
    try:
        if descriptor is None:
            return fps
        
        # Ensure descriptor has the same length as fps
        target_length = fps.shape[0]
        
        if save_res == "pd":
            new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps
            
            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                # Ensure descriptor has correct length
                if descriptor.shape[0] != target_length:
                    if descriptor.shape[0] < target_length:
                        # Pad with zeros
                        padding = np.zeros((target_length - descriptor.shape[0], descriptor.shape[1]))
                        descriptor = np.vstack([descriptor, padding])
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length, :]
                
                descriptors_df = pd.DataFrame(
                    {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
                )
                new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                del descriptor
                    
            elif isinstance(descriptor, list) and len(descriptor) > 0 and isinstance(descriptor[0], np.ndarray):
                # Process each array to ensure correct length
                processed_arrays = []
                for arr in descriptor:
                    if arr.ndim == 1:
                        if len(arr) != target_length:
                            if len(arr) < target_length:
                                # Pad with zeros
                                arr = np.pad(arr, (0, target_length - len(arr)), 'constant')
                            else:
                                # Truncate
                                arr = arr[:target_length]
                        processed_arrays.append(arr[:, None])
                    elif arr.ndim == 2:
                        if arr.shape[0] != target_length:
                            if arr.shape[0] < target_length:
                                # Pad with zeros
                                padding = np.zeros((target_length - arr.shape[0], arr.shape[1]))
                                arr = np.vstack([arr, padding])
                            else:
                                # Truncate
                                arr = arr[:target_length, :]
                        processed_arrays.append(arr)
                
                if processed_arrays:
                    combined = np.concatenate(processed_arrays, axis=1)
                    df = pd.DataFrame(
                        combined,
                        columns=[f'{descriptor_name}_{i+1}' for i in range(combined.shape[1])]
                    )
                    new_fps = pd.concat([new_fps, df], axis=1)
                
                del descriptor, processed_arrays
                    
            elif isinstance(descriptor, list):
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        # Pad with zeros
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length]
                if descriptor.ndim > 1:
                    descriptors_df = pd.DataFrame(
                        {f"{descriptor_name}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
                    )
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                else:
                    new_fps[descriptor_name] = descriptor.flatten()
                del descriptor
                
            new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
            return new_fps
            
        else:  # save_res == "np"
            new_fps = fps
            
            if descriptor is None:
                pass
            elif isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                # Ensure descriptor has correct length
                if descriptor.shape[0] != target_length:
                    if descriptor.shape[0] < target_length:
                        # Pad with zeros
                        padding = np.zeros((target_length - descriptor.shape[0], descriptor.shape[1]))
                        descriptor = np.vstack([descriptor, padding])
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length, :]
                
                new_fps = np.concatenate([new_fps, descriptor], axis=1)
                del descriptor
                
            elif isinstance(descriptor, list) and len(descriptor) > 0 and isinstance(descriptor[0], np.ndarray):
                # Process each array to ensure correct length
                processed_arrays = []
                for arr in descriptor:
                    if arr.ndim == 1:
                        if len(arr) != target_length:
                            if len(arr) < target_length:
                                # Pad with zeros
                                arr = np.pad(arr, (0, target_length - len(arr)), 'constant')
                            else:
                                # Truncate
                                arr = arr[:target_length]
                        processed_arrays.append(arr[:, None])
                    elif arr.ndim == 2:
                        if arr.shape[0] != target_length:
                            if arr.shape[0] < target_length:
                                # Pad with zeros
                                padding = np.zeros((target_length - arr.shape[0], arr.shape[1]))
                                arr = np.vstack([arr, padding])
                            else:
                                # Truncate
                                arr = arr[:target_length, :]
                        processed_arrays.append(arr)
                
                if processed_arrays:
                    combined = np.concatenate(processed_arrays, axis=1)
                    to_concat = [new_fps, combined]
                    new_fps = np.concatenate(to_concat, axis=1)
                
                del descriptor, processed_arrays
                
            else:
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        # Pad with zeros
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length]
                if descriptor.ndim == 1:
                    new_fps = np.concatenate([new_fps, descriptor[:,None]], axis=1)
                else:
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)
                del descriptor
                
            new_fps = np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float')
            return new_fps

    except Exception as e:
        print(f"Error occurred in {descriptor_name}: {e}")
        return fps

def load_cached_descriptors_for_search(dataset_name, split_type):
    """
    Load cached descriptors for dynamic feature search (like module 4)

    Args:
        dataset_name: Dataset name (e.g., "ws")
        split_type: Split type (e.g., "rm")

    Returns:
        tuple: (train_descriptors_dict, test_descriptors_dict, descriptor_names)
               Returns (None, None, None) if cache not found
    """
    import os
    import numpy as np

    # Standard cache path structure
    descriptor_dir = f"result/chemical_descriptors/{dataset_name}/{split_type}"
    train_desc_file = f"{descriptor_dir}/{dataset_name}_{split_type}_train_descriptors.npz"
    test_desc_file = f"{descriptor_dir}/{dataset_name}_{split_type}_test_descriptors.npz"

    # Check if cache files exist
    if not os.path.exists(train_desc_file) or not os.path.exists(test_desc_file):
        return None, None, None

    try:
        # Load cached descriptor data
        train_data = np.load(train_desc_file, allow_pickle=True)
        test_data = np.load(test_desc_file, allow_pickle=True)

        # Extract descriptor names (exclude metadata keys)
        descriptor_names = [key for key in train_data.keys()
                           if key not in ['train_mols', 'test_mols', 'descriptor_array', '3d_conformers']]

        return train_data, test_data, descriptor_names

    except Exception as e:
        print(f"Error loading cached descriptors for search: {e}")
        return None, None, None

def search_data_descriptor_compress(trial, fps, mols, dataset_name, split_type=None, target_path="result", save_res="np", mols_3d=None, **kwargs):
    """
    Enhanced version that uses cached descriptors for dynamic feature search (like module 4)

    This function dynamically selects descriptors using Optuna trial suggestions,
    but loads them from cache instead of calculating them in real-time.

    Args:
        trial: Optuna trial object
        fps: Base fingerprints
        mols: Molecules (can be None)
        dataset_name: Dataset name (e.g., "ws") OR full name (e.g., "ws_rm") for backward compatibility
        split_type: Split type (e.g., "rm") - optional if dataset_name contains both
        target_path: Output directory
        save_res: Save format
        mols_3d: 3D molecules
    """
    import numpy as np

    # Handle both new and old calling conventions
    if split_type is None:
        # Old convention: try to parse from dataset_name
        name = dataset_name
        dataset_name = None

        if isinstance(name, str):
            # Handle formats like "ws-rm", "ws_rm", etc.
            if '-' in name:
                parts = name.split('-')
                if len(parts) >= 2:
                    dataset_name, split_type = parts[0], parts[1]
            elif '_' in name:
                parts = name.split('_')
                if len(parts) >= 2:
                    dataset_name, split_type = parts[0], parts[1]
    else:
        # New convention: dataset and split_type provided separately
        name = f"{dataset_name}_{split_type}"

    # Try to load cached descriptors
    if dataset_name and split_type:
        train_data, test_data, cached_descriptor_names = load_cached_descriptors_for_search(dataset_name, split_type)

        if train_data is not None and test_data is not None:
            print(f"  ✅ Using cached descriptors for feature search from result/chemical_descriptors/{dataset_name}/{split_type}/")

            # Define descriptor-to-optuna mapping for dynamic selection
            descriptor_optuna_mapping = {
                'MolWt': 'MolWt', 'MolLogP': 'MolLogP', 'MolMR': 'MolMR', 'TPSA': 'TPSA',
                'NumRotatableBonds': 'NumRotatableBonds', 'HeavyAtomCount': 'HeavyAtomCount',
                'NumHAcceptors': 'NumHAcceptors', 'NumHDonors': 'NumHDonors',
                'NumHeteroatoms': 'NumHeteroatoms', 'NumValenceElectrons': 'NumValenceElectrons',
                'NHOHCount': 'NHOHCount', 'NOCount': 'NOCount', 'RingCount': 'RingCount',
                'NumAromaticRings': 'NumAromaticRings', 'NumSaturatedRings': 'NumSaturatedRings',
                'NumAliphaticRings': 'NumAliphaticRings', 'LabuteASA': 'LabuteASA',
                'BalabanJ': 'BalabanJ', 'BertzCT': 'BertzCT', 'Ipc': 'Ipc',
                'kappa_Series[1-3]_ind': 'kappa_Series[1-3]_ind', 'Chi_Series[1-98]_ind': 'Chi_Series[1-98]_ind',
                'Phi': 'Phi', 'HallKierAlpha': 'HallKierAlpha', 'NumAmideBonds': 'NumAmideBonds',
                'FractionCSP3': 'FractionCSP3', 'NumSpiroAtoms': 'NumSpiroAtoms',
                'NumBridgeheadAtoms': 'NumBridgeheadAtoms', 'PEOE_VSA_Series[1-14]_ind': 'PEOE_VSA_Series[1-14]_ind',
                'SMR_VSA_Series[1-10]_ind': 'SMR_VSA_Series[1-10]_ind', 'SlogP_VSA_Series[1-12]_ind': 'SlogP_VSA_Series[1-12]_ind',
                'EState_VSA_Series[1-11]_ind': 'EState_VSA_Series[1-11]_ind', 'VSA_EState_Series[1-10]_ind': 'VSA_EState_Series[1-10]_ind',
                'MQNs': 'MQNs', 'AUTOCORR2D': 'AUTOCORR2D', 'BCUT2D': 'BCUT2D',
                'Asphericity': 'Asphericity', 'InertialShapeFactor': 'InertialShapeFactor',
                'Eccentricity': 'Eccentricity', 'SpherocityIndex': 'SpherocityIndex',
                'RadiusOfGyration': 'RadiusOfGyration', 'VABC': 'VABC',
                'PMI_series[1-3]_ind': 'PMI_series[1-3]_ind', 'NPR_series[1-2]_ind': 'NPR_series[1-2]_ind',
                'AUTOCORR3D': 'AUTOCORR3D', 'RDF': 'RDF', 'MORSE': 'MORSE', 'WHIM': 'WHIM', 'GETAWAY': 'GETAWAY'
            }

            # Dynamic feature selection using trial suggestions
            selected_descriptors = []
            combined_fps_list = [fps]  # Start with base fingerprints

            for desc_name, optuna_name in descriptor_optuna_mapping.items():
                # Use trial to dynamically suggest whether to include this descriptor
                if trial.suggest_categorical(optuna_name, [0, 1]) == 1:
                    if desc_name in cached_descriptor_names:
                        # Load descriptor data from cache
                        train_desc = train_data[desc_name]
                        test_desc = test_data[desc_name]

                        # Combine train and test (assuming fps contains both)
                        if len(train_desc.shape) == 1:
                            train_desc = train_desc.reshape(-1, 1)
                        if len(test_desc.shape) == 1:
                            test_desc = test_desc.reshape(-1, 1)

                        combined_desc = np.vstack([train_desc, test_desc])
                        combined_fps_list.append(combined_desc)
                        selected_descriptors.append(desc_name)

            # Combine all selected features
            if len(combined_fps_list) > 1:
                combined_fps = np.hstack(combined_fps_list)
            else:
                combined_fps = fps

            print(f"  📊 Selected {len(selected_descriptors)} descriptors for search: {selected_descriptors[:5]}...")
            excluded_descriptors = [desc for desc in cached_descriptor_names if desc not in selected_descriptors]
            return combined_fps, selected_descriptors, excluded_descriptors

    # Fallback: use original calculation method
    print(f"  ⚠️ No cache found for search, falling back to real-time calculation...")
    return search_data_descriptor_compress_original(trial, fps, mols, name, target_path, save_res, mols_3d, dataset_name, split_type)

# Keep original function as fallback
def search_data_descriptor_compress_original(trial, fps, mols, name, target_path="result", save_res="np", mols_3d=None, dataset=None, split=None):
    """
    Main function for Optuna-driven chemical descriptor selection.

    This function implements the complete descriptor calculation for all 49 descriptors
    as defined in the original ANO framework.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object for hyperparameter suggestions
    fps : np.ndarray or pd.DataFrame
        Initial fingerprints (Morgan + MACCS + Avalon)
    mols : list
        List of RDKit mol objects
    name : str
        Name for saving results
    target_path : str
        Path to save results
    save_res : str
        Save format ("np" or "pd")
    mols_3d : list, optional
        Pre-generated 3D conformers

    Returns:
    --------
    tuple
        (fps_updated, selected_descriptors, excluded_descriptors)
    """
    selected_descriptors = []
    excluded_descriptors = []

    # Generate selection array from trial suggestions
    selection = np.zeros(49, dtype=int)

    # Get all descriptor names (same order as in ano_feature_selection.py)
    descriptor_names = [
        "MolWeight", "MolLogP", "MolMR", "TPSA", "NumRotatableBonds", "HeavyAtomCount", "HBA", "HBD",
        "NumHeteroatoms", "NumValenceElectrons", "NHOHCount", "NOCount", "RingCount", "AromaticRings",
        "SaturatedRings", "AliphaticRings", "LabuteASA", "BalabanJ", "BertzCT", "Ipc", "kappa_Series[1-3]_ind",
        "Phi", "HallKierAlpha", "NumAmideBonds", "FractionCSP3", "NumSpiroAtoms", "Chi0n-Chi4n", "Chi0v-Chi4v",
        "PEOE_VSA_Series[1-14]_ind", "SMR_VSA_Series[1-10]_ind", "SlogP_VSA_Series[1-12]_ind",
        "EState_VSA_Series[1-11]_ind", "VSA_EState_Series[1-10]_ind", "MQNs", "AUTOCORR2D", "ALOGP",
        "Asphericity", "PBF", "RadiusOfGyration", "InertialShapeFactor", "Eccentricity", "SpherocityIndex",
        "PMI_series[1-3]_ind", "NPR_series[1-2]_ind", "AUTOCORR3D", "RDF", "MORSE", "WHIM", "GETAWAY"
    ]

    # Generate selection from trial
    for i, desc_name in enumerate(descriptor_names):
        selection[i] = trial.suggest_categorical(desc_name, [0, 1])

    
    # === Part 2: 2D Descriptors (27 types) ===
    
    # 0. MolWeight
    if trial.suggest_categorical("MolWeight", [0, 1]) == 1:
        descriptor = [Descriptors.ExactMolWt(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolWeight', save_res)
        selected_descriptors.append("MolWeight")
        clear_descriptor_memory(descriptor)
    
    # 1. MolLogP
    if trial.suggest_categorical("MolLogP", [0, 1]) == 1:
        descriptor = [Chem.Crippen.MolLogP(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolLogP', save_res)
        selected_descriptors.append("MolLogP")
        clear_descriptor_memory(descriptor)
    
    # 2. MolMR
    if trial.suggest_categorical("MolMR", [0, 1]) == 1:
        descriptor = [Chem.Crippen.MolMR(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolMR', save_res)
        selected_descriptors.append("MolMR")
        clear_descriptor_memory(descriptor)
    
    # 3. TPSA
    if trial.suggest_categorical("TPSA", [0, 1]) == 1:
        descriptor = [Descriptors.TPSA(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'TPSA', save_res)
        selected_descriptors.append("TPSA")
        clear_descriptor_memory(descriptor)
    
    # 4. NumRotatableBonds
    if trial.suggest_categorical("NumRotatableBonds", [0, 1]) == 1:
        descriptor = [Lipinski.NumRotatableBonds(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumRotatableBonds', save_res)
        selected_descriptors.append("NumRotatableBonds")
        clear_descriptor_memory(descriptor)
    
    # 5. HeavyAtomCount
    if trial.suggest_categorical("HeavyAtomCount", [0, 1]) == 1:
        descriptor = [Lipinski.HeavyAtomCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'HeavyAtomCount', save_res)
        selected_descriptors.append("HeavyAtomCount")
        clear_descriptor_memory(descriptor)
    
    # 6. NumHAcceptors
    if trial.suggest_categorical("NumHAcceptors", [0, 1]) == 1:
        descriptor = [Lipinski.NumHAcceptors(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHAcceptors', save_res)
        selected_descriptors.append("NumHAcceptors")
        clear_descriptor_memory(descriptor)
    
    # 7. NumHDonors
    if trial.suggest_categorical("NumHDonors", [0, 1]) == 1:
        descriptor = [Lipinski.NumHDonors(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHDonors', save_res)
        selected_descriptors.append("NumHDonors")
        clear_descriptor_memory(descriptor)
    
    # 8. NumHeteroatoms
    if trial.suggest_categorical("NumHeteroatoms", [0, 1]) == 1:
        descriptor = [Lipinski.NumHeteroatoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHeteroatoms', save_res)
        selected_descriptors.append("NumHeteroatoms")
        clear_descriptor_memory(descriptor)
    
    # 9. NumValenceElectrons
    if trial.suggest_categorical("NumValenceElectrons", [0, 1]) == 1:
        descriptor = [Descriptors.NumValenceElectrons(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumValenceElectrons', save_res)
        selected_descriptors.append("NumValenceElectrons")
        clear_descriptor_memory(descriptor)
    
    # 10. NHOHCount
    if trial.suggest_categorical("NHOHCount", [0, 1]) == 1:
        descriptor = [Lipinski.NHOHCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NHOHCount', save_res)
        selected_descriptors.append("NHOHCount")
        clear_descriptor_memory(descriptor)
    
    # 11. NOCount
    if trial.suggest_categorical("NOCount", [0, 1]) == 1:
        descriptor = [Lipinski.NOCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NOCount', save_res)
        selected_descriptors.append("NOCount")
        clear_descriptor_memory(descriptor)
    
    # 12. RingCount
    if trial.suggest_categorical("RingCount", [0, 1]) == 1:
        descriptor = [Lipinski.RingCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'RingCount', save_res)
        selected_descriptors.append("RingCount")
        clear_descriptor_memory(descriptor)
    
    # 13. NumAromaticRings
    if trial.suggest_categorical("NumAromaticRings", [0, 1]) == 1:
        descriptor = [Lipinski.NumAromaticRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAromaticRings', save_res)
        selected_descriptors.append("NumAromaticRings")
        clear_descriptor_memory(descriptor)
    
    # 14. NumSaturatedRings
    if trial.suggest_categorical("NumSaturatedRings", [0, 1]) == 1:
        descriptor = [Lipinski.NumSaturatedRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumSaturatedRings', save_res)
        selected_descriptors.append("NumSaturatedRings")
        clear_descriptor_memory(descriptor)
    
    # 15. NumAliphaticRings
    if trial.suggest_categorical("NumAliphaticRings", [0, 1]) == 1:
        descriptor = [Lipinski.NumAliphaticRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAliphaticRings', save_res)
        selected_descriptors.append("NumAliphaticRings")
        clear_descriptor_memory(descriptor)
    
    # 16. LabuteASA
    if trial.suggest_categorical("LabuteASA", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcLabuteASA(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'LabuteASA', save_res)
        selected_descriptors.append("LabuteASA")
        clear_descriptor_memory(descriptor)
    
    # 17. BalabanJ
    if trial.suggest_categorical("BalabanJ", [0, 1]) == 1:
        descriptor = [Chem.GraphDescriptors.BalabanJ(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'BalabanJ', save_res)
        selected_descriptors.append("BalabanJ")
        clear_descriptor_memory(descriptor)
    
    # 18. BertzCT
    if trial.suggest_categorical("BertzCT", [0, 1]) == 1:
        descriptor = [Chem.GraphDescriptors.BertzCT(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'BertzCT', save_res)
        selected_descriptors.append("BertzCT")
        clear_descriptor_memory(descriptor)
    
    # 19. Ipc
    if trial.suggest_categorical("Ipc", [0, 1]) == 1:
        descriptor = [Chem.GraphDescriptors.Ipc(mol) for mol in mols]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'Ipc', save_res)
        selected_descriptors.append("Ipc")
        clear_descriptor_memory(descriptor)
    
    # 20. kappa_Series[1-3]_ind - Kappa indices
    if trial.suggest_categorical("kappa_Series[1-3]_ind", [0, 1]) == 1:
        d1 = [Chem.GraphDescriptors.Kappa1(mol) for mol in mols]
        d2 = [Chem.GraphDescriptors.Kappa2(mol) for mol in mols]
        d3 = [Chem.GraphDescriptors.Kappa3(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        fps = generating_newfps(fps, [d1,d2,d3], 'kappa_Series[1-3]_ind', save_res)
        selected_descriptors.append("kappa_Series[1-3]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    
    # 21. Chi_Series[1-98]_ind - Chi indices
    if trial.suggest_categorical("Chi_Series[1-98]_ind", [0, 1]) == 1:
        descriptor1 = generate_chi(mols, 'n')
        descriptor2 = generate_chi(mols, 'v')
        fps = generating_newfps(fps, descriptor1, 'Chi_Series_n', save_res)
        fps = generating_newfps(fps, descriptor2, 'Chi_Series_v', save_res)
        selected_descriptors.append("Chi_Series[1-98]_ind")
        clear_descriptor_memory(descriptor1)
        clear_descriptor_memory(descriptor2)
    
    # 22. Phi
    if trial.suggest_categorical("Phi", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcPhi(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'Phi', save_res)
        selected_descriptors.append("Phi")
        clear_descriptor_memory(descriptor)
    
    # 23. HallKierAlpha
    if trial.suggest_categorical("HallKierAlpha", [0, 1]) == 1:
        descriptor = [Chem.GraphDescriptors.HallKierAlpha(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'HallKierAlpha', save_res)
        selected_descriptors.append("HallKierAlpha")
        clear_descriptor_memory(descriptor)
    
    # 24. NumAmideBonds
    if trial.suggest_categorical("NumAmideBonds", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcNumAmideBonds(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAmideBonds', save_res)
        selected_descriptors.append("NumAmideBonds")
        clear_descriptor_memory(descriptor)
    
    # 25. FractionCSP3
    if trial.suggest_categorical("FractionCSP3", [0, 1]) == 1:
        descriptor = [Lipinski.FractionCSP3(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'FractionCSP3', save_res)
        selected_descriptors.append("FractionCSP3")
        clear_descriptor_memory(descriptor)
    
    # 26. NumSpiroAtoms
    if trial.suggest_categorical("NumSpiroAtoms", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcNumSpiroAtoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumSpiroAtoms', save_res)
        selected_descriptors.append("NumSpiroAtoms")
        clear_descriptor_memory(descriptor)
    
    # 27. NumBridgeheadAtoms
    if trial.suggest_categorical("NumBridgeheadAtoms", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcNumBridgeheadAtoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumBridgeheadAtoms', save_res)
        selected_descriptors.append("NumBridgeheadAtoms")
        clear_descriptor_memory(descriptor)
    
    # === Part 3: VSA Descriptors (5 series) ===
    
    # 28. PEOE_VSA_Series[1-14]_ind - PEOE VSA (14 individual descriptors)
    if trial.suggest_categorical("PEOE_VSA_Series[1-14]_ind", [0, 1]) == 1:
        d1 = [Chem.MolSurf.PEOE_VSA1(mol) for mol in mols]
        d2 = [Chem.MolSurf.PEOE_VSA2(mol) for mol in mols]
        d3 = [Chem.MolSurf.PEOE_VSA3(mol) for mol in mols]
        d4 = [Chem.MolSurf.PEOE_VSA4(mol) for mol in mols]
        d5 = [Chem.MolSurf.PEOE_VSA5(mol) for mol in mols]
        d6 = [Chem.MolSurf.PEOE_VSA6(mol) for mol in mols]
        d7 = [Chem.MolSurf.PEOE_VSA7(mol) for mol in mols]
        d8 = [Chem.MolSurf.PEOE_VSA8(mol) for mol in mols]
        d9 = [Chem.MolSurf.PEOE_VSA9(mol) for mol in mols]
        d10 = [Chem.MolSurf.PEOE_VSA10(mol) for mol in mols]
        d11 = [Chem.MolSurf.PEOE_VSA11(mol) for mol in mols]
        d12 = [Chem.MolSurf.PEOE_VSA12(mol) for mol in mols]
        d13 = [Chem.MolSurf.PEOE_VSA13(mol) for mol in mols]
        d14 = [Chem.MolSurf.PEOE_VSA14(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        d4 = np.asarray(d4)
        d5 = np.asarray(d5)
        d6 = np.asarray(d6)
        d7 = np.asarray(d7)
        d8 = np.asarray(d8)
        d9 = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        d13 = np.asarray(d13)
        d14 = np.asarray(d14)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14], 
                               'PEOE_VSA_Series[1-14]_ind', save_res)
        selected_descriptors.append("PEOE_VSA_Series[1-14]_ind")
        for d in [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14]:
            clear_descriptor_memory(d)
    
    # 29. SMR_VSA_Series[1-10]_ind - SMR VSA (10 individual descriptors)
    if trial.suggest_categorical("SMR_VSA_Series[1-10]_ind", [0, 1]) == 1:
        d1 = [Chem.MolSurf.SMR_VSA1(mol) for mol in mols]
        d2 = [Chem.MolSurf.SMR_VSA2(mol) for mol in mols]
        d3 = [Chem.MolSurf.SMR_VSA3(mol) for mol in mols]
        d4 = [Chem.MolSurf.SMR_VSA4(mol) for mol in mols]
        d5 = [Chem.MolSurf.SMR_VSA5(mol) for mol in mols]
        d6 = [Chem.MolSurf.SMR_VSA6(mol) for mol in mols]
        d7 = [Chem.MolSurf.SMR_VSA7(mol) for mol in mols]
        d8 = [Chem.MolSurf.SMR_VSA8(mol) for mol in mols]
        d9 = [Chem.MolSurf.SMR_VSA9(mol) for mol in mols]
        d10 = [Chem.MolSurf.SMR_VSA10(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        d4 = np.asarray(d4)
        d5 = np.asarray(d5)
        d6 = np.asarray(d6)
        d7 = np.asarray(d7)
        d8 = np.asarray(d8)
        d9 = np.asarray(d9)
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10], 
                               'SMR_VSA_Series[1-10]_ind', save_res)
        selected_descriptors.append("SMR_VSA_Series[1-10]_ind")
        for d in [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]:
            clear_descriptor_memory(d)
    
    # 30. SlogP_VSA_Series[1-12]_ind - SlogP VSA (12 individual descriptors)
    if trial.suggest_categorical("SlogP_VSA_Series[1-12]_ind", [0, 1]) == 1:
        d1 = [Chem.MolSurf.SlogP_VSA1(mol) for mol in mols]
        d2 = [Chem.MolSurf.SlogP_VSA2(mol) for mol in mols]
        d3 = [Chem.MolSurf.SlogP_VSA3(mol) for mol in mols]
        d4 = [Chem.MolSurf.SlogP_VSA4(mol) for mol in mols]
        d5 = [Chem.MolSurf.SlogP_VSA5(mol) for mol in mols]
        d6 = [Chem.MolSurf.SlogP_VSA6(mol) for mol in mols]
        d7 = [Chem.MolSurf.SlogP_VSA7(mol) for mol in mols]
        d8 = [Chem.MolSurf.SlogP_VSA8(mol) for mol in mols]
        d9 = [Chem.MolSurf.SlogP_VSA9(mol) for mol in mols]
        d10 = [Chem.MolSurf.SlogP_VSA10(mol) for mol in mols]
        d11 = [Chem.MolSurf.SlogP_VSA11(mol) for mol in mols]
        d12 = [Chem.MolSurf.SlogP_VSA12(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        d4 = np.asarray(d4)
        d5 = np.asarray(d5)
        d6 = np.asarray(d6)
        d7 = np.asarray(d7)
        d8 = np.asarray(d8)
        d9 = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12], 
                               'SlogP_VSA_Series[1-12]_ind', save_res)
        selected_descriptors.append("SlogP_VSA_Series[1-12]_ind")
        for d in [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12]:
            clear_descriptor_memory(d)
    
    # 31. EState_VSA_Series[1-11]_ind - EState VSA (11 individual descriptors)
    if trial.suggest_categorical("EState_VSA_Series[1-11]_ind", [0, 1]) == 1:
        d1 = [Chem.EState.EState_VSA.EState_VSA1(mol) for mol in mols]
        d2 = [Chem.EState.EState_VSA.EState_VSA2(mol) for mol in mols]
        d3 = [Chem.EState.EState_VSA.EState_VSA3(mol) for mol in mols]
        d4 = [Chem.EState.EState_VSA.EState_VSA4(mol) for mol in mols]
        d5 = [Chem.EState.EState_VSA.EState_VSA5(mol) for mol in mols]
        d6 = [Chem.EState.EState_VSA.EState_VSA6(mol) for mol in mols]
        d7 = [Chem.EState.EState_VSA.EState_VSA7(mol) for mol in mols]
        d8 = [Chem.EState.EState_VSA.EState_VSA8(mol) for mol in mols]
        d9 = [Chem.EState.EState_VSA.EState_VSA9(mol) for mol in mols]
        d10 = [Chem.EState.EState_VSA.EState_VSA10(mol) for mol in mols]
        d11 = [Chem.EState.EState_VSA.EState_VSA11(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        d4 = np.asarray(d4)
        d5 = np.asarray(d5)
        d6 = np.asarray(d6)
        d7 = np.asarray(d7)
        d8 = np.asarray(d8)
        d9 = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11], 
                               'EState_VSA_Series[1-11]_ind', save_res)
        selected_descriptors.append("EState_VSA_Series[1-11]_ind")
        for d in [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11]:
            clear_descriptor_memory(d)
    
    # 32. VSA_EState_Series[1-10]_ind - VSA EState (10 individual descriptors)
    if trial.suggest_categorical("VSA_EState_Series[1-10]_ind", [0, 1]) == 1:
        d1 = [Chem.EState.EState_VSA.VSA_EState1(mol) for mol in mols]
        d2 = [Chem.EState.EState_VSA.VSA_EState2(mol) for mol in mols]
        d3 = [Chem.EState.EState_VSA.VSA_EState3(mol) for mol in mols]
        d4 = [Chem.EState.EState_VSA.VSA_EState4(mol) for mol in mols]
        d5 = [Chem.EState.EState_VSA.VSA_EState5(mol) for mol in mols]
        d6 = [Chem.EState.EState_VSA.VSA_EState6(mol) for mol in mols]
        d7 = [Chem.EState.EState_VSA.VSA_EState7(mol) for mol in mols]
        d8 = [Chem.EState.EState_VSA.VSA_EState8(mol) for mol in mols]
        d9 = [Chem.EState.EState_VSA.VSA_EState9(mol) for mol in mols]
        d10 = [Chem.EState.EState_VSA.VSA_EState10(mol) for mol in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        d4 = np.asarray(d4)
        d5 = np.asarray(d5)
        d6 = np.asarray(d6)
        d7 = np.asarray(d7)
        d8 = np.asarray(d8)
        d9 = np.asarray(d9)
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10], 
                               'VSA_EState_Series[1-10]_ind', save_res)
        selected_descriptors.append("VSA_EState_Series[1-10]_ind")
        for d in [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10]:
            clear_descriptor_memory(d)
    
    # 33. MQNs - Molecular Quantum Numbers
    if trial.suggest_categorical("MQNs", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.MQNs_(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MQNs', save_res)
        selected_descriptors.append("MQNs")
        clear_descriptor_memory(descriptor)
    
    # 34. AUTOCORR2D - 2D Autocorrelation
    if trial.suggest_categorical("AUTOCORR2D", [0, 1]) == 1:
        descriptor = [rdMolDescriptors.CalcAUTOCORR2D(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR2D', save_res)
        selected_descriptors.append("AUTOCORR2D")
        clear_descriptor_memory(descriptor)
    
    # 35. BCUT2D - 2D BCUT Descriptors
    if trial.suggest_categorical("BCUT2D", [0, 1]) == 1:
        descriptor = compute_descriptors_parallel(mols)
        fps = generating_newfps(fps, descriptor, 'BCUT2D', save_res)
        selected_descriptors.append("BCUT2D")
        clear_descriptor_memory(descriptor)
    
    # === Part 4: 3D Descriptors (13 types) ===

    # Check if 3D descriptors are already cached to avoid conformer generation
    def check_3d_descriptors_cached(dataset_name, split_name, full_name):
        """Check if required 3D descriptors are already cached using config.py structure"""
        import os

        # Standard cache paths (relative to working directory)
        # Structure: result/chemical_descriptors/dataset/split_type/
        cache_base = "result/chemical_descriptors"

        if not os.path.exists(cache_base):
            return False

        # Check for cached descriptors for this specific dataset/split combination
        if dataset_name and split_name:
            cache_path = os.path.join(cache_base, dataset_name, split_name)
            if os.path.exists(cache_path):
                files = os.listdir(cache_path)
                # Check if descriptor files exist
                if any(f.endswith('_descriptors.npz') for f in files):
                    return True

        # Also check using the name format if dataset/split not provided
        if full_name:
            # Try to extract dataset/split from name
            parts = full_name.replace('-', '_').split('_')
            if len(parts) >= 2:
                ds = parts[0]
                sp = parts[1]
                cache_path = os.path.join(cache_base, ds, sp)
                if os.path.exists(cache_path):
                    files = os.listdir(cache_path)
                    if any(f.endswith('_descriptors.npz') for f in files):
                        return True

        return False  # No cached descriptors found for this dataset/split

    # Generate 3D conformers if not provided and not cached
    if mols_3d is None:
        if check_3d_descriptors_cached(dataset, split, name):
            print("  3D descriptors found in cache, skipping conformer generation...")
            # Create dummy mols_3d to avoid None issues
            mols_3d = [None] * fps.shape[0]
        else:
            print("  Generating 3D conformers for 3D descriptor calculation...")
            # Use ChemDescriptorCalculator to generate 3D conformers
            calc = ChemDescriptorCalculator()
            mols_3d = calc.process_molecules_parallel(mols, max_workers=8)
            # Ensure mols_3d has the same length as fps
            if len(mols_3d) != fps.shape[0]:
                print(f"  Warning: 3D conformers count ({len(mols_3d)}) doesn't match fps shape ({fps.shape[0]})")
                # Pad with None values to match fps length
                while len(mols_3d) < fps.shape[0]:
                    mols_3d.append(None)
                mols_3d = mols_3d[:fps.shape[0]]
    
    def safe_3d_descriptor(func, mol):
        """Safely compute 3D descriptor"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                return func(mol)
        except:
            pass
        return 0.0
    
    def compute_3d_descriptor_list(mols_3d, func, name):
        """Compute 3D descriptor for list of molecules"""
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = func(mol)
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        descriptor.append(val)
                    else:
                        descriptor.append(0.0)
                else:
                    descriptor.append(0.0)
            except Exception as e:
                descriptor.append(0.0)
        return descriptor
    
    # 36. Asphericity
    if trial.suggest_categorical("Asphericity", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcAsphericity, "Asphericity")
        fps = generating_newfps(fps, descriptor, 'Asphericity', save_res)
        selected_descriptors.append("Asphericity")
        clear_descriptor_memory(descriptor)
    
    # 37. PBF - Principal Moments of Inertia B
    if trial.suggest_categorical("PBF", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcPBF, "PBF")
        fps = generating_newfps(fps, descriptor, 'PBF', save_res)
        selected_descriptors.append("PBF")
        clear_descriptor_memory(descriptor)
    
    # 38. RadiusOfGyration
    if trial.suggest_categorical("RadiusOfGyration", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcRadiusOfGyration, "RadiusOfGyration")
        fps = generating_newfps(fps, descriptor, 'RadiusOfGyration', save_res)
        selected_descriptors.append("RadiusOfGyration")
        clear_descriptor_memory(descriptor)
    
    # 39. InertialShapeFactor
    if trial.suggest_categorical("InertialShapeFactor", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcInertialShapeFactor, "InertialShapeFactor")
        fps = generating_newfps(fps, descriptor, 'InertialShapeFactor', save_res)
        selected_descriptors.append("InertialShapeFactor")
        clear_descriptor_memory(descriptor)
    
    # 40. Eccentricity
    if trial.suggest_categorical("Eccentricity", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcEccentricity, "Eccentricity")
        fps = generating_newfps(fps, descriptor, 'Eccentricity', save_res)
        selected_descriptors.append("Eccentricity")
        clear_descriptor_memory(descriptor)
    
    # 41. SpherocityIndex
    if trial.suggest_categorical("SpherocityIndex", [0, 1]) == 1:
        descriptor = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcSpherocityIndex, "SpherocityIndex")
        fps = generating_newfps(fps, descriptor, 'SpherocityIndex', save_res)
        selected_descriptors.append("SpherocityIndex")
        clear_descriptor_memory(descriptor)
    
    # 42. PMI_series[1-3]_ind - Principal Moments of Inertia (3 descriptors)
    if trial.suggest_categorical("PMI_series[1-3]_ind", [0, 1]) == 1:
        d1 = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcPMI1, "PMI1")
        d2 = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcPMI2, "PMI2")
        d3 = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcPMI3, "PMI3")
        d1 = Normalization(d1)
        d2 = Normalization(d2)
        d3 = Normalization(d3)
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        fps = generating_newfps(fps, [d1,d2,d3], 'PMI_series[1-3]_ind', save_res)
        selected_descriptors.append("PMI_series[1-3]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    
    # 43. NPR_series[1-2]_ind - Normalized Principal Moments Ratio (2 descriptors)
    if trial.suggest_categorical("NPR_series[1-2]_ind", [0, 1]) == 1:
        d1 = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcNPR1, "NPR1")
        d2 = compute_3d_descriptor_list(mols_3d, rdMolDescriptors.CalcNPR2, "NPR2")
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        fps = generating_newfps(fps, [d1,d2], 'NPR_series[1-2]_ind', save_res)
        selected_descriptors.append("NPR_series[1-2]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
    
    # 44. AUTOCORR3D - 3D Autocorrelation
    if trial.suggest_categorical("AUTOCORR3D", [0, 1]) == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcAUTOCORR3D(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 80)  # Default size for AUTOCORR3D
            except:
                descriptor.append([0] * 80)
        fps = generating_newfps(fps, descriptor, 'AUTOCORR3D', save_res)
        selected_descriptors.append("AUTOCORR3D")
        clear_descriptor_memory(descriptor)
    
    # 45. RDF - Radial Distribution Function
    if trial.suggest_categorical("RDF", [0, 1]) == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcRDF(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 210)  # Default size for RDF
            except:
                descriptor.append([0] * 210)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'RDF', save_res)
        selected_descriptors.append("RDF")
        clear_descriptor_memory(descriptor)
    
    # 46. MORSE - Morse Descriptors
    if trial.suggest_categorical("MORSE", [0, 1]) == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcMORSE(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 224)  # Default size for MORSE
            except:
                descriptor.append([0] * 224)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'MORSE', save_res)
        selected_descriptors.append("MORSE")
        clear_descriptor_memory(descriptor)
    
    # 47. WHIM - WHIM Descriptors
    if trial.suggest_categorical("WHIM", [0, 1]) == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcWHIM(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 114)  # Default size for WHIM
            except:
                descriptor.append([0] * 114)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'WHIM', save_res)
        selected_descriptors.append("WHIM")
        clear_descriptor_memory(descriptor)
    
    # 48. GETAWAY - GETAWAY Descriptors
    if trial.suggest_categorical("GETAWAY", [0, 1]) == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcGETAWAY(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 273)  # Default size for GETAWAY
            except:
                descriptor.append([0] * 273)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'GETAWAY', save_res)
        selected_descriptors.append("GETAWAY")
        clear_descriptor_memory(descriptor)
    

    # Save results if requested
    if save_res == "pd":
        fps.to_csv(f'{target_path}/{name}_feature_selection.csv')

    # Ensure final data type
    fps = fps.astype('float32')

    return fps, selected_descriptors, excluded_descriptors


