#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer: Lee, Seungjin (arer90)

Feature Selection Combined: PyTorch-Compatible Feature Selection Module
======================================================================

PURPOSE:
This module provides comprehensive molecular descriptor calculation and selection
capabilities for the ANO optimization framework. It combines multiple descriptor
types and enables efficient feature selection through Optuna optimization.

KEY COMPONENTS:
1. 2D Molecular Descriptors (28 types): Basic molecular properties
2. VSA Descriptors (5 series): Surface-based properties
3. rdMol Descriptors (3 types): RDKit molecular descriptors
4. 3D Molecular Descriptors (13 types): Conformation-dependent properties
5. Feature Selection Functions: Modular processing for optimal descriptor combinations

DESCRIPTOR CATEGORIES (49 total):

[2D Descriptors: 0-27]
- Basic properties: MolWeight(0), LogP(1), MolMR(2), TPSA(3)
- Counts: NumRotatableBonds(4), HeavyAtomCount(5), HBA(6), HBD(7)
- Heteroatoms: NumHeteroatoms(8), NumValenceElectrons(9)
- Functional groups: NHOHCount(10), NOCount(11)
- Rings: RingCount(12), AromaticRings(13), SaturatedRings(14), AliphaticRings(15)
- Surface: LabuteASA(16)
- Connectivity: BalabanJ(17), BertzCT(18), Ipc(19)
- Shape indices: Kappa1-3(20), Chi0n-Chi4n(21), Chi0v-Chi4v(22)
- Others: Phi(23), HallKierAlpha(24), NumAmideBonds(25), FractionCSP3(26), NumSpiroAtoms(27)

[VSA Descriptors: 28-32]
- PEOE_VSA(28): Partial charge VSA (14 bins)
- SMR_VSA(29): Molar refractivity VSA (10 bins)
- SlogP_VSA(30): LogP VSA (12 bins)
- EState_VSA(31): E-state VSA (11 bins)
- VSA_EState(32): VSA by E-state (10 bins)

[rdMol Descriptors: 33-35]
- CalcAUTOCORR2D(33): 2D autocorrelation (192 values)
- CalcALOGP(34): Atom-based LogP
- CalcAsphericity(35): Molecular asphericity

[3D Descriptors: 36-48] - Requires valid 3D conformers
- Geometric: PBF(36), RadiusOfGyration(37), InertialShapeFactor(38)
- Shape: Asphericity(39), Eccentricity(40), SpherocityIndex(41)
- PMI series(42-44): Principal moments of inertia
- NPR series(45-46): Normalized principal moments ratios
- Advanced: AUTOCORR3D(47), RDF(48), MORSE(49), WHIM(50), GETAWAY(51)

MODULAR FUNCTIONS:
- process_2d_descriptors_selection(): Handles 2D descriptors (0-27)
- process_vsa_descriptors_selection(): Handles VSA descriptors (28-32)
- process_rdmol_descriptors_selection(): Handles rdMol descriptors (33-35)
- process_3d_descriptors_selection(): Handles 3D descriptors (36-48)

USAGE:
This module is used by ANO modules (ano_4_FO.py through ano_7_NO2.py) for 
feature selection optimization. It provides modular functions to calculate 
descriptors and combine them with fingerprints for optimal model performance.

DEPENDENCIES:
- mol3d_generator: Centralized 3D conformer generation
- RDKit: Molecular descriptor calculation
- Optuna: Feature selection optimization
- PyTorch (optional): Neural network compatibility
"""

import os
import numpy as np
import pandas as pd
import gc
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import Union, List, Optional
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdDistGeom, rdPartialCharges
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Descriptors import ExactMolWt

# PyTorch imports (optional, for compatibility)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

import optuna

# Import model optimization module for creating models with best architecture
try:
    from ano_model_optimization import create_model_from_params, OptimizedDNN
    MODEL_OPT_AVAILABLE = True
except ImportError:
    MODEL_OPT_AVAILABLE = False

# Import config for descriptor names
try:
    from config import CHEMICAL_DESCRIPTORS
except ImportError:
    # Default descriptor names if config not available
    CHEMICAL_DESCRIPTORS = [
        'MolWeight', 'MolLogP', 'MolMR', 'TPSA', 'NumRotatableBonds',
        'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
        'NumValenceElectrons', 'NHOHCount', 'NOCount', 'RingCount',
        'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
        'LabuteASA', 'BalabanJ', 'BertzCT', 'Ipc', 'kappa_Series[1-3]_ind',
        'Chi_Series[13]_ind', 'Phi', 'HallKierAlpha', 'NumAmideBonds',
        'FractionCSP3', 'NumSpiroAtoms', 'NumBridgeheadAtoms',
        'PEOE_VSA_Series[1-14]_ind', 'SMR_VSA_Series[1-10]_ind',
        'SlogP_VSA_Series[1-12]_ind', 'EState_VSA_Series[1-11]_ind',
        'VSA_EState_Series[1-10]_ind', 'MQNs', 'AUTOCORR2D', 'BCUT2D',
        'Asphericity', 'PBF', 'RadiusOfGyration', 'InertialShapeFactor',
        'Eccentricity', 'SpherocityIndex', 'PMI_series[1-3]_ind',
        'NPR_series[1-2]_ind', 'AUTOCORR3D', 'RDF', 'MORSE', 'WHIM', 'GETAWAY'
    ]

# 3D conformer generation functions (inline implementations)
def process_molecules_parallel(mols, max_workers=4):
    """Simple 3D conformer generation without external dependencies"""
    result_mols = []
    for mol in mols:
        if mol is not None:
            try:
                mol_copy = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol_copy)
                result_mols.append(mol_copy)
            except:
                result_mols.append(None)
        else:
            result_mols.append(None)
    return result_mols

def clear_3d_memory():
    """Memory cleanup"""
    import gc
    gc.collect()

def process_chunk_optimized(chunk_data):
    """
    Process data chunks in parallel to generate DataFrame
    
    Parameters:
    -----------
    chunk_data : tuple
        Tuple in format (chunk, name_prefix, start_idx)
        - chunk: Data chunk to process
        - name_prefix: Column name prefix
        - start_idx: Starting index
    
    Returns:
    --------
    pd.DataFrame
        Processed chunk as DataFrame
    """
    chunk, name_prefix, start_idx = chunk_data
    return pd.DataFrame(
        chunk,
        columns=[f"{name_prefix}_{j+1}" for j in range(start_idx, start_idx + chunk.shape[1])]
    )

def generate_df_concurrently(descriptor: np.ndarray, name_prefix: str, chunk_size: int = 1000) -> Optional[pd.DataFrame]:
    """Generate DataFrame concurrently for large descriptors"""
    try:
        chunks = [
            (descriptor[:, i:min(i + chunk_size, descriptor.shape[1])], name_prefix, i)
            for i in range(0, descriptor.shape[1], chunk_size)
        ]
        
        with ProcessPoolExecutor() as executor:
            chunk_dfs = list(executor.map(process_chunk_optimized, chunks))
            
        return pd.concat(chunk_dfs, axis=1) if chunk_dfs else None
        
    except Exception as e:
        print(f"[-1-] Error in generating DataFrame concurrently: {e}")
        return pd.DataFrame(
            {f"{name_prefix}_{i+1}": descriptor[:, i] for i in range(descriptor.shape[1])}
        )

def generating_newfps(
    fps: Union[np.ndarray, pd.DataFrame],
    descriptor: Optional[Union[np.ndarray, List[np.ndarray], List[List]]],
    descriptor_name: str,
    save_res: str = "np",
    chunk_size: int = 1000
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Generate new fingerprints by combining existing ones with new descriptors.
    
    This function efficiently combines molecular fingerprints with calculated
    descriptors, handling large datasets through chunked processing.
    
    Parameters:
    -----------
    fps : np.ndarray or pd.DataFrame
        Existing fingerprints (base features)
    descriptor : np.ndarray, List, or None
        New descriptors to add
    descriptor_name : str
        Name prefix for descriptor columns
    save_res : str
        Output format: 'np' for numpy array, 'pd' for pandas DataFrame
    chunk_size : int
        Size of chunks for parallel processing
    
    Returns:
    --------
    combined : np.ndarray or pd.DataFrame
        Combined fingerprints and descriptors
    """
    try:
        if descriptor is None:
            return fps

        target_length = fps.shape[0] if hasattr(fps, 'shape') else len(fps)

        if save_res == "pd":
            new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps

            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                # Ensure correct length
                if descriptor.shape[0] != target_length:
                    if descriptor.shape[0] < target_length:
                        padding = np.zeros((target_length - descriptor.shape[0], descriptor.shape[1]))
                        descriptor = np.vstack([descriptor, padding])
                    else:
                        descriptor = descriptor[:target_length, :]
                
                descriptors_df = generate_df_concurrently(descriptor, descriptor_name, chunk_size)
                if descriptors_df is not None:
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)

            elif isinstance(descriptor, list) and len(descriptor) > 0 and isinstance(descriptor[0], np.ndarray):
                # Handle list of arrays
                processed_arrays = []
                for arr in descriptor:
                    if arr.ndim == 1:
                        if len(arr) != target_length:
                            if len(arr) < target_length:
                                arr = np.pad(arr, (0, target_length - len(arr)), 'constant')
                            else:
                                arr = arr[:target_length]
                        processed_arrays.append(arr.reshape(-1, 1))
                    else:
                        if arr.shape[0] != target_length:
                            if arr.shape[0] < target_length:
                                padding = np.zeros((target_length - arr.shape[0], arr.shape[1]))
                                arr = np.vstack([arr, padding])
                            else:
                                arr = arr[:target_length, :]
                        processed_arrays.append(arr)
                
                if processed_arrays:
                    combined = np.hstack(processed_arrays)
                    df = pd.DataFrame(combined, columns=[f'{descriptor_name}_{i+1}' for i in range(combined.shape[1])])
                    new_fps = pd.concat([new_fps, df], axis=1)

            else:
                # Handle single descriptor or list of values
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        descriptor = descriptor[:target_length]
                
                if descriptor.ndim > 1:
                    df = pd.DataFrame(descriptor, columns=[f'{descriptor_name}_{i+1}' for i in range(descriptor.shape[1])])
                    new_fps = pd.concat([new_fps, df], axis=1)
                else:
                    new_fps[descriptor_name] = descriptor

            new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
            return new_fps

        else:  # save_res == "np"
            new_fps = fps if isinstance(fps, np.ndarray) else np.array(fps)

            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                if descriptor.shape[0] != target_length:
                    if descriptor.shape[0] < target_length:
                        padding = np.zeros((target_length - descriptor.shape[0], descriptor.shape[1]))
                        descriptor = np.vstack([descriptor, padding])
                    else:
                        descriptor = descriptor[:target_length, :]
                new_fps = np.concatenate([new_fps, descriptor], axis=1)

            elif isinstance(descriptor, list) and len(descriptor) > 0 and isinstance(descriptor[0], np.ndarray):
                processed_arrays = []
                for arr in descriptor:
                    if arr.ndim == 1:
                        if len(arr) != target_length:
                            if len(arr) < target_length:
                                arr = np.pad(arr, (0, target_length - len(arr)), 'constant')
                            else:
                                arr = arr[:target_length]
                        processed_arrays.append(arr.reshape(-1, 1))
                    else:
                        if arr.shape[0] != target_length:
                            if arr.shape[0] < target_length:
                                padding = np.zeros((target_length - arr.shape[0], arr.shape[1]))
                                arr = np.vstack([arr, padding])
                            else:
                                arr = arr[:target_length, :]
                        processed_arrays.append(arr)
                
                if processed_arrays:
                    combined = np.hstack(processed_arrays)
                    new_fps = np.concatenate([new_fps, combined], axis=1)

            else:
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        descriptor = descriptor[:target_length]
                
                if descriptor.ndim == 1:
                    new_fps = np.concatenate([new_fps, descriptor.reshape(-1, 1)], axis=1)
                else:
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)

            new_fps = np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
            return new_fps

    except Exception as e:
        print(f"Error in generating_newfps: {e}")
        return fps

def Normalization(descriptor):
    """Normalize descriptor values using log transformation"""
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
    n_jobs = min(os.cpu_count(), 4)
    Executor = ThreadPoolExecutor if platform.system() == "Windows" else ProcessPoolExecutor
    with Executor(max_workers=n_jobs) as executor:
        futures = [executor.submit(values_chi, mol, chi_type) for mol in mols]
        descriptor = [future.result() for future in futures]
    
    if descriptor:
        max_length = max(len(x) for x in descriptor)
        padded_descriptor = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in descriptor])
        return padded_descriptor
    return np.array([[0] for _ in mols])

def sanitize_and_compute_descriptor(mol):
    """Sanitize molecule and compute BCUT2D descriptor"""
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            return [0] * 8
        
        try:
            return rdMolDescriptors.BCUT2D(mol)
        except Exception:
            return [0] * 8
    except Exception:
        return [0] * 8

def compute_descriptors_parallel(mols, n_jobs=None):
    """Compute descriptors in parallel"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count(), 4)
    Executor = ThreadPoolExecutor if platform.system() == "Windows" else ProcessPoolExecutor
    with Executor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols]
        descriptors = [future.result() for future in futures]
    return descriptors

# process_molecules_parallel fallback function defined above

def clear_descriptor_memory(descriptor):
    """Clear descriptor memory"""
    del descriptor
    gc.collect()

# ============================================================================
# Modular Processing Functions
# ============================================================================

def process_2d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """
    Process all 2D descriptors (28 descriptors) using selection list
    
    Descriptors 0-27:
    - Basic molecular properties
    - Counts and connectivity indices
    - Graph descriptors
    """
    
    # 0. MolWeight
    if selection[0] == 1:
        descriptor = [Descriptors.ExactMolWt(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolWeight', save_res)
        selected_descriptors.append("MolWeight")
        clear_descriptor_memory(descriptor)
    
    # 1. MolLogP
    if selection[1] == 1:
        descriptor = [Chem.Crippen.MolLogP(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolLogP', save_res)
        selected_descriptors.append("MolLogP")
        clear_descriptor_memory(descriptor)
    
    # 2. MolMR
    if selection[2] == 1:
        descriptor = [Chem.Crippen.MolMR(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MolMR', save_res)
        selected_descriptors.append("MolMR")
        clear_descriptor_memory(descriptor)
    
    # 3. TPSA
    if selection[3] == 1:
        descriptor = [Descriptors.TPSA(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'TPSA', save_res)
        selected_descriptors.append("TPSA")
        clear_descriptor_memory(descriptor)
    
    # 4. NumRotatableBonds
    if selection[4] == 1:
        descriptor = [Lipinski.NumRotatableBonds(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumRotatableBonds', save_res)
        selected_descriptors.append("NumRotatableBonds")
        clear_descriptor_memory(descriptor)
    
    # 5. HeavyAtomCount
    if selection[5] == 1:
        descriptor = [Lipinski.HeavyAtomCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'HeavyAtomCount', save_res)
        selected_descriptors.append("HeavyAtomCount")
        clear_descriptor_memory(descriptor)
    
    # 6. NumHAcceptors
    if selection[6] == 1:
        descriptor = [Lipinski.NumHAcceptors(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHAcceptors', save_res)
        selected_descriptors.append("NumHAcceptors")
        clear_descriptor_memory(descriptor)
    
    # 7. NumHDonors
    if selection[7] == 1:
        descriptor = [Lipinski.NumHDonors(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHDonors', save_res)
        selected_descriptors.append("NumHDonors")
        clear_descriptor_memory(descriptor)
    
    # 8. NumHeteroatoms
    if selection[8] == 1:
        descriptor = [Lipinski.NumHeteroatoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumHeteroatoms', save_res)
        selected_descriptors.append("NumHeteroatoms")
        clear_descriptor_memory(descriptor)
    
    # 9. NumValenceElectrons
    if selection[9] == 1:
        descriptor = [Descriptors.NumValenceElectrons(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumValenceElectrons', save_res)
        selected_descriptors.append("NumValenceElectrons")
        clear_descriptor_memory(descriptor)
    
    # 10. NHOHCount
    if selection[10] == 1:
        descriptor = [Lipinski.NHOHCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NHOHCount', save_res)
        selected_descriptors.append("NHOHCount")
        clear_descriptor_memory(descriptor)
    
    # 11. NOCount
    if selection[11] == 1:
        descriptor = [Lipinski.NOCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NOCount', save_res)
        selected_descriptors.append("NOCount")
        clear_descriptor_memory(descriptor)
    
    # 12. RingCount
    if selection[12] == 1:
        descriptor = [Lipinski.RingCount(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'RingCount', save_res)
        selected_descriptors.append("RingCount")
        clear_descriptor_memory(descriptor)
    
    # 13. NumAromaticRings
    if selection[13] == 1:
        descriptor = [Lipinski.NumAromaticRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAromaticRings', save_res)
        selected_descriptors.append("NumAromaticRings")
        clear_descriptor_memory(descriptor)
    
    # 14. NumSaturatedRings
    if selection[14] == 1:
        descriptor = [Lipinski.NumSaturatedRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumSaturatedRings', save_res)
        selected_descriptors.append("NumSaturatedRings")
        clear_descriptor_memory(descriptor)
    
    # 15. NumAliphaticRings
    if selection[15] == 1:
        descriptor = [Lipinski.NumAliphaticRings(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAliphaticRings', save_res)
        selected_descriptors.append("NumAliphaticRings")
        clear_descriptor_memory(descriptor)
    
    # 16. LabuteASA
    if selection[16] == 1:
        descriptor = [rdMolDescriptors.CalcLabuteASA(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'LabuteASA', save_res)
        selected_descriptors.append("LabuteASA")
        clear_descriptor_memory(descriptor)
    
    # 17. BalabanJ
    if selection[17] == 1:
        descriptor = [Chem.GraphDescriptors.BalabanJ(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'BalabanJ', save_res)
        selected_descriptors.append("BalabanJ")
        clear_descriptor_memory(descriptor)
    
    # 18. BertzCT
    if selection[18] == 1:
        descriptor = [Chem.GraphDescriptors.BertzCT(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'BertzCT', save_res)
        selected_descriptors.append("BertzCT")
        clear_descriptor_memory(descriptor)
    
    # 19. Ipc
    if selection[19] == 1:
        descriptor = [Chem.GraphDescriptors.Ipc(mol) for mol in mols]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'Ipc', save_res)
        selected_descriptors.append("Ipc")
        clear_descriptor_memory(descriptor)
    
    # 20. kappa_Series[1-3]_ind
    if selection[20] == 1:
        d1 = [Chem.GraphDescriptors.Kappa1(mol) for mol in mols]
        d2 = [Chem.GraphDescriptors.Kappa2(mol) for mol in mols]
        d3 = [Chem.GraphDescriptors.Kappa3(mol) for mol in mols]
        fps = generating_newfps(fps, [np.asarray(d1), np.asarray(d2), np.asarray(d3)], 
                               'kappa_Series[1-3]_ind', save_res)
        selected_descriptors.append("kappa_Series[1-3]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    
    # 21. Chi_Series[13]_ind
    if selection[21] == 1:
        descriptor1 = generate_chi(mols, 'n')
        descriptor2 = generate_chi(mols, 'v')
        fps = generating_newfps(fps, descriptor1, 'Chi_Series_n', save_res)
        fps = generating_newfps(fps, descriptor2, 'Chi_Series_v', save_res)
        selected_descriptors.append("Chi_Series[13]_ind")
        clear_descriptor_memory(descriptor1)
        clear_descriptor_memory(descriptor2)
    
    # 22. Phi
    if selection[22] == 1:
        descriptor = [rdMolDescriptors.CalcPhi(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'Phi', save_res)
        selected_descriptors.append("Phi")
        clear_descriptor_memory(descriptor)
    
    # 23. HallKierAlpha
    if selection[23] == 1:
        descriptor = [Chem.GraphDescriptors.HallKierAlpha(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'HallKierAlpha', save_res)
        selected_descriptors.append("HallKierAlpha")
        clear_descriptor_memory(descriptor)
    
    # 24. NumAmideBonds
    if selection[24] == 1:
        descriptor = [rdMolDescriptors.CalcNumAmideBonds(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumAmideBonds', save_res)
        selected_descriptors.append("NumAmideBonds")
        clear_descriptor_memory(descriptor)
    
    # 25. FractionCSP3
    if selection[25] == 1:
        descriptor = [Lipinski.FractionCSP3(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'FractionCSP3', save_res)
        selected_descriptors.append("FractionCSP3")
        clear_descriptor_memory(descriptor)
    
    # 26. NumSpiroAtoms
    if selection[26] == 1:
        descriptor = [rdMolDescriptors.CalcNumSpiroAtoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumSpiroAtoms', save_res)
        selected_descriptors.append("NumSpiroAtoms")
        clear_descriptor_memory(descriptor)
    
    # 27. NumBridgeheadAtoms
    if selection[27] == 1:
        descriptor = [rdMolDescriptors.CalcNumBridgeheadAtoms(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'NumBridgeheadAtoms', save_res)
        selected_descriptors.append("NumBridgeheadAtoms")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors

def process_vsa_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """
    Process VSA descriptors (5 series) using selection list
    
    Descriptors 28-32:
    - PEOE_VSA_Series[1-14]_ind
    - SMR_VSA_Series[1-10]_ind
    - SlogP_VSA_Series[1-12]_ind
    - EState_VSA_Series[1-11]_ind
    - VSA_EState_Series[1-10]_ind
    """
    
    # 28. PEOE_VSA_Series[1-14]_ind
    if selection[28] == 1:
        descriptors = []
        for i in range(1, 15):
            func = getattr(Chem.MolSurf, f'PEOE_VSA{i}')
            descriptors.append(np.asarray([func(mol) for mol in mols]))
        fps = generating_newfps(fps, descriptors, 'PEOE_VSA_Series[1-14]_ind', save_res)
        selected_descriptors.append("PEOE_VSA_Series[1-14]_ind")
        for d in descriptors:
            clear_descriptor_memory(d)
    
    # 29. SMR_VSA_Series[1-10]_ind
    if selection[29] == 1:
        descriptors = []
        for i in range(1, 11):
            func = getattr(Chem.MolSurf, f'SMR_VSA{i}')
            descriptors.append(np.asarray([func(mol) for mol in mols]))
        fps = generating_newfps(fps, descriptors, 'SMR_VSA_Series[1-10]_ind', save_res)
        selected_descriptors.append("SMR_VSA_Series[1-10]_ind")
        for d in descriptors:
            clear_descriptor_memory(d)
    
    # 30. SlogP_VSA_Series[1-12]_ind
    if selection[30] == 1:
        descriptors = []
        for i in range(1, 13):
            func = getattr(Chem.MolSurf, f'SlogP_VSA{i}')
            descriptors.append(np.asarray([func(mol) for mol in mols]))
        fps = generating_newfps(fps, descriptors, 'SlogP_VSA_Series[1-12]_ind', save_res)
        selected_descriptors.append("SlogP_VSA_Series[1-12]_ind")
        for d in descriptors:
            clear_descriptor_memory(d)
    
    # 31. EState_VSA_Series[1-11]_ind
    if selection[31] == 1:
        descriptors = []
        for i in range(1, 12):
            func = getattr(Chem.EState.EState_VSA, f'EState_VSA{i}')
            descriptors.append(np.asarray([func(mol) for mol in mols]))
        fps = generating_newfps(fps, descriptors, 'EState_VSA_Series[1-11]_ind', save_res)
        selected_descriptors.append("EState_VSA_Series[1-11]_ind")
        for d in descriptors:
            clear_descriptor_memory(d)
    
    # 32. VSA_EState_Series[1-10]_ind
    if selection[32] == 1:
        descriptors = []
        for i in range(1, 11):
            func = getattr(Chem.EState.EState_VSA, f'VSA_EState{i}')
            descriptors.append(np.asarray([func(mol) for mol in mols]))
        fps = generating_newfps(fps, descriptors, 'VSA_EState_Series[1-10]_ind', save_res)
        selected_descriptors.append("VSA_EState_Series[1-10]_ind")
        for d in descriptors:
            clear_descriptor_memory(d)
    
    return fps, selected_descriptors

def process_rdmol_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """
    Process rdMolDescriptors (3 descriptors) using selection list
    
    Descriptors 33-35:
    - MQNs
    - AUTOCORR2D
    - BCUT2D
    """
    
    # 33. MQNs
    if selection[33] == 1:
        descriptor = [rdMolDescriptors.MQNs_(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'MQNs', save_res)
        selected_descriptors.append("MQNs")
        clear_descriptor_memory(descriptor)
    
    # 34. AUTOCORR2D
    if selection[34] == 1:
        descriptor = [rdMolDescriptors.CalcAUTOCORR2D(mol) for mol in mols]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR2D', save_res)
        selected_descriptors.append("AUTOCORR2D")
        clear_descriptor_memory(descriptor)
    
    # 35. BCUT2D
    if selection[35] == 1:
        descriptor = compute_descriptors_parallel(mols)
        fps = generating_newfps(fps, descriptor, 'BCUT2D', save_res)
        selected_descriptors.append("BCUT2D")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors

def process_3d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np", mols_3d=None):
    """
    Process 3D descriptors (13 descriptors) using selection list
    
    Descriptors 36-48:
    - Shape descriptors
    - PMI and NPR series
    - Advanced 3D descriptors (AUTOCORR3D, RDF, MORSE, WHIM, GETAWAY)
    """
    
    # Generate 3D conformers if needed and not provided
    need_3d = any(selection[i] == 1 for i in range(36, 49))
    if need_3d and mols_3d is None:
        print("  Generating 3D conformers for 3D descriptor calculation...")
        mols_3d = process_molecules_parallel(mols, max_workers=4)
        # Ensure same length
        while len(mols_3d) < fps.shape[0]:
            mols_3d.append(None)
        mols_3d = mols_3d[:fps.shape[0]]
    
    if not need_3d:
        return fps, selected_descriptors
    
    def safe_3d_descriptor(func, mol):
        """Safely compute 3D descriptor"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                val = func(mol)
                if val is not None and not np.isnan(val) and not np.isinf(val):
                    return val
        except:
            pass
        return 0.0
    
    # 36. Asphericity
    if selection[36] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcAsphericity, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'Asphericity', save_res)
        selected_descriptors.append("Asphericity")
        clear_descriptor_memory(descriptor)
    
    # 37. PBF
    if selection[37] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcPBF, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'PBF', save_res)
        selected_descriptors.append("PBF")
        clear_descriptor_memory(descriptor)
    
    # 38. RadiusOfGyration
    if selection[38] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcRadiusOfGyration, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'RadiusOfGyration', save_res)
        selected_descriptors.append("RadiusOfGyration")
        clear_descriptor_memory(descriptor)
    
    # 39. InertialShapeFactor
    if selection[39] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcInertialShapeFactor, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'InertialShapeFactor', save_res)
        selected_descriptors.append("InertialShapeFactor")
        clear_descriptor_memory(descriptor)
    
    # 40. Eccentricity
    if selection[40] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcEccentricity, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'Eccentricity', save_res)
        selected_descriptors.append("Eccentricity")
        clear_descriptor_memory(descriptor)
    
    # 41. SpherocityIndex
    if selection[41] == 1:
        descriptor = [safe_3d_descriptor(rdMolDescriptors.CalcSpherocityIndex, mol) for mol in mols_3d]
        fps = generating_newfps(fps, descriptor, 'SpherocityIndex', save_res)
        selected_descriptors.append("SpherocityIndex")
        clear_descriptor_memory(descriptor)
    
    # 42. PMI_series[1-3]_ind
    if selection[42] == 1:
        d1 = [safe_3d_descriptor(rdMolDescriptors.CalcPMI1, mol) for mol in mols_3d]
        d2 = [safe_3d_descriptor(rdMolDescriptors.CalcPMI2, mol) for mol in mols_3d]
        d3 = [safe_3d_descriptor(rdMolDescriptors.CalcPMI3, mol) for mol in mols_3d]
        d1 = Normalization(d1)
        d2 = Normalization(d2)
        d3 = Normalization(d3)
        fps = generating_newfps(fps, [np.asarray(d1), np.asarray(d2), np.asarray(d3)], 
                               'PMI_series[1-3]_ind', save_res)
        selected_descriptors.append("PMI_series[1-3]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
    
    # 43. NPR_series[1-2]_ind
    if selection[43] == 1:
        d1 = [safe_3d_descriptor(rdMolDescriptors.CalcNPR1, mol) for mol in mols_3d]
        d2 = [safe_3d_descriptor(rdMolDescriptors.CalcNPR2, mol) for mol in mols_3d]
        fps = generating_newfps(fps, [np.asarray(d1), np.asarray(d2)], 
                               'NPR_series[1-2]_ind', save_res)
        selected_descriptors.append("NPR_series[1-2]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
    
    # 44. AUTOCORR3D
    if selection[44] == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcAUTOCORR3D(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 80)
            except:
                descriptor.append([0] * 80)
        fps = generating_newfps(fps, descriptor, 'AUTOCORR3D', save_res)
        selected_descriptors.append("AUTOCORR3D")
        clear_descriptor_memory(descriptor)
    
    # 45. RDF
    if selection[45] == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcRDF(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 210)
            except:
                descriptor.append([0] * 210)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'RDF', save_res)
        selected_descriptors.append("RDF")
        clear_descriptor_memory(descriptor)
    
    # 46. MORSE
    if selection[46] == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcMORSE(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 224)
            except:
                descriptor.append([0] * 224)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'MORSE', save_res)
        selected_descriptors.append("MORSE")
        clear_descriptor_memory(descriptor)
    
    # 47. WHIM
    if selection[47] == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcWHIM(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 114)
            except:
                descriptor.append([0] * 114)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'WHIM', save_res)
        selected_descriptors.append("WHIM")
        clear_descriptor_memory(descriptor)
    
    # 48. GETAWAY
    if selection[48] == 1:
        descriptor = []
        for mol in mols_3d:
            try:
                if mol is not None and mol.GetNumConformers() > 0:
                    val = rdMolDescriptors.CalcGETAWAY(mol)
                    descriptor.append(val)
                else:
                    descriptor.append([0] * 273)
            except:
                descriptor.append([0] * 273)
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'GETAWAY', save_res)
        selected_descriptors.append("GETAWAY")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors

# ============================================================================
# Main Selection Function
# ============================================================================

def selection_data_descriptor_compress(selection, fps, mols, name, target_path="result", save_res="np", mols_3d=None):
    """
    Main function to process descriptor selection using modular approach.

    This function combines all descriptor categories using the selection array
    and returns the updated fingerprints with selected descriptors.

    Parameters:
    -----------
    selection : np.ndarray
        Binary array of length 49 indicating which descriptors to use
    fps : np.ndarray or pd.DataFrame
        Initial fingerprints
    mols : list
        List of RDKit mol objects
    name : str
        Name for saving results
    target_path : str
        Path to save results
    save_res : str
        Output format ('np' or 'pd')
    mols_3d : list, optional
        Pre-generated 3D conformers
    unfixed : bool
        Compatibility parameter (not used but kept for backward compatibility)

    Returns:
    --------
    tuple
        (fps_updated, selected_descriptors)
    """
    selected_descriptors = []
    excluded_descriptors = []

    # Try to load cached descriptors first
    cached_result = load_cached_descriptors_with_selection(selection, name, target_path, len(mols))
    if cached_result is not None:
        fps_cached, selected_descriptors_cached = cached_result
        print(f"  ✅ Loaded cached descriptors for {name}: {fps_cached.shape}")
        print(f"  📊 Cached descriptors: {selected_descriptors_cached}")

        # Combine with existing fingerprints
        if isinstance(fps, pd.DataFrame):
            fps = fps.values
        if isinstance(fps_cached, pd.DataFrame):
            fps_cached = fps_cached.values

        fps_combined = np.hstack([fps, fps_cached])
        return fps_combined.astype('float32'), selected_descriptors_cached

    # If no exact cache match, inform about calculation
    print(f"  📊 Applying feature selection for {name}...")

    # Process 2D descriptors (0-27)
    fps, selected_descriptors = process_2d_descriptors_selection(
        selection, fps, mols, selected_descriptors, save_res
    )

    # Process VSA descriptors (28-32)
    fps, selected_descriptors = process_vsa_descriptors_selection(
        selection, fps, mols, selected_descriptors, save_res
    )

    # Process rdMol descriptors (33-35)
    fps, selected_descriptors = process_rdmol_descriptors_selection(
        selection, fps, mols, selected_descriptors, save_res
    )

    # Process 3D descriptors (36-48)
    fps, selected_descriptors = process_3d_descriptors_selection(
        selection, fps, mols, selected_descriptors, save_res, mols_3d
    )

    # Save descriptors for future use
    save_descriptors_for_caching(fps, selected_descriptors, selection, name, target_path, len(mols))

    # Save results if requested
    if save_res == "pd" and isinstance(fps, pd.DataFrame):
        fps.to_csv(f'{target_path}/{name}_feature_selection.csv', index=False)

    # Ensure final data type
    if isinstance(fps, np.ndarray):
        fps = fps.astype('float32')

    return fps, selected_descriptors

# ============================================================================
# Descriptor Caching Functions
# ============================================================================

def get_cache_filename(name, target_path, n_mols, selection):
    """Generate cache filename based on dataset parameters.

    Uses unified cache format: result/chemical_descriptors/{dataset}/{split_type}/{dataset}_{split_type}_selected_descriptors.npz
    No hash - single cache per dataset/split combination
    """
    # Parse dataset and split_type from name (format: "dataset-split_type")
    if '-' in name:
        dataset, split_type = name.split('-', 1)
        # Use the unified cache directory structure
        cache_dir = Path('result/chemical_descriptors') / dataset / split_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Simple filename without hash
        return cache_dir / f"{dataset}_{split_type}_selected_descriptors.npz"
    else:
        # Fallback for old format
        cache_dir = Path(target_path) / "descriptor_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{name}_n{n_mols}_selected_descriptors.npz"

def load_cached_descriptors_with_selection(selection, name, target_path, n_mols):
    """
    Load cached descriptors that match the given selection.

    First tries to load from unified cache, then from ChemDescriptorCalculator format.

    Returns:
        tuple: (descriptors_array, selected_descriptors_list) or None if not found
    """
    # Try unified cache format first
    cache_file = get_cache_filename(name, target_path, n_mols, selection)

    # Don't check for ChemDescriptorCalculator cache here - let it be handled elsewhere
    # The cached_descriptor_calculation function will handle loading from existing cache

    if cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=True)
            descriptors = data['descriptors']
            selected_descriptors = data['selected_descriptors'].tolist()
            selection_cached = data['selection']

            # Verify selection matches
            if np.array_equal(selection, selection_cached):
                return descriptors, selected_descriptors
            else:
                print(f"  ⚠️  Selection mismatch in cache file {cache_file}")
                return None
        except Exception as e:
            print(f"  ⚠️  Error loading cache {cache_file}: {e}")
            return None

    return None

def save_descriptors_for_caching(fps, selected_descriptors, selection, name, target_path, n_mols):
    """
    Save calculated descriptors to cache for future use.

    Uses unified cache format: result/chemical_descriptors/{dataset}/{split_type}/
    """
    cache_file = get_cache_filename(name, target_path, n_mols, selection)

    try:
        # Extract only the descriptor part (excluding original fingerprints)
        if isinstance(fps, np.ndarray) and fps.shape[1] > 2727:  # 2727 = original fingerprint size
            descriptors_only = fps[:, 2727:]  # Extract descriptors part
        else:
            descriptors_only = fps

        np.savez_compressed(
            cache_file,
            descriptors=descriptors_only,
            selected_descriptors=np.array(selected_descriptors, dtype=object),
            selection=selection
        )
        print(f"  💾 Saved descriptors to cache: {cache_file}")
    except Exception as e:
        print(f"  ⚠️  Error saving cache {cache_file}: {e}")

# ============================================================================
# Optuna Study Functions
# ============================================================================

def selection_fromStudy_compress(study_name, storage, unfixed=False, showlog=True):
    """
    Load best feature selection from Optuna study.
    
    Parameters:
    -----------
    study_name : str
        Name of the Optuna study
    storage : str
        Storage location (e.g., sqlite database)
    unfixed : bool
        If True, all descriptors are selectable
    showlog : bool
        If True, print debug information
    
    Returns:
    --------
    np.ndarray
        Binary array of length 49 indicating selected descriptors
    """
    model_fea = np.zeros(49, dtype=int)
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    best_trial = study.best_trial
    
    # Use descriptor names from config
    descriptor_names = CHEMICAL_DESCRIPTORS
    
    if not unfixed:
        # Fix first 4 descriptors
        model_fea[0:4] = 1
        required_features = descriptor_names[0:4]
    else:
        required_features = []
    
    # Map parameter names to indices
    param_to_index = {name: i for i, name in enumerate(descriptor_names)}
    
    # Process best trial parameters
    for param_name, param_value in best_trial.params.items():
        if param_name in param_to_index:
            idx = param_to_index[param_name]
            if unfixed or idx >= 4:  # Only set if unfixed or not in fixed range
                model_fea[idx] = param_value
    
    if showlog:
        print(f"Best trial for study '{study_name}':")
        print(f"  Best value: {best_trial.value}")
        print(f"  Selected features: {np.sum(model_fea)} out of 49")
        if not unfixed:
            print(f"  Fixed features: {required_features}")
    
    return model_fea

def convert_params_to_selection(params):
    """
    Convert Optuna trial parameters to selection array.
    
    Parameters:
    -----------
    params : dict
        Optuna trial parameters
    unfixed : bool
        If True, all descriptors are selectable
    
    Returns:
    --------
    np.ndarray
        Binary array of length 49
    """
    selection = np.zeros(49, dtype=int)
    
    # No fixed descriptors - use only what was selected
    
    # Use descriptor names from config
    descriptor_names = CHEMICAL_DESCRIPTORS
    
    for i, name in enumerate(descriptor_names):
        if name in params:
            selection[i] = params[name]
    
    return selection

def selection_structure_compress(study_name, storage, input_dim, returnOnly=False):
    """
    Load best model structure from Optuna study.
    
    Parameters:
    -----------
    study_name : str
        Name of the Optuna study
    storage : str
        Storage location
    input_dim : int
        Input dimension for the model
    returnOnly : bool
        If True, only return learning rate
    
    Returns:
    --------
    dict or float
        Model parameters or just learning rate
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    
    lr = best_trial.params.get("learning_rate", best_trial.params.get("lr", 0.001))
    
    if returnOnly:
        return lr
    
    # Extract model architecture parameters
    model_params = {
        'learning_rate': lr,
        'n_layers': best_trial.params.get('n_layers', 3),
        'hidden_dims': [],
        'dropout_rate': best_trial.params.get('dropout_rate', 0.2),
        'batch_size': best_trial.params.get('batch_size', 32),
        'activation': best_trial.params.get('activation', 'relu'),
        'optimizer': best_trial.params.get('optimizer', 'adam'),
        'weight_decay': best_trial.params.get('weight_decay', 1e-5)
    }
    
    # Get hidden dimensions
    for i in range(model_params['n_layers']):
        hidden_dim = best_trial.params.get(f'hidden_dim_{i}', 256)
        model_params['hidden_dims'].append(hidden_dim)
    
    # Build full architecture
    model_params['architecture'] = [input_dim] + model_params['hidden_dims'] + [1]
    
    print(f"Model architecture: {model_params['architecture']}")
    print(f"Learning rate: {model_params['learning_rate']}")
    print(f"Dropout rate: {model_params['dropout_rate']}")
    
    return model_params

def create_model_with_best_architecture(input_dim: int, 
                                       study_name: str = None,
                                       storage: str = None,
                                       model_params: dict = None,
                                       device: str = 'cpu'):
    """
    Create a model using the best architecture from ano_model_optimization.py
    
    Parameters:
    -----------
    input_dim : int
        Input dimension for the model
    study_name : str, optional
        Name of the Optuna study to load best params from
    storage : str, optional
        Storage location for the Optuna study
    model_params : dict, optional
        Pre-loaded model parameters (if None, will load from study)
    device : str
        Device to place the model on ('cpu' or 'cuda')
    
    Returns:
    --------
    model : nn.Module or None
        The created model, or None if model optimization module not available
    """
    if not MODEL_OPT_AVAILABLE:
        print("Warning: ano_model_optimization module not available. Using default architecture.")
        if PYTORCH_AVAILABLE:
            # Fallback to simple architecture if model optimization not available
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
            return model.to(device)
        else:
            print("PyTorch not available. Cannot create model.")
            return None
    
    # Get model parameters if not provided
    if model_params is None:
        if study_name is not None and storage is not None:
            model_params = get_best_model_params(study_name, storage, input_dim)
        else:
            # Use default optimized architecture
            model_params = {
                'hidden_dims': [256, 128, 64],
                'activation': 'relu',
                'dropout_rate': 0.2,
                'use_batch_norm': True
            }
    
    # Create model using the optimization module
    try:
        # Use OptimizedDNN from ano_model_optimization
        model = OptimizedDNN(
            input_dim=input_dim,
            hidden_dims=model_params.get('hidden_dims', [256, 128, 64]),
            activation=model_params.get('activation', 'relu'),
            dropout_rate=model_params.get('dropout_rate', 0.2),
            use_batch_norm=model_params.get('use_batch_norm', True)
        )
        
        print(f"Created OptimizedDNN model with architecture: {[input_dim] + model_params.get('hidden_dims', [256, 128, 64]) + [1]}")
        return model.to(device)
        
    except Exception as e:
        print(f"Error creating OptimizedDNN: {e}")
        # Fallback to create_model_from_params if available
        if hasattr(create_model_from_params, '__call__'):
            try:
                model = create_model_from_params(input_dim, model_params)
                return model.to(device)
            except Exception as e2:
                print(f"Error with create_model_from_params: {e2}")
        
        # Final fallback
        if PYTORCH_AVAILABLE:
            print("Using fallback architecture")
            hidden_dims = model_params.get('hidden_dims', [256, 128, 64])
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(model_params.get('dropout_rate', 0.2)))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            model = nn.Sequential(*layers)
            return model.to(device)

        return None

# ============================================================================
# SimpleDNN Class Definition
# ============================================================================

class SimpleDNN(nn.Module):
    """Simple Deep Neural Network for molecular property prediction with strong regularization"""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.2,
                 use_batch_norm=True, l2_reg=1e-4, activation='relu'):
        super(SimpleDNN, self).__init__()

        # Adaptive regularization based on input dimension
        if input_dim > 2000:
            # Moderate regularization for high-dimensional inputs
            self.dropout_rate = max(0.3, dropout_rate)  # At least 0.3 (reduced from 0.5)
            self.use_multiple_dropouts = False  # Disable progressive dropout for now
        else:
            self.dropout_rate = dropout_rate
            self.use_multiple_dropouts = False

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()  # Multiple dropout layers

        # Store for potential L2 regularization computation
        self.l2_reg = l2_reg

        # Build layers
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Progressive dropout - stronger for earlier layers
            if self.use_multiple_dropouts:
                # Gradually decrease dropout from input to output
                layer_dropout = self.dropout_rate * (1 - i * 0.1 / len(hidden_dims))
                layer_dropout = max(0.2, layer_dropout)  # Minimum 0.2
                self.dropouts.append(nn.Dropout(layer_dropout))
            else:
                self.dropouts.append(nn.Dropout(self.dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

        # Backward compatibility - create single dropout attribute
        self.dropout = self.dropouts[0] if self.dropouts else nn.Dropout(self.dropout_rate)

        # Initialize weights normally
        self.apply(self._init_weights)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

    def _init_weights(self, m):
        """Standard weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_weights_constrained(self, m):
        """Constrained weight initialization for high-dimensional inputs"""
        if isinstance(m, nn.Linear):
            # Smaller initial weights for high-dim inputs
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                # BatchNorm safety: skip if batch size is 1 during training
                if self.training and x.size(0) == 1:
                    pass  # Skip BatchNorm for single sample batches
                else:
                    x = self.batch_norms[i](x)
            x = self.activation(x)
            # Use layer-specific dropout
            x = self.dropouts[i](x)

        x = self.output_layer(x)
        return x

    def get_l2_loss(self):
        """Compute L2 regularization loss"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss

# Compatibility alias
class FlexibleDNNModel(nn.Module):
    """Flexible Deep Neural Network that accepts n_layers and hidden_dims parameters"""

    def __init__(self, input_dim, n_layers, hidden_dims, activation='relu', dropout_rate=0.2, use_batch_norm=False,
                 weight_decay=0.0, optimizer_name='Adam', lr=0.001, batch_size=32, scheduler=None, **kwargs):
        super(FlexibleDNNModel, self).__init__()

        # Ensure hidden_dims is valid
        if not hidden_dims or len(hidden_dims) < n_layers + 1:
            # Auto-generate hidden dims if not provided correctly
            hidden_dims = []
            current_dim = input_dim
            for i in range(n_layers):
                current_dim = max(32, current_dim // 2)
                hidden_dims.append(current_dim)
            hidden_dims.append(1)  # Output layer

        # Remove just the hidden layer dims (not output layer) if too many
        if len(hidden_dims) > n_layers + 1:
            hidden_dims = hidden_dims[:n_layers] + [1]

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)

        # Select activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'elu':
            self.activation = nn.ELU()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # Build layers
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims[:-1]):  # All except output
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, hidden_dims[-1])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.output_layer(x)
        x = x.squeeze(-1)
        return x