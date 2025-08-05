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
1. 2D Molecular Descriptors: Basic molecular properties
2. VSA (Van der Waals Surface Area) Descriptors: Surface-based properties
3. 3D Molecular Descriptors: Conformation-dependent properties
4. Feature Selection Functions: Methods to select optimal descriptor combinations

DESCRIPTOR CATEGORIES (51 total):
- Basic properties: MolWt, LogP, TPSA, etc.
- Counts: HBD, HBA, rotatable bonds, rings, etc.
- Connectivity: Balaban J, Kappa indices, etc.
- VSA descriptors: PEOE_VSA, SMR_VSA, SlogP_VSA, etc.
- 3D descriptors: RDF, MORSE, WHIM, GETAWAY, etc.
- Autocorrelation: 2D and 3D autocorrelation functions

USAGE:
This module is used by ANO modules 5, 7, and 8 for feature selection optimization.
It provides functions to calculate descriptors and combine them with fingerprints.
"""

# ============================================================================
# PART 1: FEATURE_SELECTION_SELECTION_PART1_IMPORTS
# ============================================================================

"""
Feature Selection Part 1: Imports and Basic Functions
====================================================

This section contains all necessary imports and basic utility functions
for molecular descriptor calculation and feature selection.
"""

import os
import numpy as np
import pandas as pd
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import Union, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdDistGeom, rdPartialCharges
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Descriptors import ExactMolWt

# PyTorch imports (replacing TensorFlow for better performance)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna

def mol3d(mol):
    """
    Generate 3D conformers for a molecule.
    
    This function attempts multiple optimization methods to generate a stable
    3D conformation for molecular descriptor calculation. It tries:
    1. ETKDG v3 embedding (best quality)
    2. UFF optimization (universal force field)
    3. MMFF optimization (Merck molecular force field)
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        Input molecule
    
    Returns:
    --------
    mol : rdkit.Chem.Mol or None
        Molecule with 3D conformer or None if generation fails
    """
    mol = Chem.AddHs(mol)
    optimization_methods = [
        (AllChem.EmbedMolecule, (mol, AllChem.ETKDGv3()), {}),
        (AllChem.UFFOptimizeMolecule, (mol,), {'maxIters': 200}),
        (AllChem.MMFFOptimizeMolecule, (mol,), {'maxIters': 200})
    ]

    for method, args, kwargs in optimization_methods:
        try:
            method(*args, **kwargs)
            if mol.GetNumConformers() > 0:
                return mol
        except ValueError as e:
            print(f"Error: {e} - Trying next optimization method [{method}]")

    print(f"Invalid mol for 3d {Chem.MolToSmiles(mol)} - No conformer generated")
    return None

def process_chunk_optimized(chunk_data):
    """Process data chunks for parallel processing"""
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

        if save_res == "pd":
            new_fps = pd.DataFrame(fps) if not isinstance(fps, pd.DataFrame) else fps

            if isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                descriptors_df = generate_df_concurrently(descriptor, descriptor_name, chunk_size)
                if descriptors_df is not None:
                    new_fps = pd.concat([new_fps, descriptors_df], axis=1)

            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
                    # Handle dimension mismatch by ensuring all arrays have the same length
                    target_length = len(fps)
                    padded_arrays = []
                    for arr in descriptor:
                        if len(arr) != target_length:
                            # Pad or truncate to match target length
                            if len(arr) < target_length:
                                # Pad with zeros
                                padded = np.zeros(target_length)
                                padded[:len(arr)] = arr
                                padded_arrays.append(padded.reshape(-1, 1))
                            else:
                                # Truncate
                                padded_arrays.append(arr[:target_length].reshape(-1, 1))
                        else:
                            padded_arrays.append(arr.reshape(-1, 1))
                    
                    combined = np.hstack(padded_arrays)
                    new_fps = np.concatenate([new_fps, combined], axis=1)
                except Exception as e:
                    print(f"[-2-] Error occurred: {e}")

            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
                    descriptors_df = generate_df_concurrently(descriptor, descriptor_name, chunk_size)
                    if descriptors_df is not None:
                        new_fps = pd.concat([new_fps, descriptors_df], axis=1)
                except Exception as e:
                    print(f"[-3-] Error occurred: {e}")

            else:
                descriptor = np.asarray(descriptor).astype('float')
                new_fps[descriptor_name] = descriptor.flatten()

            new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
            return new_fps

        else:
            new_fps = fps

            if descriptor is None:
                pass
            elif isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                try:
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)
                except Exception as e:
                    print(f"[-1-] Error occurred: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
                    combined = np.hstack([
                        arr if arr.ndim > 1 else arr.reshape(-1, 1)
                        for arr in descriptor
                    ])
                    new_fps = np.concatenate([new_fps, combined], axis=1)
                except Exception as e:
                    print(f"[-2-] Error occurred: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
                    new_fps = np.concatenate([new_fps, descriptor], axis=1)
                except Exception as e:
                    print(f"[-3-] Error occurred: {e}")
            else:
                descriptor = np.asarray(descriptor).astype('float')
                new_fps = np.concatenate([new_fps, descriptor[:, None]], axis=1)

            new_fps = np.nan_to_num(new_fps, nan=0.0, posinf=0.0, neginf=0.0).astype('float')
            return new_fps

    except Exception as e:
        print(f"Error occurred in {descriptor_name}: {e}")
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
    n_jobs = os.cpu_count()
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
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
            return [Descriptors.MolWt(mol)] * 8
    except Exception as e:
        return [0] * 8

def compute_descriptors_parallel(mols, n_jobs=None):
    """Compute BCUT2D descriptors in parallel"""
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols if mol is not None]
        descriptors = [future.result() for future in futures]

    max_length = max(len(d) for d in descriptors)
    padded_descriptors = np.array([np.pad(d, (0, max_length - len(d)), 'constant') for d in descriptors])

    return padded_descriptors

def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    """Process molecules in parallel for 3D conformer generation"""
    results = [None] * len(mols)

    for i in range(0, len(mols), chunk_size):
        chunk = mols[i:i + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(mol3d, mol): idx for idx, mol in enumerate(chunk)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    results[i + idx] = result

        gc.collect()

    return [mol for mol in results if mol is not None] 

# ============================================================================
# PART 2: FEATURE_SELECTION_SELECTION_PART2_2D_DESCRIPTORS
# ============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Selection Part 2: 2D Descriptors Processing
기능 선택 선택 2부: 2D 서술자 처리
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski

# Import functions from part 1
def process_2d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """Process all 2D descriptors (28 descriptors) using selection list"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        import gc
        gc.collect()
    
    # 1. MolWt - Molecular Weight
    if selection[0] == 1:
        descriptor = [Descriptors.ExactMolWt(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolWt', save_res)
        selected_descriptors.append("MolWt")
        clear_descriptor_memory(descriptor)
        
    # 2. MolLogP - Molecular LogP
    if selection[1] == 1:
        descriptor = [Chem.Crippen.MolLogP(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolLogP', save_res)
        selected_descriptors.append("MolLogP")
        clear_descriptor_memory(descriptor)
        
    # 3. MolMR - Molecular Refractivity
    if selection[2] == 1:
        descriptor = [Chem.Crippen.MolMR(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolMR', save_res)
        selected_descriptors.append("MolMR")
        clear_descriptor_memory(descriptor)
        
    # 4. TPSA - Topological Polar Surface Area
    if selection[3] == 1:
        descriptor = [Descriptors.TPSA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'TPSA', save_res)
        selected_descriptors.append("TPSA")
        clear_descriptor_memory(descriptor)
        
    # 5. NumRotatableBonds - Number of Rotatable Bonds
    if selection[4] == 1:
        descriptor = [Chem.Lipinski.NumRotatableBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumRotatableBonds', save_res)
        selected_descriptors.append("NumRotatableBonds")
        clear_descriptor_memory(descriptor)
        
    # 6. HeavyAtomCount - Heavy Atom Count
    if selection[5] == 1:
        descriptor = [Descriptors.HeavyAtomCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HeavyAtomCount', save_res)
        selected_descriptors.append("HeavyAtomCount")
        clear_descriptor_memory(descriptor)
        
    # 7. NumHAcceptors - Number of H Acceptors
    if selection[6] == 1:
        descriptor = [Chem.Lipinski.NumHAcceptors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHAcceptors', save_res)
        selected_descriptors.append("NumHAcceptors")
        clear_descriptor_memory(descriptor)
        
    # 8. NumHDonors - Number of H Donors
    if selection[7] == 1:
        descriptor = [Chem.Lipinski.NumHDonors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHDonors', save_res)
        selected_descriptors.append("NumHDonors")
        clear_descriptor_memory(descriptor)
        
    # 9. NumHeteroatoms - Number of Heteroatoms
    if selection[8] == 1:
        descriptor = [Descriptors.NumHeteroatoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHeteroatoms', save_res)
        selected_descriptors.append("NumHeteroatoms")
        clear_descriptor_memory(descriptor)
        
    # 10. NumValenceElectrons - Number of Valence Electrons
    if selection[9] == 1:
        descriptor = [Descriptors.NumValenceElectrons(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumValenceElectrons', save_res)
        selected_descriptors.append("NumValenceElectrons")
        clear_descriptor_memory(descriptor)
        
    # 11. NHOHCount - NHOH Count
    if selection[10] == 1:
        descriptor = [Chem.Lipinski.NHOHCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NHOHCount', save_res)
        selected_descriptors.append("NHOHCount")
        clear_descriptor_memory(descriptor)
        
    # 12. NOCount - NO Count
    if selection[11] == 1:
        descriptor = [Chem.Lipinski.NOCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NOCount', save_res)
        selected_descriptors.append("NOCount")
        clear_descriptor_memory(descriptor)
        
    # 13. RingCount - Ring Count
    if selection[12] == 1:
        descriptor = [Chem.Lipinski.RingCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'RingCount', save_res)
        selected_descriptors.append("RingCount")
        clear_descriptor_memory(descriptor)
        
    # 14. NumAromaticRings - Number of Aromatic Rings
    if selection[13] == 1:
        descriptor = [Chem.Lipinski.NumAromaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAromaticRings', save_res)
        selected_descriptors.append("NumAromaticRings")
        clear_descriptor_memory(descriptor)
        
    # 15. NumSaturatedRings - Number of Saturated Rings
    if selection[14] == 1:
        descriptor = [Chem.Lipinski.NumSaturatedRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSaturatedRings', save_res)
        selected_descriptors.append("NumSaturatedRings")
        clear_descriptor_memory(descriptor)
        
    # 16. NumAliphaticRings - Number of Aliphatic Rings
    if selection[15] == 1:
        descriptor = [Chem.Lipinski.NumAliphaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAliphaticRings', save_res)
        selected_descriptors.append("NumAliphaticRings")
        clear_descriptor_memory(descriptor)
        
    # 17. LabuteASA - Labute ASA
    if selection[16] == 1:
        descriptor = [Chem.Descriptors.LabuteASA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'LabuteASA', save_res)
        selected_descriptors.append("LabuteASA")
        clear_descriptor_memory(descriptor)
        
    # 18. BalabanJ - Balaban J Index
    if selection[17] == 1:
        descriptor = [Chem.GraphDescriptors.BalabanJ(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'BalabanJ', save_res)
        selected_descriptors.append("BalabanJ")
        clear_descriptor_memory(descriptor)
        
    # 19. BertzCT - Bertz Complexity Index
    if selection[18] == 1:
        descriptor = [Chem.GraphDescriptors.BertzCT(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'BertzCT', save_res)
        selected_descriptors.append("BertzCT")
        clear_descriptor_memory(descriptor)
        
    # 20. Ipc - Information Content
    if selection[19] == 1:
        descriptor = [Chem.GraphDescriptors.Ipc(alpha) for alpha in mols]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'Ipc', save_res)
        selected_descriptors.append("Ipc")
        clear_descriptor_memory(descriptor)
        
    # 21. kappa_Series[1-3]_ind - Kappa Shape Indices (3 descriptors)
    if selection[20] == 1:
        d1 = [Chem.GraphDescriptors.Kappa1(alpha) for alpha in mols]
        d2 = [Chem.GraphDescriptors.Kappa2(alpha) for alpha in mols]
        d3 = [Chem.GraphDescriptors.Kappa3(alpha) for alpha in mols]
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        d3 = np.asarray(d3)
        fps = generating_newfps(fps, [d1,d2,d3], 'kappa_Series[1-3]_ind', save_res)
        selected_descriptors.append("kappa_Series[1-3]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        
    # 22. Chi_Series[13]_ind - Chi Connectivity Indices (13 descriptors)
    if selection[21] == 1:
        d1 = [Chem.GraphDescriptors.Chi0(alpha) for alpha in mols]
        d2 = [Chem.GraphDescriptors.Chi0n(alpha) for alpha in mols]
        d3 = [Chem.GraphDescriptors.Chi0v(alpha) for alpha in mols]
        d4 = [Chem.GraphDescriptors.Chi1(alpha) for alpha in mols]
        d5 = [Chem.GraphDescriptors.Chi1n(alpha) for alpha in mols]
        d6 = [Chem.GraphDescriptors.Chi1v(alpha) for alpha in mols]
        d7 = [Chem.GraphDescriptors.Chi2n(alpha) for alpha in mols]
        d8 = [Chem.GraphDescriptors.Chi2v(alpha) for alpha in mols]
        d9 = [Chem.GraphDescriptors.Chi3n(alpha) for alpha in mols]
        d10 = [Chem.GraphDescriptors.Chi3v(alpha) for alpha in mols]
        d11 = [Chem.GraphDescriptors.Chi4n(alpha) for alpha in mols]
        d12 = [Chem.GraphDescriptors.Chi4v(alpha) for alpha in mols]
        d13 = generate_chi(mols, 'n')
        d14 = generate_chi(mols, 'v')
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        d13 = np.asarray(d13)
        d14 = np.asarray(d14)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14], 'Chi_Series[13]_ind', save_res)
        selected_descriptors.append("Chi_Series[13]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
        clear_descriptor_memory(d13)
        clear_descriptor_memory(d14)
        
    # 23. Phi - Flexibility Index
    if selection[22] == 1:
        descriptor = [Chem.rdMolDescriptors.CalcPhi(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'Phi', save_res)
        selected_descriptors.append("Phi")
        clear_descriptor_memory(descriptor)
        
    # 24. HallKierAlpha - Hall-Kier Alpha
    if selection[23] == 1:
        descriptor = [Chem.GraphDescriptors.HallKierAlpha(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HallKierAlpha', save_res)
        selected_descriptors.append("HallKierAlpha")
        clear_descriptor_memory(descriptor)
        
    # 25. NumAmideBonds - Number of Amide Bonds
    if selection[24] == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumAmideBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAmideBonds', save_res)
        selected_descriptors.append("NumAmideBonds")
        clear_descriptor_memory(descriptor)
        
    # 26. FractionCSP3 - Fraction of sp3 Carbon Atoms (2D descriptor)
    if selection[25] == 1:
        descriptor = [Chem.Lipinski.FractionCSP3(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'FractionCSP3', save_res)
        selected_descriptors.append("FractionCSP3")
        clear_descriptor_memory(descriptor)
        
    # 27. NumSpiroAtoms - Number of Spiro Atoms
    if selection[26] == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumSpiroAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSpiroAtoms', save_res)
        selected_descriptors.append("NumSpiroAtoms")
        clear_descriptor_memory(descriptor)
        
    # 28. NumBridgeheadAtoms - Number of Bridgehead Atoms
    if selection[27] == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumBridgeheadAtoms', save_res)
        selected_descriptors.append("NumBridgeheadAtoms")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors 

# ============================================================================
# PART 3: FEATURE_SELECTION_SELECTION_PART3_VSA_DESCRIPTORS
# ============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Selection Part 3: VSA Descriptors Processing
기능 선택 선택 3부: VSA 서술자 처리
"""

import numpy as np
from rdkit import Chem

# Import functions from part 1
def process_vsa_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """Process all VSA descriptors (5 series, 57 individual descriptors) using selection list"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        import gc
        gc.collect()
    
    # 29. PEOE_VSA_Series[1-14]_ind - PEOE VSA (14 individual descriptors)
    if selection[28] == 1:
        d1 = [Chem.MolSurf.PEOE_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.PEOE_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.PEOE_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.PEOE_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.PEOE_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.PEOE_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.PEOE_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.PEOE_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.PEOE_VSA9(alpha) for alpha in mols]
        d10 = [Chem.MolSurf.PEOE_VSA10(alpha) for alpha in mols]
        d11 = [Chem.MolSurf.PEOE_VSA11(alpha) for alpha in mols]
        d12 = [Chem.MolSurf.PEOE_VSA12(alpha) for alpha in mols]
        d13 = [Chem.MolSurf.PEOE_VSA13(alpha) for alpha in mols]
        d14 = [Chem.MolSurf.PEOE_VSA14(alpha) for alpha in mols]
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        d13 = np.asarray(d13)
        d14 = np.asarray(d14)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14],'PEOE_VSA_Series[1-14]_ind', save_res)
        selected_descriptors.append("PEOE_VSA_Series[1-14]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
        clear_descriptor_memory(d13)
        clear_descriptor_memory(d14)
    
    # 30. SMR_VSA_Series[1-10]_ind - SMR VSA (10 individual descriptors)
    if selection[29] == 1:
        d1 = [Chem.MolSurf.SMR_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.SMR_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.SMR_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.SMR_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.SMR_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.SMR_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.SMR_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.SMR_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.SMR_VSA9(alpha) for alpha in mols]
        d10 = [Chem.MolSurf.SMR_VSA10(alpha) for alpha in mols]
        d1  = np.asarray(d1)
        d2  = np.asarray(d2)
        d3  = np.asarray(d3)
        d4  = np.asarray(d4)
        d5  = np.asarray(d5)
        d6  = np.asarray(d6)
        d7  = np.asarray(d7)
        d8  = np.asarray(d8)
        d9  = np.asarray(d9)
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],'SMR_VSA_Series[1-10]_ind', save_res)
        selected_descriptors.append("SMR_VSA_Series[1-10]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
    
    # 31. SlogP_VSA_Series[1-12]_ind - SlogP VSA (12 individual descriptors)
    if selection[30] == 1:
        d1 = [Chem.MolSurf.SlogP_VSA1(alpha) for alpha in mols]
        d2 = [Chem.MolSurf.SlogP_VSA2(alpha) for alpha in mols]
        d3 = [Chem.MolSurf.SlogP_VSA3(alpha) for alpha in mols]
        d4 = [Chem.MolSurf.SlogP_VSA4(alpha) for alpha in mols]
        d5 = [Chem.MolSurf.SlogP_VSA5(alpha) for alpha in mols]
        d6 = [Chem.MolSurf.SlogP_VSA6(alpha) for alpha in mols]
        d7 = [Chem.MolSurf.SlogP_VSA7(alpha) for alpha in mols]
        d8 = [Chem.MolSurf.SlogP_VSA8(alpha) for alpha in mols]
        d9 = [Chem.MolSurf.SlogP_VSA9(alpha) for alpha in mols]
        d10= [Chem.MolSurf.SlogP_VSA10(alpha) for alpha in mols]
        d11= [Chem.MolSurf.SlogP_VSA11(alpha) for alpha in mols]
        d12= [Chem.MolSurf.SlogP_VSA12(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        d12 = np.asarray(d12)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12],'SlogP_VSA_Series[1-12]_ind', save_res)
        selected_descriptors.append("SlogP_VSA_Series[1-12]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
        clear_descriptor_memory(d12)
    
    # 32. EState_VSA_Series[1-11]_ind - EState VSA (11 individual descriptors)
    if selection[31] == 1:
        d1 = [Chem.EState.EState_VSA.EState_VSA1(alpha) for alpha in mols]
        d2 = [Chem.EState.EState_VSA.EState_VSA2(alpha) for alpha in mols]
        d3 = [Chem.EState.EState_VSA.EState_VSA3(alpha) for alpha in mols]
        d4 = [Chem.EState.EState_VSA.EState_VSA4(alpha) for alpha in mols]
        d5 = [Chem.EState.EState_VSA.EState_VSA5(alpha) for alpha in mols]
        d6 = [Chem.EState.EState_VSA.EState_VSA6(alpha) for alpha in mols]
        d7 = [Chem.EState.EState_VSA.EState_VSA7(alpha) for alpha in mols]
        d8 = [Chem.EState.EState_VSA.EState_VSA8(alpha) for alpha in mols]
        d9 = [Chem.EState.EState_VSA.EState_VSA9(alpha) for alpha in mols]
        d10 = [Chem.EState.EState_VSA.EState_VSA10(alpha) for alpha in mols]
        d11 = [Chem.EState.EState_VSA.EState_VSA11(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        d11 = np.asarray(d11)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11],'EState_VSA_Series[1-11]_ind', save_res)
        selected_descriptors.append("EState_VSA_Series[1-11]_ind")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
        clear_descriptor_memory(d11)
    
    # 33. VSA_EState_Series[1-10] - VSA EState (10 individual descriptors)
    if selection[32] == 1:
        d1  = [Chem.EState.EState_VSA.VSA_EState1(alpha) for alpha in mols]
        d2  = [Chem.EState.EState_VSA.VSA_EState2(alpha) for alpha in mols]
        d3  = [Chem.EState.EState_VSA.VSA_EState3(alpha) for alpha in mols]
        d4  = [Chem.EState.EState_VSA.VSA_EState4(alpha) for alpha in mols]
        d5  = [Chem.EState.EState_VSA.VSA_EState5(alpha) for alpha in mols]
        d6  = [Chem.EState.EState_VSA.VSA_EState6(alpha) for alpha in mols]
        d7  = [Chem.EState.EState_VSA.VSA_EState7(alpha) for alpha in mols]
        d8  = [Chem.EState.EState_VSA.VSA_EState8(alpha) for alpha in mols]
        d9  = [Chem.EState.EState_VSA.VSA_EState9(alpha) for alpha in mols]
        d10  = [Chem.EState.EState_VSA.VSA_EState10(alpha) for alpha in mols]
        d1  = np.asarray(d1) 
        d2  = np.asarray(d2) 
        d3  = np.asarray(d3) 
        d4  = np.asarray(d4) 
        d5  = np.asarray(d5) 
        d6  = np.asarray(d6) 
        d7  = np.asarray(d7) 
        d8  = np.asarray(d8) 
        d9  = np.asarray(d9) 
        d10 = np.asarray(d10)
        fps = generating_newfps(fps, [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10],'VSA_EState_Series[1-10]', save_res)
        selected_descriptors.append("VSA_EState_Series[1-10]")
        clear_descriptor_memory(d1)
        clear_descriptor_memory(d2)
        clear_descriptor_memory(d3)
        clear_descriptor_memory(d4)
        clear_descriptor_memory(d5)
        clear_descriptor_memory(d6)
        clear_descriptor_memory(d7)
        clear_descriptor_memory(d8)
        clear_descriptor_memory(d9)
        clear_descriptor_memory(d10)
    
    return fps, selected_descriptors 

# ============================================================================
# PART 4: FEATURE_SELECTION_SELECTION_PART4_3D_DESCRIPTORS
# ============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Selection Part 4: 3D Descriptors and rdMolDescriptors Processing
기능 선택 선택 4부: 3D 서술자 및 rdMolDescriptors 처리
"""

import numpy as np
import gc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Import functions from part 1
def process_rdmol_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np"):
    """Process rdMolDescriptors (3 descriptors) using selection list"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        gc.collect()
    
    # 34. MQNs - Molecular Quantum Numbers
    if selection[33] == 1:
        descriptor = [Chem.rdMolDescriptors.MQNs_(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'MQNs', save_res)
        selected_descriptors.append("MQNs")
        clear_descriptor_memory(descriptor)
        
    # 35. AUTOCORR2D - 2D Autocorrelation
    if selection[34] == 1:
        descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR2D(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR2D', save_res)
        selected_descriptors.append("AUTOCORR2D")
        clear_descriptor_memory(descriptor)
        
    # 36. BCUT2D - 2D BCUT Descriptors
    if selection[35] == 1:
        descriptor = compute_descriptors_parallel(mols)
        fps = generating_newfps(fps, descriptor, 'BCUT2D', save_res)
        selected_descriptors.append("BCUT2D")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors

def process_3d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res="np", mols_3d=None):
    """Process 3D descriptors (15 descriptors including FractionCSP3) with conformer generation using selection list"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        gc.collect()
    
    def safe_compute_descriptor(mols_list, descriptor_func, descriptor_name):
        """Safely compute descriptor with error handling and size validation"""
        try:
            descriptor = []
            for mol in mols_list:
                try:
                    val = descriptor_func(mol)
                    descriptor.append(val)
                except Exception as e:
                    print(f"Error computing {descriptor_name} for molecule: {e}")
                    descriptor.append(None)
            
            # Check if all values are valid
            valid_descriptor = []
            for i, val in enumerate(descriptor):
                # Handle numpy arrays properly
                if isinstance(val, np.ndarray):
                    if val.size == 1:  # Single value array
                        val = val.item()
                    else:  # Multi-value array, take mean
                        # Remove NaN and Inf values before taking mean
                        flattened = val.flatten()
                        if flattened.size > 0:
                            # Use np.any() and np.all() to avoid "array ambiguous" error
                            valid_mask = ~(np.isnan(flattened) | np.isinf(flattened))
                            if np.any(valid_mask):
                                valid_values = flattened[valid_mask]
                                val = float(np.mean(valid_values))
                            else:
                                val = 0.0
                        else:
                            val = 0.0
                
                # Handle all value types safely
                if val is not None:
                    if isinstance(val, np.ndarray):
                        if val.size == 1:
                            val_item = val.item()
                            if not np.isnan(val_item) and not np.isinf(val_item):
                                valid_descriptor.append(val_item)
                            else:
                                valid_descriptor.append(0.0)
                        else:
                            # Multi-dimensional array - flatten and take mean
                            flattened = val.flatten()
                            if flattened.size > 0:
                                # Use np.any() and np.all() to avoid "array ambiguous" error
                                valid_mask = ~(np.isnan(flattened) | np.isinf(flattened))
                                if np.any(valid_mask):
                                    valid_values = flattened[valid_mask]
                                    valid_descriptor.append(float(np.mean(valid_values)))
                                else:
                                    valid_descriptor.append(0.0)
                            else:
                                valid_descriptor.append(0.0)
                    else:
                        # For non-array values
                        if not np.isnan(val) and not np.isinf(val):
                            valid_descriptor.append(val)
                        else:
                            valid_descriptor.append(0.0)
                else:
                    valid_descriptor.append(0.0)
            
            # Ensure the descriptor has the same length as input
            if len(valid_descriptor) != len(mols_list):
                print(f"Warning: {descriptor_name} size mismatch ({len(valid_descriptor)} vs {len(mols_list)}). Padding with zeros.")
                while len(valid_descriptor) < len(mols_list):
                    valid_descriptor.append(0.0)
                valid_descriptor = valid_descriptor[:len(mols_list)]
            
            return valid_descriptor
        except Exception as e:
            print(f"Error occurred in {descriptor_name}: {e}")
            # Return zeros for failed descriptor
            return [0.0] * len(mols_list)
    
    def safe_compute_multidimensional_descriptor(mols_list, descriptor_func, descriptor_name):
        """Safely compute multi-dimensional descriptor with robust array handling"""
        try:
            descriptor = []
            for mol in mols_list:
                try:
                    val = descriptor_func(mol)
                    descriptor.append(val)
                except Exception as e:
                    print(f"Error computing {descriptor_name} for molecule: {e}")
                    descriptor.append(None)
            
            # Check if all values are valid
            valid_descriptor = []
            for i, val in enumerate(descriptor):
                if val is not None:
                    if isinstance(val, np.ndarray):
                        # Handle multi-dimensional arrays
                        if val.size == 1:
                            # Single value array
                            val_item = val.item()
                            if not np.isnan(val_item) and not np.isinf(val_item):
                                valid_descriptor.append(val_item)
                            else:
                                valid_descriptor.append(0.0)
                        else:
                            # Multi-dimensional array - robust flattening and mean calculation
                            try:
                                flattened = val.flatten()
                                if flattened.size > 0:
                                    # Create a copy to avoid modifying original array
                                    flat_copy = flattened.copy()
                                    # Replace NaN and Inf with zeros
                                    flat_copy = np.where(np.isnan(flat_copy), 0.0, flat_copy)
                                    flat_copy = np.where(np.isinf(flat_copy), 0.0, flat_copy)
                                    # Take mean of all values (including zeros)
                                    val = float(np.mean(flat_copy))
                                    valid_descriptor.append(val)
                                else:
                                    valid_descriptor.append(0.0)
                            except Exception as e:
                                print(f"Error processing {descriptor_name} array: {e}")
                                valid_descriptor.append(0.0)
                    else:
                        # For non-array values
                        if not np.isnan(val) and not np.isinf(val):
                            valid_descriptor.append(val)
                        else:
                            valid_descriptor.append(0.0)
                else:
                    valid_descriptor.append(0.0)
            
            # Ensure the descriptor has the same length as input
            if len(valid_descriptor) != len(mols_list):
                print(f"Warning: {descriptor_name} size mismatch ({len(valid_descriptor)} vs {len(mols_list)}). Padding with zeros.")
                while len(valid_descriptor) < len(mols_list):
                    valid_descriptor.append(0.0)
                valid_descriptor = valid_descriptor[:len(mols_list)]
            
            return valid_descriptor
        except Exception as e:
            print(f"Error occurred in {descriptor_name}: {e}")
            # Return zeros for failed descriptor
            return [0.0] * len(mols_list)
    
    # Use pre-generated 3D conformers if available, otherwise generate them
    if mols_3d is not None:
        mols2 = mols_3d
    else:
        mols2 = process_molecules_parallel(mols, max_workers=8)
    del mols  # Free memory
    gc.collect()
    
    if mols2:
        # 37. Asphericity - Asphericity
        if selection[36] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcAsphericity, "Asphericity")
            fps = generating_newfps(fps, descriptor, 'Asphericity', save_res)
            selected_descriptors.append("Asphericity")
            clear_descriptor_memory(descriptor)
            
        # 38. PBF - Principal Moments of Inertia B
        if selection[37] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcPBF, "PBF")
            fps = generating_newfps(fps, descriptor, 'PBF', save_res)
            selected_descriptors.append("PBF")
            clear_descriptor_memory(descriptor)
            
        # 39. RadiusOfGyration - Radius of Gyration
        if selection[38] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcRadiusOfGyration, "RadiusOfGyration")
            fps = generating_newfps(fps, descriptor, 'RadiusOfGyration', save_res)
            selected_descriptors.append("RadiusOfGyration")
            clear_descriptor_memory(descriptor)
            
        # 40. InertialShapeFactor - Inertial Shape Factor
        if selection[39] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcInertialShapeFactor, "InertialShapeFactor")
            fps = generating_newfps(fps, descriptor, 'InertialShapeFactor', save_res)
            selected_descriptors.append("InertialShapeFactor")
            clear_descriptor_memory(descriptor)
            
        # 41. Eccentricity - Eccentricity
        if selection[40] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcEccentricity, "Eccentricity")
            fps = generating_newfps(fps, descriptor, 'Eccentricity', save_res)
            selected_descriptors.append("Eccentricity")
            clear_descriptor_memory(descriptor)
            
        # 42. SpherocityIndex - Spherocity Index
        if selection[41] == 1:
            descriptor = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcSpherocityIndex, "SpherocityIndex")
            fps = generating_newfps(fps, descriptor, 'SpherocityIndex', save_res)
            selected_descriptors.append("SpherocityIndex")
            clear_descriptor_memory(descriptor)
            
        # 43. PMI_series[1-3]_ind - Principal Moments of Inertia (3 descriptors)
        if selection[42] == 1:
            d1 = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcPMI1, "PMI1")
            d2 = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcPMI2, "PMI2")
            d3 = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcPMI3, "PMI3")
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
            
        # 44. NPR_series[1-2]_ind - Normalized Principal Moments Ratio (2 descriptors)
        if selection[43] == 1:
            d1 = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcNPR1, "NPR1")
            d2 = safe_compute_descriptor(mols2, Chem.rdMolDescriptors.CalcNPR2, "NPR2")
            d1 = np.asarray(d1)
            d2 = np.asarray(d2)
            fps = generating_newfps(fps, [d1,d2], 'NPR_series[1-2]_ind', save_res)
            selected_descriptors.append("NPR_series[1-2]_ind")
            clear_descriptor_memory(d1)
            clear_descriptor_memory(d2)
            
                # 45. AUTOCORR3D - 3D Autocorrelation
        if selection[44] == 1:
            descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR3D(mols) for mols in mols2]
            fps = generating_newfps(fps, descriptor, 'AUTOCORR3D', save_res)
            selected_descriptors.append("AUTOCORR3D")
            clear_descriptor_memory(descriptor)
            
        # 46. RDF - Radial Distribution Function
        if selection[45] == 1:
            descriptor = [Chem.rdMolDescriptors.CalcRDF(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'RDF', save_res)
            selected_descriptors.append("RDF")
            clear_descriptor_memory(descriptor)
            
        # 47. MORSE - Morse Descriptors
        if selection[46] == 1:
            descriptor = [Chem.rdMolDescriptors.CalcMORSE(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'MORSE', save_res)
            selected_descriptors.append("MORSE")
            clear_descriptor_memory(descriptor)
            
        # 48. WHIM - WHIM Descriptors
        if selection[47] == 1:
            descriptor = [Chem.rdMolDescriptors.CalcWHIM(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'WHIM', save_res)
            selected_descriptors.append("WHIM")
            clear_descriptor_memory(descriptor)
            
        # 49. GETAWAY - GETAWAY Descriptors
        if selection[48] == 1:
            descriptor = [Chem.rdMolDescriptors.CalcGETAWAY(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'GETAWAY', save_res)
            selected_descriptors.append("GETAWAY")
            clear_descriptor_memory(descriptor)
    else:
        print("Warning: No 3D conformers generated. 3D descriptors will be skipped.")
    
    return fps, selected_descriptors 

# ============================================================================
# PART 5: FEATURE_SELECTION_SELECTION_PART5_MAIN_FUNCTIONS
# ============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Selection Selection Part 5: Main Functions and PyTorch Model Creation
기능 선택 선택 5부: 메인 함수들 및 PyTorch 모델 생성
"""

import numpy as np
import torch
import torch.nn as nn
import optuna

# Import functions from previous parts
# PyTorch DNN Model (TensorFlow 대신)
class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, hidden_dim2=496, dropout_rate=0.2):
        super(DNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim2, momentum=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
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

def selection_data_descriptor_compress(selection, fps, mols, name, target_path="result", save_res="np", mols_3d=None):
    """
    Main function for chemical descriptor selection using pre-selected descriptors
    Based on the original feature_selection.py with PyTorch compatibility
    
    Args:
        selection: List of selected descriptor indices (0 or 1 for each descriptor)
        fps: Input fingerprints (Morgan + MACCS + Avalon = "all")
        mols: List of RDKit mol objects
        name: Name for saving results
        target_path: Path to save results
        save_res: Save format ("np" or "pd")
        mols_3d: Pre-generated 3D conformers (optional)
    
    Returns:
        fps_updated: Updated fingerprints with selected descriptors
        selected_descriptors: List of selected descriptor names
    """
    selected_descriptors = []
    
    # Process 2D descriptors (28 descriptors)
    fps, selected_descriptors = process_2d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res)
    
    # Process VSA descriptors (5 series, 57 individual descriptors)
    fps, selected_descriptors = process_vsa_descriptors_selection(selection, fps, mols, selected_descriptors, save_res)
    
    # Process rdMolDescriptors (3 descriptors)
    fps, selected_descriptors = process_rdmol_descriptors_selection(selection, fps, mols, selected_descriptors, save_res)
    
    # Process 3D descriptors (15 descriptors including FractionCSP3)
    fps, selected_descriptors = process_3d_descriptors_selection(selection, fps, mols, selected_descriptors, save_res, mols_3d)
    
    # Limit to 49 descriptors if more are selected
    if len(selected_descriptors) > 49:
        selected_descriptors = selected_descriptors[:49]
    
    # Save results if requested
    if save_res == "pd":
        fps.to_csv(f'{target_path}/{name}_feature_selection.csv')
    
    # Ensure final data type
    fps = fps.astype('float')
    
    return fps, selected_descriptors

def selection_fromStudy_compress(study_name, storage, unfixed=False, showlog=True):
    """
    Get best feature selection from Optuna study
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    
    if showlog:
        print(f"Best trial: {best_trial.number}")
        print(f"Best value: {best_trial.value}")
        print(f"Best parameters: {best_trial.params}")
    
    return best_trial.params

def selection_structure_compress(study_name, storage, input_dim, returnOnly=False):
    """
    Get best network structure from Optuna study and create PyTorch model
    """
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    
    # Create PyTorch model with best parameters
    model = DNNModel(input_dim=input_dim)
    
    if not returnOnly:
        # Save model using PyTorch
        torch.save(model.state_dict(), f'model_{study_name}.pth')
    
    return model, best_trial.params

def convert_params_to_selection(best_feature_params):
    """
    Convert best feature parameters to selection list
    """
    # Map descriptor names to their indices in the selection list
    descriptor_to_index = {
        'MolWt': 0, 'MolLogP': 1, 'MolMR': 2, 'TPSA': 3, 'NumRotatableBonds': 4,
        'HeavyAtomCount': 5, 'NumHAcceptors': 6, 'NumHDonors': 7, 'NumHeteroatoms': 8,
        'NumValenceElectrons': 9, 'NHOHCount': 10, 'NOCount': 11, 'RingCount': 12,
        'NumAromaticRings': 13, 'NumSaturatedRings': 14, 'NumAliphaticRings': 15,
        'LabuteASA': 16, 'BalabanJ': 17, 'BertzCT': 18, 'Ipc': 19, 'kappa_Series[1-3]_ind': 20,
        'Chi_Series[13]_ind': 21, 'Phi': 22, 'HallKierAlpha': 23, 'NumAmideBonds': 24,
        'FractionCSP3': 25, 'NumSpiroAtoms': 26, 'NumBridgeheadAtoms': 27,
        'PEOE_VSA_Series[1-14]_ind': 28, 'SMR_VSA_Series[1-10]_ind': 29,
        'SlogP_VSA_Series[1-12]_ind': 30, 'EState_VSA_Series[1-11]_ind': 31,
        'VSA_EState_Series[1-10]': 32, 'MQNs': 33, 'AUTOCORR2D': 34, 'BCUT2D': 35,
        'Asphericity': 36, 'PBF': 37, 'RadiusOfGyration': 38, 'InertialShapeFactor': 39,
        'Eccentricity': 40, 'SpherocityIndex': 41, 'PMI_series[1-3]_ind': 42,
        'NPR_series[1-2]_ind': 43, 'AUTOCORR3D': 44, 'RDF': 45, 'MORSE': 46,
        'WHIM': 47, 'GETAWAY': 48
    }
    
    # Initialize selection list with zeros
    selection = [0] * 49
    
    # Set values based on best_feature_params
    for descriptor_name, value in best_feature_params.items():
        if descriptor_name in descriptor_to_index:
            selection[descriptor_to_index[descriptor_name]] = value
    
    return selection 
