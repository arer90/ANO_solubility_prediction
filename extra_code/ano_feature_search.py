#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer: Lee, Seungjin (arer90)

Chemical Descriptor Selection with Optuna - Complete Module
==========================================================

PURPOSE:
This module implements Optuna-based feature selection for molecular descriptors.
It systematically searches through 51 different descriptor categories to find
the optimal combination for molecular property prediction.

KEY FEATURES:
- Binary selection of descriptor categories (include/exclude)
- Parallel processing for efficient descriptor calculation
- 3D conformer generation with fallback mechanisms
- Memory-efficient implementation for large datasets
- Integration with Optuna for Bayesian optimization

DESCRIPTOR STRUCTURE:
The module is organized into logical sections:
1. Part 1: Imports and Helper Functions
2. Part 2: 2D Descriptors (27 types) - Basic molecular properties
3. Part 3: VSA Descriptors (5 series) - Surface area based properties
4. Part 4: 3D Descriptors (14 types) - Conformation-dependent properties
5. Part 5: Main Function and Integration - Optuna trial handling

TOTAL FEATURES:
- Base fingerprints: Morgan (2048) + MACCS (167) + Avalon (512) = 2727
- Optional descriptors: Up to 51 categories (~882 individual features)
- Maximum features: ~3609 (depending on 3D availability)

USAGE:
This module is called by ANO module 5 for feature optimization, where Optuna
suggests which descriptor categories to include in each trial.
"""


# === CHUNK 1 ===
"""
Chemical Descriptor Selection with Optuna - Part 1: Imports and Helper Functions
==============================================================================

This section contains essential imports and utility functions for molecular
descriptor calculation, including 3D conformer generation and parallel processing.
"""

import os
import numpy as np
import pandas as pd
import gc
import platform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import RDConfig
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, rdDistGeom, rdPartialCharges
from rdkit.Chem.AllChem import GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.Descriptors import ExactMolWt

# OS-specific multiprocessing support
if platform.system() == "Windows":
    import multiprocessing
    multiprocessing.freeze_support()

# Elements not supported for 3D descriptor calculation
UNSUPPORTED_ELEMENTS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Se', 'Te', 'Rn', 'Xe', 'Kr', 'He', 'Ne', 'Ar'
}

def prefilter_3d_conformers(smiles_list, y_list):
    """
    Pre-filter molecules that can generate 3D conformers.
    
    This function identifies molecules that can successfully generate 3D conformations
    needed for 3D descriptor calculation. It filters out:
    - Invalid SMILES strings
    - Molecules with unsupported elements
    - Molecules that fail conformer generation
    
    The function uses parallel processing for efficiency and implements fallback
    mechanisms for difficult molecules.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    y_list : list
        List of target values (solubility)
    
    Returns:
    --------
    tuple
        (filtered_smiles, filtered_y, filtered_mols, filtered_mols_3d)
        Only molecules that successfully generated 3D conformers
    """
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
    
    # Generate 3D conformers with stricter filtering
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
                # Additional check: ensure the conformer is valid
                try:
                    # Try to access conformer properties to ensure it's valid
                    conf = mol.GetConformer(0)
                    if conf.Is3D():
                        successful_indices.append(original_index)
                        successful_mols.append(mols[i])
                        successful_mols_3d.append(mol)
                        successful_smiles.append(smiles_list[original_index])
                        successful_y.append(y_list[original_index])
                    else:
                        failed_indices.append(original_index)
                        print(f"  Warning: Non-3D conformer for molecule {original_index}")
                except Exception as e:
                    failed_indices.append(original_index)
                    print(f"  Warning: Invalid conformer for molecule {original_index}: {e}")
            else:
                failed_indices.append(original_index)
        
        print(f"3D conformer generation successful: {len(successful_mols)} molecules")
        print(f"3D conformer generation failed: {len(failed_indices)} molecules")
        
        # Save information about failed molecules
        if failed_indices:
            import os
            failed_file = f"3d_failed_molecules_{len(smiles_list)}_total.txt"
            with open(failed_file, 'w') as f:
                f.write(f"3D Conformer Generation Failed Molecules\\n")
                f.write(f"="*50 + "\\n")
                f.write(f"Total molecules: {len(smiles_list)}\\n")
                f.write(f"Valid molecules: {len(mols)}\\n")
                f.write(f"3D conformer generation successful: {len(successful_mols)} molecules\\n")
                f.write(f"3D conformer generation failed: {len(failed_indices)} molecules\\n")
                f.write(f"Success rate: {len(successful_mols) / len(smiles_list) * 100:.2f}%\\n\\n")
                f.write(f"Failed molecule indices: {failed_indices}\\n\\n")
                f.write(f"Failed molecule details:\\n")
                for idx in failed_indices:
                    f.write(f"  Index {idx}: {smiles_list[idx]}\\n")
            
            print(f"3D failed molecules info saved to: {failed_file}")
        
        return successful_smiles, successful_y, successful_mols, successful_mols_3d
    else:
        print("No 3D conformers generated, returning original data")
        return smiles_list, y_list, mols, []

def get_executor():
    """Get appropriate executor based on OS for multiprocessing"""
    if platform.system() == "Windows":
        return ThreadPoolExecutor
    else:
        return ProcessPoolExecutor

def has_unsupported_elements(mol):
    """Check if molecule contains unsupported elements for 3D descriptor calculation"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in UNSUPPORTED_ELEMENTS:
            return True
    return False

def mol3d_safe(mol, timeout_seconds=30):
    """Generate 3D conformer with timeout to prevent infinite loops"""
    import threading
    import time
    
    result = {'mol': None, 'error': None}
    
    def generate_conformer():
        try:
            # Check for unsupported elements
            if has_unsupported_elements(mol):
                result['error'] = "Unsupported elements"
                return
            
            # Generate 3D conformer
            mol3d = Chem.AddHs(mol)
            embed_result = AllChem.EmbedMolecule(mol3d, randomSeed=42)
            
            # Check if embedding was successful
            if embed_result == -1:
                result['error'] = "Embedding failed"
                return
            
            # Optimize the conformer
            optimize_result = AllChem.MMFFOptimizeMolecule(mol3d)
            
            # Check if optimization was successful
            if optimize_result == -1:
                result['error'] = "Optimization failed"
                return
            
            # Additional validation
            if mol3d.GetNumConformers() == 0:
                result['error'] = "No conformers generated"
                return
            
            # Check if the conformer is actually 3D
            conf = mol3d.GetConformer(0)
            if not conf.Is3D():
                result['error'] = "Generated conformer is not 3D"
                return
            
            result['mol'] = mol3d
        except Exception as e:
            result['error'] = str(e)
    
    # Run in separate thread with timeout
    thread = threading.Thread(target=generate_conformer)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        print(f"  Warning: 3D conformer generation timed out for molecule")
        return None
    
    if result['error']:
        print(f"  Warning: 3D conformer generation failed: {result['error']}")
        return None
    
    return result['mol']

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
                try:
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
                except Exception as e:
                    print(f"[-1-] Error occurred: {e}")
                    
            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
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
                except Exception as e:
                    print(f"[-2-] Error occurred: {e}")
                    
            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
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
                except Exception as e:
                    print(f"[-3-] Error occurred: {e}")
                    
            else:
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        # Pad with zeros
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length]
                new_fps[descriptor_name] = descriptor.flatten()
                del descriptor
                
            new_fps = new_fps.replace([np.inf, -np.inf], np.nan).fillna(0)
            return new_fps
            
        else:
            new_fps = fps
            
            if descriptor is None:
                pass
            elif isinstance(descriptor, np.ndarray) and descriptor.ndim >= 2:
                try:
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
                except Exception as e:
                    print(f"[-1-] Error occurred: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], np.ndarray):
                try:
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
                except Exception as e:
                    print(f"[-2-] Error occurred: {e}")
            elif isinstance(descriptor, list) and isinstance(descriptor[0], list):
                try:
                    descriptor = np.asarray(descriptor).astype('float')
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
                except Exception as e:
                    print(f"[-3-] Error occurred: {e}")
            else:
                descriptor = np.asarray(descriptor).astype('float')
                if len(descriptor) != target_length:
                    if len(descriptor) < target_length:
                        # Pad with zeros
                        descriptor = np.pad(descriptor, (0, target_length - len(descriptor)), 'constant')
                    else:
                        # Truncate
                        descriptor = descriptor[:target_length]
                new_fps = np.concatenate([new_fps, descriptor[:,None]], axis=1)
                del descriptor
                
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
    Executor = get_executor()
    with Executor(max_workers=n_jobs) as executor:
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
    Executor = get_executor()
    with Executor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sanitize_and_compute_descriptor, mol) for mol in mols if mol is not None]
        descriptors = [future.result() for future in futures]

    max_length = max(len(d) for d in descriptors)
    padded_descriptors = np.array([np.pad(d, (0, max_length - len(d)), 'constant') for d in descriptors])

    return padded_descriptors

def process_molecules_parallel(mols, max_workers=4, chunk_size=100):
    """Process molecules in parallel for 3D conformer generation"""
    results = [None] * len(mols)  # Initialize with None to maintain length

    for i in range(0, len(mols), chunk_size):
        chunk = mols[i:i + chunk_size]
        Executor = get_executor()
        with Executor(max_workers=max_workers) as executor:
            futures = {executor.submit(mol3d_safe, mol): idx for idx, mol in enumerate(chunk)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    results[i + idx] = result  # Store at original position

        gc.collect()
    
    return [mol for mol in results if mol is not None]  # Remove None values

def compute_3d_descriptor_safe(mols, descriptor_func, descriptor_name, timeout_seconds=60):
    """Compute 3D descriptors with timeout to prevent infinite loops"""
    import threading
    
    result = {'descriptor': None, 'error': None}
    
    def compute_descriptor():
        try:
            descriptor = descriptor_func(mols)
            result['descriptor'] = descriptor
        except Exception as e:
            result['error'] = str(e)
    
    # Run in separate thread with timeout
    thread = threading.Thread(target=compute_descriptor)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"  Warning: {descriptor_name} computation timed out")
        return None
            
    if result['error']:
        print(f"  Warning: {descriptor_name} computation failed: {result['error']}")
        return None 
    
    return result['descriptor']


# === CHUNK 2 ===

# Import functions from part 1 (will be merged)

def process_2d_descriptors(trial, fps, mols, selected_descriptors, save_res="np"):
    """Process all 2D descriptors (27 descriptors)"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        import gc
        gc.collect()
    
    # 1. MolWt - Molecular Weight
    if trial.suggest_int("MolWt", 0, 1) == 1:
        descriptor = [Descriptors.ExactMolWt(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolWt', save_res)
        selected_descriptors.append("MolWt")
        clear_descriptor_memory(descriptor)
        
    # 2. MolLogP - Molecular LogP
    if trial.suggest_int("MolLogP", 0, 1) == 1:
        descriptor = [Chem.Crippen.MolLogP(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolLogP', save_res)
        selected_descriptors.append("MolLogP")
        clear_descriptor_memory(descriptor)
        
    # 3. MolMR - Molecular Refractivity
    if trial.suggest_int("MolMR", 0, 1) == 1:
        descriptor = [Chem.Crippen.MolMR(alpha) for alpha in mols]    
        fps = generating_newfps(fps, descriptor, 'MolMR', save_res)
        selected_descriptors.append("MolMR")
        clear_descriptor_memory(descriptor)
        
    # 4. TPSA - Topological Polar Surface Area
    if trial.suggest_int("TPSA", 0, 1) == 1:
        descriptor = [Descriptors.TPSA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'TPSA', save_res)
        selected_descriptors.append("TPSA")
        clear_descriptor_memory(descriptor)
        
    # 5. NumRotatableBonds - Number of Rotatable Bonds
    if trial.suggest_int("NumRotatableBonds", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumRotatableBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumRotatableBonds', save_res)
        selected_descriptors.append("NumRotatableBonds")
        clear_descriptor_memory(descriptor)
        
    # 6. HeavyAtomCount - Heavy Atom Count
    if trial.suggest_int("HeavyAtomCount", 0, 1) == 1:
        descriptor = [Descriptors.HeavyAtomCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HeavyAtomCount', save_res)
        selected_descriptors.append("HeavyAtomCount")
        clear_descriptor_memory(descriptor)
        
    # 7. NumHAcceptors - Number of H Acceptors
    if trial.suggest_int("NumHAcceptors", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumHAcceptors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHAcceptors', save_res)
        selected_descriptors.append("NumHAcceptors")
        clear_descriptor_memory(descriptor)
        
    # 8. NumHDonors - Number of H Donors
    if trial.suggest_int("NumHDonors", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumHDonors(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHDonors', save_res)
        selected_descriptors.append("NumHDonors")
        clear_descriptor_memory(descriptor)
        
    # 9. NumHeteroatoms - Number of Heteroatoms
    if trial.suggest_int("NumHeteroatoms", 0, 1) == 1:
        descriptor = [Descriptors.NumHeteroatoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumHeteroatoms', save_res)
        selected_descriptors.append("NumHeteroatoms")
        clear_descriptor_memory(descriptor)
        
    # 10. NumValenceElectrons - Number of Valence Electrons
    if trial.suggest_int("NumValenceElectrons", 0, 1) == 1:
        descriptor = [Descriptors.NumValenceElectrons(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumValenceElectrons', save_res)
        selected_descriptors.append("NumValenceElectrons")
        clear_descriptor_memory(descriptor)
        
    # 11. NHOHCount - NHOH Count
    if trial.suggest_int("NHOHCount", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NHOHCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NHOHCount', save_res)
        selected_descriptors.append("NHOHCount")
        clear_descriptor_memory(descriptor)
        
    # 12. NOCount - NO Count
    if trial.suggest_int("NOCount", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NOCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NOCount', save_res)
        selected_descriptors.append("NOCount")
        clear_descriptor_memory(descriptor)
        
    # 13. RingCount - Ring Count
    if trial.suggest_int("RingCount", 0, 1) == 1:
        descriptor = [Chem.Lipinski.RingCount(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'RingCount', save_res)
        selected_descriptors.append("RingCount")
        clear_descriptor_memory(descriptor)
        
    # 14. NumAromaticRings - Number of Aromatic Rings
    if trial.suggest_int("NumAromaticRings", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumAromaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAromaticRings', save_res)
        selected_descriptors.append("NumAromaticRings")
        clear_descriptor_memory(descriptor)
        
    # 15. NumSaturatedRings - Number of Saturated Rings
    if trial.suggest_int("NumSaturatedRings", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumSaturatedRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSaturatedRings', save_res)
        selected_descriptors.append("NumSaturatedRings")
        clear_descriptor_memory(descriptor)
        
    # 16. NumAliphaticRings - Number of Aliphatic Rings
    if trial.suggest_int("NumAliphaticRings", 0, 1) == 1:
        descriptor = [Chem.Lipinski.NumAliphaticRings(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAliphaticRings', save_res)
        selected_descriptors.append("NumAliphaticRings")
        clear_descriptor_memory(descriptor)
        
    # 17. LabuteASA - Labute ASA
    if trial.suggest_int("LabuteASA", 0, 1) == 1:
        descriptor = [Chem.Descriptors.LabuteASA(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'LabuteASA', save_res)
        selected_descriptors.append("LabuteASA")
        clear_descriptor_memory(descriptor)
        
    # 18. BalabanJ - Balaban J Index
    if trial.suggest_int("BalabanJ", 0, 1) == 1:
        descriptor = [Chem.GraphDescriptors.BalabanJ(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'BalabanJ', save_res)
        selected_descriptors.append("BalabanJ")
        clear_descriptor_memory(descriptor)
        
    # 19. BertzCT - Bertz Complexity Index
    if trial.suggest_int("BertzCT", 0, 1) == 1:
        descriptor = [Chem.GraphDescriptors.BertzCT(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'BertzCT', save_res)
        selected_descriptors.append("BertzCT")
        clear_descriptor_memory(descriptor)
        
    # 20. Ipc - Information Content
    if trial.suggest_int("Ipc", 0, 1) == 1:
        descriptor = [Chem.GraphDescriptors.Ipc(alpha) for alpha in mols]
        descriptor = Normalization(descriptor)
        fps = generating_newfps(fps, descriptor, 'Ipc', save_res)
        selected_descriptors.append("Ipc")
        clear_descriptor_memory(descriptor)
        
    # 21. kappa_Series[1-3]_ind - Kappa Shape Indices (3 descriptors)
    if trial.suggest_int("kappa_Series[1-3]_ind", 0, 1) == 1:
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
    if trial.suggest_int("Chi_Series[13]_ind", 0, 1) == 1:
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
    if trial.suggest_int("Phi", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.CalcPhi(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'Phi', save_res)
        selected_descriptors.append("Phi")
        clear_descriptor_memory(descriptor)
        
    # 24. HallKierAlpha - Hall-Kier Alpha
    if trial.suggest_int("HallKierAlpha", 0, 1) == 1:
        descriptor = [Chem.GraphDescriptors.HallKierAlpha(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'HallKierAlpha', save_res)
        selected_descriptors.append("HallKierAlpha")
        clear_descriptor_memory(descriptor)
        
    # 25. NumAmideBonds - Number of Amide Bonds
    if trial.suggest_int("NumAmideBonds", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumAmideBonds(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumAmideBonds', save_res)
        selected_descriptors.append("NumAmideBonds")
        clear_descriptor_memory(descriptor)
        
    # 26. NumSpiroAtoms - Number of Spiro Atoms
    if trial.suggest_int("NumSpiroAtoms", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumSpiroAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumSpiroAtoms', save_res)
        selected_descriptors.append("NumSpiroAtoms")
        clear_descriptor_memory(descriptor)
        
    # 27. NumBridgeheadAtoms - Number of Bridgehead Atoms
    if trial.suggest_int("NumBridgeheadAtoms", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'NumBridgeheadAtoms', save_res)
        selected_descriptors.append("NumBridgeheadAtoms")
        clear_descriptor_memory(descriptor)
    
    # 35. FractionCSP3 - Fraction of sp3 Carbon Atoms (2D descriptor)
    if trial.suggest_int("FractionCSP3", 0, 1) == 1:
        descriptor = [Chem.Lipinski.FractionCSP3(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'FractionCSP3', save_res)
        selected_descriptors.append("FractionCSP3")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors 


# === CHUNK 3 ===

import numpy as np
from rdkit import Chem

# Import functions from part 1 (will be merged)

def process_vsa_descriptors(trial, fps, mols, selected_descriptors, save_res="np"):
    """Process all VSA descriptors (5 series, 57 individual descriptors)"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        import gc
        gc.collect()
    
    # 28. PEOE_VSA_Series[1-14]_ind - PEOE VSA (14 individual descriptors)
    if trial.suggest_int("PEOE_VSA_Series[1-14]_ind", 0, 1) == 1:
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
    
    # 29. SMR_VSA_Series[1-10]_ind - SMR VSA (10 individual descriptors)
    if trial.suggest_int("SMR_VSA_Series[1-10]_ind", 0, 1) == 1:
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
    
    # 30. SlogP_VSA_Series[1-12]_ind - SlogP VSA (12 individual descriptors)
    if trial.suggest_int("SlogP_VSA_Series[1-12]_ind", 0, 1) == 1:
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
    
    # 31. EState_VSA_Series[1-11]_ind - EState VSA (11 individual descriptors)
    if trial.suggest_int("EState_VSA_Series[1-11]_ind", 0, 1) == 1:
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
    
    # 32. VSA_EState_Series[1-10] - VSA EState (10 individual descriptors)
    if trial.suggest_int("VSA_EState_Series[1-10]", 0, 1) == 1:
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


# === CHUNK 4 ===

import numpy as np
import gc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Import functions from part 1 (will be merged)

def process_rdmol_descriptors(trial, fps, mols, selected_descriptors, save_res="np"):
    """Process rdMolDescriptors (3 descriptors)"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        gc.collect()
    
    # 33. MQNs - Molecular Quantum Numbers
    if trial.suggest_int("MQNs", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.MQNs_(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'MQNs', save_res)
        selected_descriptors.append("MQNs")
        clear_descriptor_memory(descriptor)
        
    # 34. AUTOCORR2D - 2D Autocorrelation
    if trial.suggest_int("AUTOCORR2D", 0, 1) == 1:
        descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR2D(alpha) for alpha in mols]
        fps = generating_newfps(fps, descriptor, 'AUTOCORR2D', save_res)
        selected_descriptors.append("AUTOCORR2D")
        clear_descriptor_memory(descriptor)
        
    # 35. BCUT2D - 2D BCUT Descriptors
    if trial.suggest_int("BCUT2D", 0, 1) == 1:
        descriptor = compute_descriptors_parallel(mols)
        fps = generating_newfps(fps, descriptor, 'BCUT2D', save_res)
        selected_descriptors.append("BCUT2D")
        clear_descriptor_memory(descriptor)
    
    return fps, selected_descriptors

def process_3d_descriptors(trial, fps, mols, selected_descriptors, save_res="np", mols_3d=None):
    """Process 3D descriptors (15 descriptors including FractionCSP3) with conformer generation"""
    
    def clear_descriptor_memory(descriptor):
        del descriptor
        gc.collect()
    
    def safe_compute_descriptor(mols_list, descriptor_func, descriptor_name):
        """Safely compute descriptor with error handling and size validation"""
        try:
            descriptor = []
            for mol in mols_list:
                if mol is None:
                    descriptor.append(0.0)
                else:
                    try:
                        val = descriptor_func(mol)
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            descriptor.append(val)
                        else:
                            descriptor.append(0.0)
                    except Exception as e:
                        print(f"Error occurred in {descriptor_name}: {e}")
                        descriptor.append(0.0)
            
            return descriptor
        except Exception as e:
            print(f"Error occurred in {descriptor_name}: {e}")
            # Return zeros for failed descriptor
            return [0.0] * len(mols_list)
    
    def compute_3d_descriptor_with_conformers(mols_list, descriptor_func, descriptor_name):
        """Safely compute 3D descriptors with conformer generation"""
        try:
            # Generate 3D conformers first
            mols_3d = process_molecules_parallel(mols_list, max_workers=4)
            
            if not mols_3d:
                print(f"No 3D conformers generated for {descriptor_name}")
                return [0.0] * len(mols_list)
                
            # Compute descriptors
            descriptor = []
            for mol in mols_3d:
                if mol is None:
                    descriptor.append(0.0)
                else:
                    try:
                        val = descriptor_func(mol)
                        if val is not None:
                            # Handle numpy arrays properly
                            if isinstance(val, np.ndarray):
                                # For arrays, check if any value is NaN or inf using np.any()
                                if val.size == 1:
                                    # Single value array
                                    if not np.isnan(val.item()) and not np.isinf(val.item()):
                                        descriptor.append(val.item())
                                    else:
                                        descriptor.append(0.0)
                                else:
                                    # Multi-dimensional array - flatten and take mean
                                    flattened = val.flatten()
                                    if flattened.size > 0:
                                        # Remove NaN and Inf values before taking mean
                                        valid_values = flattened[~np.isnan(flattened) & ~np.isinf(flattened)]
                                        if len(valid_values) > 0:
                                            descriptor.append(float(np.mean(valid_values)))
                                        else:
                                            descriptor.append(0.0)
                                    else:
                                        descriptor.append(0.0)
                            else:
                                # For scalar values
                                if not np.isnan(val) and not np.isinf(val):
                                    descriptor.append(val)
                                else:
                                    descriptor.append(0.0)
                        else:
                            descriptor.append(0.0)
                    except Exception as e:
                        print(f"Error computing {descriptor_name}: {e}")
                        descriptor.append(0.0)
            
            return descriptor
            
        except Exception as e:
            print(f"Error computing {descriptor_name}: {e}")
            return [0.0] * len(mols_list)
    
    # Use pre-generated 3D conformers if available, otherwise generate them
    if mols_3d is not None:
        mols2 = mols_3d
        # Ensure mols2 has the same length as fps
        if len(mols2) != fps.shape[0]:
            print(f"Warning: 3D conformers count ({len(mols2)}) doesn't match fps shape ({fps.shape[0]})")
            # Pad with None values to match fps length
            while len(mols2) < fps.shape[0]:
                mols2.append(None)
            mols2 = mols2[:fps.shape[0]]
    else:
        mols2 = process_molecules_parallel(mols, max_workers=8)
        del mols  # Free memory
        gc.collect()
        # Ensure mols2 has the same length as fps
        if len(mols2) != fps.shape[0]:
            print(f"Warning: 3D conformers count ({len(mols2)}) doesn't match fps shape ({fps.shape[0]})")
            # Pad with None values to match fps length
            while len(mols2) < fps.shape[0]:
                mols2.append(None)
            mols2 = mols2[:fps.shape[0]]
    
    if mols2:
        # 36. Asphericity - Asphericity
        if trial.suggest_int("Asphericity", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcAsphericity, "Asphericity")
            fps = generating_newfps(fps, descriptor, 'Asphericity', save_res)
            selected_descriptors.append("Asphericity")
            clear_descriptor_memory(descriptor)
            
        # 37. PBF - Principal Moments of Inertia B
        if trial.suggest_int("PBF", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcPBF, "PBF")
            fps = generating_newfps(fps, descriptor, 'PBF', save_res)
            selected_descriptors.append("PBF")
            clear_descriptor_memory(descriptor)
            
        # 38. RadiusOfGyration - Radius of Gyration
        if trial.suggest_int("RadiusOfGyration", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcRadiusOfGyration, "RadiusOfGyration")
            fps = generating_newfps(fps, descriptor, 'RadiusOfGyration', save_res)
            selected_descriptors.append("RadiusOfGyration")
            clear_descriptor_memory(descriptor)
            
        # 39. InertialShapeFactor - Inertial Shape Factor
        if trial.suggest_int("InertialShapeFactor", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcInertialShapeFactor, "InertialShapeFactor")
            fps = generating_newfps(fps, descriptor, 'InertialShapeFactor', save_res)
            selected_descriptors.append("InertialShapeFactor")
            clear_descriptor_memory(descriptor)
            
        # 40. Eccentricity - Eccentricity
        if trial.suggest_int("Eccentricity", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcEccentricity, "Eccentricity")
            fps = generating_newfps(fps, descriptor, 'Eccentricity', save_res)
            selected_descriptors.append("Eccentricity")
            clear_descriptor_memory(descriptor)
            
        # 41. SpherocityIndex - Spherocity Index
        if trial.suggest_int("SpherocityIndex", 0, 1) == 1:
            descriptor = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcSpherocityIndex, "SpherocityIndex")
            fps = generating_newfps(fps, descriptor, 'SpherocityIndex', save_res)
            selected_descriptors.append("SpherocityIndex")
            clear_descriptor_memory(descriptor)
            
        # 42. PMI_series[1-3]_ind - Principal Moments of Inertia (3 descriptors)
        if trial.suggest_int("PMI_series[1-3]_ind", 0, 1) == 1:
            d1 = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcPMI1, "PMI1")
            d2 = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcPMI2, "PMI2")
            d3 = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcPMI3, "PMI3")
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
        if trial.suggest_int("NPR_series[1-2]_ind", 0, 1) == 1:
            d1 = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcNPR1, "NPR1")
            d2 = compute_3d_descriptor_with_conformers(mols, Chem.rdMolDescriptors.CalcNPR2, "NPR2")
            d1 = np.asarray(d1)
            d2 = np.asarray(d2)
            fps = generating_newfps(fps, [d1,d2], 'NPR_series[1-2]_ind', save_res)
            selected_descriptors.append("NPR_series[1-2]_ind")
            clear_descriptor_memory(d1)
            clear_descriptor_memory(d2)
            
        ####################################################
        # Generate 3D conformers for complex 3D descriptors
        mols2 = process_molecules_parallel(mols, max_workers=8)
        ####################################################
            
        # 44. AUTOCORR3D - 3D Autocorrelation
        if trial.suggest_int("AUTOCORR3D", 0, 1) == 1:
            descriptor = [Chem.rdMolDescriptors.CalcAUTOCORR3D(mols) for mols in mols2]
            fps = generating_newfps(fps, descriptor, 'AUTOCORR3D', save_res)
            selected_descriptors.append("AUTOCORR3D")
            clear_descriptor_memory(descriptor)
            
        # 45. RDF - Radial Distribution Function
        if trial.suggest_int("RDF", 0, 1) == 1:
            descriptor = [Chem.rdMolDescriptors.CalcRDF(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'RDF', save_res)
            selected_descriptors.append("RDF")
            clear_descriptor_memory(descriptor)
            
        # 46. MORSE - Morse Descriptors
        if trial.suggest_int("MORSE", 0, 1) == 1:
            descriptor = [Chem.rdMolDescriptors.CalcMORSE(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'MORSE', save_res)
            selected_descriptors.append("MORSE")
            clear_descriptor_memory(descriptor)
            
        # 47. WHIM - WHIM Descriptors
        if trial.suggest_int("WHIM", 0, 1) == 1:
            descriptor = [Chem.rdMolDescriptors.CalcWHIM(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'WHIM', save_res)
            selected_descriptors.append("WHIM")
            clear_descriptor_memory(descriptor)
            
        # 48. GETAWAY - GETAWAY Descriptors
        if trial.suggest_int("GETAWAY", 0, 1) == 1:
            descriptor = [Chem.rdMolDescriptors.CalcGETAWAY(mols) for mols in mols2]
            descriptor = Normalization(descriptor)
            fps = generating_newfps(fps, descriptor, 'GETAWAY', save_res)
            selected_descriptors.append("GETAWAY")
            clear_descriptor_memory(descriptor)
    else:
        print("Warning: No 3D conformers generated. 3D descriptors will be skipped.")
    
    return fps, selected_descriptors 


# === CHUNK 5 ===

# Import statements are already included in the main file

def search_data_descriptor_compress_fixed(trial, fps, mols, name, target_path="result", save_res="np", mols_3d=None):
    """
    Main function for Optuna-driven chemical descriptor selection
    Based on the original feature_search.py with proper VSA implementation and 3D conformer generation
    
    Args:
        trial: Optuna trial object
        fps: Input fingerprints (Morgan + MACCS + Avalon = "all")
        mols: List of RDKit mol objects
        name: Name for saving results
        target_path: Path to save results
        save_res: Save format ("np" or "pd")
        mols_3d: Pre-generated 3D conformers (optional)
    
    Returns:
        fps_updated: Updated fingerprints with selected descriptors
        selected_descriptors: List of selected descriptor names
        excluded_descriptors: List of excluded descriptor names (empty for now)
    """
    selected_descriptors = []
    excluded_descriptors = []
    
    # Process 2D descriptors (27 descriptors)
    fps, selected_descriptors = process_2d_descriptors(trial, fps, mols, selected_descriptors, save_res)
    
    # Process VSA descriptors (5 series, 57 individual descriptors)
    fps, selected_descriptors = process_vsa_descriptors(trial, fps, mols, selected_descriptors, save_res)
    
    # Process rdMolDescriptors (3 descriptors)
    fps, selected_descriptors = process_rdmol_descriptors(trial, fps, mols, selected_descriptors, save_res)
    
    # Process 3D descriptors (15 descriptors including FractionCSP3)
    fps, selected_descriptors = process_3d_descriptors(trial, fps, mols, selected_descriptors, save_res, mols_3d)
    
    # Limit to 49 descriptors if more are selected
    if len(selected_descriptors) > 49:
        selected_descriptors = selected_descriptors[:49]
    
    # Save results if requested
    if save_res == "pd":
        fps.to_csv(f'{target_path}/{name}_feature_selection.csv')
    
    # Ensure final data type
    fps = fps.astype('float')
    
    return fps, selected_descriptors, excluded_descriptors

# Main execution block for testing
if __name__ == "__main__":
    print("Chemical Descriptor Selection Module - Part 5")
    print("This module provides the main integration function for descriptor selection.")
    print("To use this module, import search_data_descriptor_compress_fixed function.") 

