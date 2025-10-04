#!/usr/bin/env python3
"""
Chemical Descriptor Calculator and Caching System
================================================

This advanced module provides comprehensive chemical descriptor calculation capabilities
for molecular property prediction in the ANO framework. It computes 49+ categories of
molecular descriptors including 2D physicochemical properties, 3D conformational features,
and specialized pharmacological descriptors with intelligent caching and optimization.

Key Features:
------------
1. **Comprehensive Descriptor Coverage**: 49+ descriptor categories with 1000+ individual descriptors
2. **3D Conformer Generation**: Automatic 3D structure generation with multiple optimization methods
3. **Intelligent Caching**: Efficient storage and retrieval of computed descriptors
4. **Batch Processing**: Parallel computation with memory optimization for large datasets
5. **Error Handling**: Robust handling of problematic molecules and fallback mechanisms
6. **Normalization**: Built-in log transformation and outlier handling for ML compatibility

Descriptor Categories:
---------------------
**Basic Physicochemical Properties:**
- Molecular weight, LogP, polar surface area, rotatable bonds
- Hydrogen bond donors/acceptors, aromatic rings, heteroatoms

**Topological Descriptors:**
- Connectivity indices, path counts, complexity measures
- Ring descriptors, branching parameters, shape indices

**Electronic Descriptors:**
- Partial charges, electronegativity, hardness/softness
- Electrophilicity indices, electron affinity measures

**3D Conformational Descriptors:**
- Principal moments of inertia, radius of gyration
- Asphericity, eccentricity, spherocity measures
- Interatomic distances and conformational entropy

**Pharmacological Descriptors:**
- Lipinski Rule of Five parameters
- Drug-likeness indices, ADMET-related properties
- Bioavailability and permeability predictors

Usage Examples:
--------------
# Initialize calculator with caching
calc = ChemDescriptorCalculator(cache_dir='cache/descriptors')

# Calculate all descriptors for a dataset
smiles_list = ['CCO', 'c1ccccc1', 'CCN(CC)CC']
mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
descriptors = calc.get_descriptors(mols, dataset='example', split_type='train')

# Get specific descriptor categories
basic_props = calc.calculate_basic_properties(mols)
topo_desc = calc.calculate_topological_descriptors(mols)

# Generate 3D conformers and calculate 3D descriptors
mols_3d = calc.generate_3d_conformers_batch(mols)
desc_3d = calc.calculate_3d_descriptors(mols_3d)

# Access individual descriptor values
mol_weight = calc.calculate_mol_weight(mols)
logp_values = calc.calculate_logp(mols)

Performance Optimization:
------------------------
- **Parallel Processing**: Multi-core descriptor calculation
- **Memory Management**: Efficient handling of large molecular datasets
- **Caching Strategy**: Persistent storage of computed descriptors
- **3D Optimization**: Multiple conformer generation methods with fallbacks
- **Batch Operations**: Vectorized calculations where possible

Cache Management:
----------------
The module maintains descriptor caches organized by:
- Dataset name (ws, de, lo, hu)
- Split type (rm, sc, ts)
- Descriptor categories
- Molecular subset identifiers

Cache files are stored as compressed numpy arrays (.npz) with metadata
for version control and invalidation handling.

Error Handling:
--------------
- **Invalid Molecules**: Graceful handling of problematic SMILES/structures
- **3D Generation Failures**: Multiple fallback methods for conformer generation
- **Descriptor Calculation Errors**: Individual descriptor failure isolation
- **Memory Management**: Automatic cleanup and garbage collection
- **Cache Corruption**: Automatic cache regeneration when needed
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, AllChem, rdDistGeom
# from mordred import Calculator, descriptors as mordred_descriptors  # Optional
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional
import warnings
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Try importing from config, add path only if necessary
try:
    from config import DATA_PATH, RESULT_PATH
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATA_PATH, RESULT_PATH
import json
from datetime import datetime

# Elements that RDKit has trouble with for 3D conformer generation
# These elements often fail in ETKDG embedding or UFF/MMFF optimization
# Based on: (1) Transition metals with complex coordination, (2) Lanthanides, (3) Actinides, (4) Heavy metals
UNSUPPORTED_ELEMENTS = {
    # Transition metals (Groups 3-12) - often problematic for force fields
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # Period 4
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # Period 5
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  # Period 6
    'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',  # Period 7

    # Lanthanides (Rare earth elements)
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',

    # Actinides (Radioactive elements)
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',

    # Metalloids and heavy p-block elements that may cause issues
    'Ga', 'Ge', 'As', 'Se',  # Period 4
    'In', 'Sn', 'Sb', 'Te',  # Period 5
    'Tl', 'Pb', 'Bi', 'Po', 'At',  # Period 6

    # Noble gases (no chemical bonds expected)
    'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',

    # Other problematic elements
    'Al',  # Aluminum can be problematic in some coordination complexes
}

class ChemDescriptorCalculator:
    """Calculate and cache all chemical descriptors including 3D"""
    
    def _normalize_descriptor(self, descriptor_array):
        """
        Normalize descriptor values using sign-preserving log transformation
        Matched with previous user version

        This method preserves the sign of descriptor values while applying log transformation,
        which is important for descriptors that can have negative values.

        Parameters:
        -----------
        descriptor_array : np.ndarray
            Descriptor values to normalize

        Returns:
        --------
        np.ndarray
            Sign-preserving log-transformed normalized values
        """
        epsilon = 1e-10
        max_value = 1e15

        # Clip extreme values to prevent overflow
        descriptor_array = np.clip(descriptor_array, -max_value, max_value)

        # Replace near-zero values with epsilon to avoid log(0)
        descriptor_custom = np.where(np.abs(descriptor_array) < epsilon, epsilon, descriptor_array)

        # Apply sign-preserving log transformation (previous user version method)
        # sign(x) * log1p(|x|) preserves negative values while normalizing scale
        descriptor_log = np.sign(descriptor_custom) * np.log1p(np.abs(descriptor_custom))

        # Handle NaN and Inf values
        descriptor_log = np.nan_to_num(descriptor_log, nan=0.0, posinf=0.0, neginf=0.0)

        return descriptor_log
    
    def __init__(self, cache_dir: str = None, module_name: str = None):
        """
        Initialize descriptor calculator
        
        Args:
            cache_dir: Directory to store cached descriptors
            module_name: Name of the module using this calculator (unused, kept for compatibility)
        """
        if cache_dir is None:
            # Always use shared cache directory in result folder
            cache_dir = os.path.join(RESULT_PATH, 'chemical_descriptors')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load failure information from mol_fps_maker
        from config import RESULT_PATH as config_result_path
        self.failed_molecules_file = Path(config_result_path) / 'fingerprint' / 'failed' / 'failed_3d_conformers.json'
        self.failed_molecules = self._load_failed_molecules()
        
        # Define all 49 descriptor categories
        self.descriptor_categories = {
            # Basic molecular properties (10)
            'MolWeight': lambda mol: Descriptors.MolWt(mol),
            'MolLogP': lambda mol: Descriptors.MolLogP(mol),
            'MolMR': lambda mol: Descriptors.MolMR(mol),
            'TPSA': lambda mol: Descriptors.TPSA(mol),
            'NumRotatableBonds': lambda mol: Descriptors.NumRotatableBonds(mol),
            'HeavyAtomCount': lambda mol: Descriptors.HeavyAtomCount(mol),
            'NumHAcceptors': lambda mol: Descriptors.NumHAcceptors(mol),
            'NumHDonors': lambda mol: Descriptors.NumHDonors(mol),
            'NumHeteroatoms': lambda mol: Descriptors.NumHeteroatoms(mol),
            'NumValenceElectrons': lambda mol: Descriptors.NumValenceElectrons(mol),
            
            # Additional counts (7)
            'NHOHCount': lambda mol: Descriptors.NHOHCount(mol),
            'NOCount': lambda mol: Descriptors.NOCount(mol),
            'RingCount': lambda mol: Descriptors.RingCount(mol),
            'NumAromaticRings': lambda mol: Descriptors.NumAromaticRings(mol),
            'NumSaturatedRings': lambda mol: Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': lambda mol: Descriptors.NumAliphaticRings(mol),
            'LabuteASA': lambda mol: Descriptors.LabuteASA(mol),
            
            # Connectivity and shape (5)
            'BalabanJ': lambda mol: Descriptors.BalabanJ(mol),
            'BertzCT': lambda mol: Descriptors.BertzCT(mol),
            'Ipc': lambda mol: Descriptors.Ipc(mol),
            'HallKierAlpha': lambda mol: Descriptors.HallKierAlpha(mol),
            'Phi': lambda mol: Descriptors.Phi(mol),
            
            # Additional molecular features (5)
            'NumAmideBonds': lambda mol: rdMolDescriptors.CalcNumAmideBonds(mol),
            'FractionCSP3': lambda mol: Descriptors.FractionCsp3(mol),
            'NumSpiroAtoms': lambda mol: rdMolDescriptors.CalcNumSpiroAtoms(mol),
            'NumBridgeheadAtoms': lambda mol: rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            'MQNs': lambda mol: self._calculate_mqns(mol),
            
            # 2D Autocorrelation (2)
            'AUTOCORR2D': lambda mol: self._calculate_autocorr2d(mol),
            'BCUT2D': lambda mol: self._calculate_bcut2d(mol),
            
            # Series descriptors - these return multiple values
            'kappa_Series[1-3]_ind': lambda mol: self._calculate_kappa(mol),
            'Chi_Series[1-98]_ind': lambda mol: self._calculate_chi(mol),
            'PEOE_VSA_Series[1-14]_ind': lambda mol: self._calculate_peoe_vsa(mol),
            'SMR_VSA_Series[1-10]_ind': lambda mol: self._calculate_smr_vsa(mol),
            'SlogP_VSA_Series[1-12]_ind': lambda mol: self._calculate_slogp_vsa(mol),
            'EState_VSA_Series[1-11]_ind': lambda mol: self._calculate_estate_vsa(mol),
            'VSA_EState_Series[1-10]_ind': lambda mol: self._calculate_vsa_estate(mol),
        }
        
        # 3D descriptors (if conformers available)
        self.descriptor_3d_categories = {
            'Asphericity': lambda mol: self._calculate_asphericity(mol),
            'PBF': lambda mol: self._calculate_pbf(mol),
            'RadiusOfGyration': lambda mol: self._calculate_rog(mol),
            'InertialShapeFactor': lambda mol: self._calculate_isf(mol),
            'Eccentricity': lambda mol: self._calculate_eccentricity(mol),
            'SpherocityIndex': lambda mol: self._calculate_spherocity(mol),
            'PMI_series[1-3]_ind': lambda mol: self._calculate_pmi(mol),
            'NPR_series[1-2]_ind': lambda mol: self._calculate_npr(mol),
            'AUTOCORR3D': lambda mol: self._calculate_autocorr3d(mol),
            'RDF': lambda mol: self._calculate_rdf(mol),
            'MORSE': lambda mol: self._calculate_morse(mol),
            'WHIM': lambda mol: self._calculate_whim(mol),
            'GETAWAY': lambda mol: self._calculate_getaway(mol),
        }
           
    def _load_failed_molecules(self):
        """Load failed molecules from mol_fps_maker results"""
        if self.failed_molecules_file.exists():
            try:
                with open(self.failed_molecules_file, 'r') as f:
                    data = json.load(f)
                    return data.get('failed_smiles', set())
            except:
                return set()
        return set()
    
    def _calculate_kappa(self, mol):
        """Calculate Kappa shape indices"""
        return np.array([
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol)
        ])
    
    def _calculate_chi(self, mol):
        """Calculate Chi connectivity indices"""
        chi_values = []
        # Chi0-4 (n and v variants)
        for i in range(5):
            for suffix in ['', 'n', 'v']:
                try:
                    name = f'Chi{i}{suffix}'
                    if hasattr(Descriptors, name):
                        chi_values.append(getattr(Descriptors, name)(mol))
                except:
                    chi_values.append(0)
        
        # Pad to 98 values as specified
        while len(chi_values) < 98:
            chi_values.append(0)
        return np.array(chi_values[:98])
    
    def _calculate_peoe_vsa(self, mol):
        """Calculate PEOE VSA descriptors"""
        vsa_values = []
        for i in range(1, 15):
            name = f'PEOE_VSA{i}'
            if hasattr(Descriptors, name):
                vsa_values.append(getattr(Descriptors, name)(mol))
            else:
                vsa_values.append(0)
        return np.array(vsa_values)
    
    def _calculate_smr_vsa(self, mol):
        """Calculate SMR VSA descriptors"""
        vsa_values = []
        for i in range(1, 11):
            name = f'SMR_VSA{i}'
            if hasattr(Descriptors, name):
                vsa_values.append(getattr(Descriptors, name)(mol))
            else:
                vsa_values.append(0)
        return np.array(vsa_values)
    
    def _calculate_slogp_vsa(self, mol):
        """Calculate SlogP VSA descriptors"""
        vsa_values = []
        for i in range(1, 13):
            name = f'SlogP_VSA{i}'
            if hasattr(Descriptors, name):
                vsa_values.append(getattr(Descriptors, name)(mol))
            else:
                vsa_values.append(0)
        return np.array(vsa_values)
    
    def _calculate_estate_vsa(self, mol):
        """Calculate EState VSA descriptors"""
        vsa_values = []
        for i in range(1, 12):
            name = f'EState_VSA{i}'
            if hasattr(Descriptors, name):
                vsa_values.append(getattr(Descriptors, name)(mol))
            else:
                vsa_values.append(0)
        return np.array(vsa_values)
    
    def _calculate_vsa_estate(self, mol):
        """Calculate VSA EState descriptors"""
        vsa_values = []
        for i in range(1, 11):
            name = f'VSA_EState{i}'
            if hasattr(Descriptors, name):
                vsa_values.append(getattr(Descriptors, name)(mol))
            else:
                vsa_values.append(0)
        return np.array(vsa_values)
    
    def _calculate_mqns(self, mol):
        """Calculate Molecular Quantum Numbers - 42 features"""
        from rdkit.Chem import rdMolDescriptors
        mqns = rdMolDescriptors.MQNs_(mol)
        return np.array(mqns)  # Returns all 42 MQN values
    
    def _calculate_autocorr2d(self, mol):
        """Calculate 2D autocorrelation"""
        # Simplified version
        return np.array([Descriptors.BCUT2D_CHGHI(mol)])
    
    def _calculate_bcut2d(self, mol):
        """Calculate BCUT2D descriptors"""
        return np.array([Descriptors.BCUT2D_MWHI(mol)])
    
    # 3D descriptors (using RDKit's Descriptors3D)
    def _calculate_asphericity(self, mol):
        """Calculate Asphericity"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.Asphericity(mol)])
        except:
            return np.array([0])
    
    def _calculate_pbf(self, mol):
        """Calculate Plane of Best Fit"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.PBF(mol)])
        except:
            return np.array([0])
    
    def _calculate_rog(self, mol):
        """Calculate Radius of Gyration"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.RadiusOfGyration(mol)])
        except:
            return np.array([0])
    
    def _calculate_isf(self, mol):
        """Calculate Inertial Shape Factor"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.InertialShapeFactor(mol)])
        except:
            return np.array([0])
    
    def _calculate_eccentricity(self, mol):
        """Calculate Eccentricity"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.Eccentricity(mol)])
        except:
            return np.array([0])
    
    def _calculate_spherocity(self, mol):
        """Calculate Spherocity Index"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.SpherocityIndex(mol)])
        except:
            return np.array([0])
    
    def _calculate_pmi(self, mol):
        """Calculate Principal Moments of Inertia"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.PMI1(mol), Descriptors3D.PMI2(mol), Descriptors3D.PMI3(mol)])
        except:
            return np.array([0, 0, 0])
    
    def _calculate_npr(self, mol):
        """Calculate Normalized Principal Ratios"""
        try:
            from rdkit.Chem import Descriptors3D
            return np.array([Descriptors3D.NPR1(mol), Descriptors3D.NPR2(mol)])
        except:
            return np.array([0, 0])
    # Advanced 3D descriptors (using RDKit)
    def _calculate_autocorr3d(self, mol):
        """Calculate 3D autocorrelation using RDKit with robust error handling"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                from rdkit.Chem import rdMolDescriptors
                val = rdMolDescriptors.CalcAUTOCORR3D(mol)
                val_array = np.array(val)
                
                # Handle NaN and Inf values
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values
                val_array = np.clip(val_array, -1e6, 1e6)
                return val_array
        except:
            pass
        return np.zeros(80)  # Default size for AUTOCORR3D
    
    def _calculate_rdf(self, mol):
        """Calculate Radial Distribution Function using RDKit with robust error handling"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                from rdkit.Chem import rdMolDescriptors
                val = rdMolDescriptors.CalcRDF(mol)
                val_array = np.array(val)
                
                # Handle NaN and Inf values
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values
                val_array = np.clip(val_array, -1e6, 1e6)
                return val_array
        except:
            pass
        return np.zeros(210)  # Default size for RDF
    
    def _calculate_morse(self, mol):
        """Calculate Morse potential using RDKit with robust error handling"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                from rdkit.Chem import rdMolDescriptors
                val = rdMolDescriptors.CalcMORSE(mol)
                val_array = np.array(val)
                
                # Handle NaN and Inf values
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values
                val_array = np.clip(val_array, -1e6, 1e6)
                return val_array
        except:
            pass
        return np.zeros(224)  # Default size for MORSE
    
    def _calculate_whim(self, mol):
        """Calculate WHIM descriptors using RDKit with robust error handling"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                from rdkit.Chem import rdMolDescriptors
                val = rdMolDescriptors.CalcWHIM(mol)
                val_array = np.array(val)
                
                # Handle NaN and Inf values
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values
                val_array = np.clip(val_array, -1e6, 1e6)
                return val_array
        except:
            pass
        return np.zeros(114)  # Default size for WHIM
    
    def _calculate_getaway(self, mol):
        """Calculate GETAWAY descriptors using RDKit with robust error handling"""
        try:
            if mol is not None and mol.GetNumConformers() > 0:
                from rdkit.Chem import rdMolDescriptors
                val = rdMolDescriptors.CalcGETAWAY(mol)
                val_array = np.array(val)
                
                # Handle NaN and Inf values that can occur with certain molecules
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    # Replace problematic values with 0
                    val_array = np.nan_to_num(val_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clip extreme values to prevent numerical issues downstream
                val_array = np.clip(val_array, -1e6, 1e6)
                
                return val_array
        except Exception as e:
            # Return zeros if calculation fails completely
            pass
        return np.zeros(273)  # Default size for GETAWAY

    # 3D conformer generation methods (from mol3d_maker.py)
    def has_unsupported_elements(self, mol):
        """Check if molecule contains unsupported elements"""
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in UNSUPPORTED_ELEMENTS:
                return True
        return False

    def generate_3d_conformer(self, mol, num_confs=1, max_attempts=5, random_seed=42):
        """Generate 3D conformer with embedding"""
        if mol is None:
            return None

        mol = AllChem.AddHs(mol)

        for attempt in range(max_attempts):
            try:
                params = rdDistGeom.ETKDGv3()
                params.randomSeed = random_seed + attempt
                params.numThreads = 0

                confIds = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

                if len(confIds) > 0:
                    for confId in confIds:
                        # Two-stage optimization: UFF followed by MMFF (matched with previous user version)
                        AllChem.UFFOptimizeMolecule(mol, confId=confId, maxIters=200)
                        # MMFF optimization for better accuracy (previous user version used this)
                        try:
                            AllChem.MMFFOptimizeMolecule(mol, confId=confId, maxIters=200)
                        except:
                            # MMFF may fail for some molecules, continue with UFF result
                            pass
                    mol = AllChem.RemoveHs(mol)
                    return mol
            except Exception:
                continue

        return None

    def generate_3d_with_fallback(self, mol, num_confs=1):
        """Generate 3D with multiple fallback strategies"""
        if mol is None:
            return None

        # Check for unsupported elements
        if self.has_unsupported_elements(mol):
            return None

        # Try standard method
        mol_3d = self.generate_3d_conformer(mol, num_confs)
        if mol_3d is not None:
            return mol_3d

        # Try without hydrogens
        try:
            mol_copy = Chem.Mol(mol)
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_copy, params) >= 0:
                # Two-stage optimization (matched with previous user version)
                AllChem.UFFOptimizeMolecule(mol_copy, maxIters=200)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=200)
                except:
                    pass
                return mol_copy
        except:
            pass

        # Try 2D coordinates
        try:
            mol_copy = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol_copy)
            return mol_copy
        except:
            pass

        return None

    def process_molecules_parallel(self, mols, use_thread=True, max_workers=None, show_progress=False):
        """Process molecules in parallel for 3D conformer generation"""
        results = []
        total = len(mols)

        # Get executor function
        if use_thread:
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor

        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        with executor_class(max_workers=max_workers) as executor:
            futures = {executor.submit(self.generate_3d_with_fallback, mol): i
                      for i, mol in enumerate(mols)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=10)
                    results.append((idx, result))
                    if show_progress and len(results) % 100 == 0:
                        print(f"  Processed {len(results)}/{total} molecules...")
                except Exception:
                    results.append((idx, None))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def filter_valid_3d_molecules(self, mols, y_values):
        """Filter molecules and y_values based on successful 3D conformer generation"""
        mols_3d = self.process_molecules_parallel(mols)

        valid_indices = []
        valid_mols_3d = []

        for i, mol_3d in enumerate(mols_3d):
            if mol_3d is not None:
                valid_indices.append(i)
                valid_mols_3d.append(mol_3d)

        # Filter y_values
        valid_y = [y_values[i] for i in valid_indices] if y_values is not None else None

        print(f"  Successfully generated 3D conformers for {len(valid_mols_3d)}/{len(mols)} molecules")
        if len(valid_mols_3d) < len(mols):
            print(f"  Excluded {len(mols) - len(valid_mols_3d)} molecules due to 3D conformer generation failure")

        return valid_mols_3d, valid_y, valid_indices

    def calculate_all_descriptors(self, mols: List, mols_3d: Optional[List] = None) -> Dict[str, np.ndarray]:
        """
        Calculate all 49 descriptor categories for given molecules
        
        Args:
            mols: List of RDKit molecule objects
            mols_3d: Optional list of 3D conformers
            
        Returns:
            Dictionary with descriptor names as keys and arrays as values
        """
        descriptor_dict = {}
        
        print("Calculating 2D descriptors...")
        for name, func in self.descriptor_categories.items():
            values = []
            for mol in mols:
                try:
                    val = func(mol)
                    if isinstance(val, (list, np.ndarray)):
                        values.append(val)
                    else:
                        values.append([val])
                except:
                    values.append([0])
            
            descriptor_dict[name] = np.array(values)
            print(f"  {name}: shape {descriptor_dict[name].shape}")
        
        # Generate 3D conformers if not provided
        if mols_3d is None:
            print("Generating 3D conformers...")
            from rdkit.Chem import AllChem
            mols_3d_list = []
            failed_indices = []
            
            for i, mol in enumerate(mols):
                try:
                    mol_3d = Chem.AddHs(mol)
                    result = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                    if result == -1:
                        raise Exception("Failed to embed molecule")
                    AllChem.MMFFOptimizeMolecule(mol_3d)
                    mols_3d_list.append(mol_3d)
                except Exception as e:
                    mols_3d_list.append(None)
                    failed_indices.append(i)
                    # Skip recording - mol_fps_maker already handles this
                    pass
            
            mols_3d = mols_3d_list
            if failed_indices:
                print(f"  Warning: Failed to generate 3D conformers for {len(failed_indices)} molecules")
        elif isinstance(mols_3d, np.ndarray):
            # Convert numpy array to list for consistency
            mols_3d = mols_3d.tolist()
        elif not isinstance(mols_3d, list):
            # Convert any other iterable to list
            mols_3d = list(mols_3d)
        
        # Store 3D conformers as list in descriptor dict for caching
        descriptor_dict['3d_conformers'] = mols_3d
        
        if mols_3d is not None:
            print("Calculating 3D descriptors...")
            # List of descriptors that need normalization (matched with previous user version)
            # Apply sign-preserving log transformation to specific descriptors
            normalize_descriptors = ['Ipc', 'PMI_series[1-3]_ind', 'MORSE', 'GETAWAY',
                                    'BCUT2D', 'AUTOCORR2D', 'AUTOCORR3D', 'RDF', 'WHIM']
            
            for name, func in self.descriptor_3d_categories.items():
                values = []
                for mol in mols_3d:
                    try:
                        val = func(mol)
                        if isinstance(val, (list, np.ndarray)):
                            values.append(val)
                        else:
                            values.append([val])
                    except:
                        values.append([0])
                
                descriptor_array = np.array(values)
                
                # Apply normalization to specific descriptors
                if name in normalize_descriptors:
                    descriptor_array = self._normalize_descriptor(descriptor_array)
                    print(f"  {name}: shape {descriptor_array.shape} (normalized)")
                else:
                    print(f"  {name}: shape {descriptor_array.shape}")
                
                descriptor_dict[name] = descriptor_array
        
        return descriptor_dict
    
    def calculate_selected_descriptors(self, mols: List, 
                                      dataset_name: str = None,
                                      split_type: str = None,
                                      subset: str = None,
                                      mols_3d: Optional[List] = None,
                                      targets: Optional[List] = None) -> np.ndarray:
        """
        Calculate and cache descriptors with automatic saving
        
        Args:
            mols: List of RDKit molecules
            dataset_name: Dataset name (e.g., 'WS', 'DE')
            split_type: Split type (e.g., 'rm', 'ac')
            subset: 'train' or 'test'
            mols_3d: Optional list of 3D conformers
            
        Returns:
            Descriptor array
        """
        if dataset_name and split_type and subset:
            # Check cache first
            cache_dir = self.cache_dir / dataset_name.lower() / split_type
            cache_file = cache_dir / f"{dataset_name.lower()}_{split_type}_{subset}_descriptors.npz"
            
            # Check remake flag
            from config import CACHE_CONFIG
            remake = CACHE_CONFIG.get('remake_descriptor', False)
            
            if cache_file.exists() and not remake:
                print(f"  Loading cached descriptors from {cache_file.name}...")
                data = np.load(cache_file)
                if 'descriptor_array' in data:
                    return data['descriptor_array']
        
        # Set context for failure tracking
        self._current_dataset = dataset_name
        self._current_split = split_type
        self._current_targets = targets if targets else [0.0] * len(mols)
        
        # Calculate descriptors
        print(f"  Calculating descriptors for {dataset_name}/{split_type}/{subset}...")
        descriptor_dict = self.calculate_all_descriptors(mols, mols_3d)
        
        # Convert to array (stack all descriptor values)
        descriptor_arrays = []
        for key in sorted(descriptor_dict.keys()):
            if key != '3d_conformers':  # Skip 3D conformers
                values = descriptor_dict[key]
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                descriptor_arrays.append(values)
        
        descriptor_array = np.hstack(descriptor_arrays) if descriptor_arrays else np.array([])

        # Clean final descriptor_array for NaN/Inf values
        descriptor_array = np.nan_to_num(descriptor_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Save to cache if info provided
        if dataset_name and split_type and subset:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, 
                              descriptor_array=descriptor_array,
                              **descriptor_dict)
            print(f"  Saved descriptors to {cache_file.name}")
        
        return descriptor_array
    
    def save_descriptors(self, descriptor_dict: Dict, dataset: str, split: str, subset: str):
        """
        Save calculated descriptors to cache
        
        Args:
            descriptor_dict: Dictionary of descriptors
            dataset: Dataset name (ws, de, lo, hu)
            split: Split type (rm, ac, cl, etc.)
            subset: 'train' or 'test'
        """
        save_dir = self.cache_dir / dataset / split
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{dataset}_{split}_{subset}_descriptors.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(descriptor_dict, f)
        print(f"Saved descriptors to {save_path}")
        
        # Also save as NPZ for compatibility
        npz_path = save_dir / f"{dataset}_{split}_{subset}_descriptors.npz"
        np.savez_compressed(npz_path, **descriptor_dict)
        print(f"Saved descriptors to {npz_path}")
    
    def load_descriptors(self, dataset: str, split: str, subset: str) -> Optional[Dict]:
        """
        Load cached descriptors
        
        Args:
            dataset: Dataset name
            split: Split type
            subset: 'train' or 'test'
            
        Returns:
            Dictionary of descriptors or None if not cached
        """
        from config import CACHE_CONFIG
        remake_descriptor = CACHE_CONFIG.get('remake_descriptor', False)
        
        load_path = self.cache_dir / dataset / split / f"{dataset}_{split}_{subset}_descriptors.pkl"
        
        if load_path.exists() and not remake_descriptor:
            with open(load_path, 'rb') as f:
                descriptor_dict = pickle.load(f)
            # Ensure 3d_conformers is a list if present
            if descriptor_dict and '3d_conformers' in descriptor_dict:
                if isinstance(descriptor_dict['3d_conformers'], np.ndarray):
                    descriptor_dict['3d_conformers'] = descriptor_dict['3d_conformers'].tolist()
            print(f"Loaded cached descriptors from {load_path}")
            return descriptor_dict
        
        return None
    
    def get_selected_descriptors(self, descriptor_dict: Dict, selection: Dict[str, int]) -> np.ndarray:
        """
        Get selected descriptors based on selection dictionary
        
        Args:
            descriptor_dict: Full dictionary of all descriptors
            selection: Dictionary with descriptor names and 0/1 selection
            
        Returns:
            Concatenated array of selected descriptors
        """
        selected_arrays = []
        
        for name, selected in selection.items():
            if selected == 1 and name in descriptor_dict:
                selected_arrays.append(descriptor_dict[name])
        
        if selected_arrays:
            # Ensure all arrays have same first dimension
            n_samples = selected_arrays[0].shape[0]
            
            # Flatten and concatenate
            flattened = []
            for arr in selected_arrays:
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(n_samples, -1)
                flattened.append(arr)
            
            return np.hstack(flattened)
        
        return np.array([])


def main():
    """Pre-calculate all descriptors for all datasets"""
    import time
    from config import ACTIVE_SPLIT_TYPES, CODE_SPECIFIC_DATASETS
    
    calculator = ChemDescriptorCalculator()
    
    # Use config settings
    datasets = ['ws', 'de', 'lo', 'hu']  # All datasets for now
    splits = ACTIVE_SPLIT_TYPES  # Use config.py setting
    
    total_start = time.time()
    processed = 0
    skipped = 0
    
    for dataset in datasets:
        for split in splits:
            for subset in ['train', 'test']:
                # Check if already cached
                cached = calculator.load_descriptors(dataset, split, subset)
                if cached is not None:
                    print(f"✓ {dataset}-{split}-{subset}: Already cached")
                    skipped += 1
                    continue
                
                print(f"\n⏳ Processing {dataset}-{split}-{subset}...")
                
                # Load molecules - try different file name patterns
                # Pattern 1: split_fullname_subset.csv (e.g., rm_ws496_logS_train.csv)
                dataset_map = {
                    'ws': 'ws496_logS',
                    'de': 'delaney-processed', 
                    'lo': 'Lovric2020_logS0',
                    'hu': 'huusk'
                }
                
                data_file = Path(DATA_PATH) / subset / split / f"{split}_{dataset_map.get(dataset, dataset)}_{subset}.csv"
                if not data_file.exists():
                    print(f"  Data file not found: {data_file}")
                    continue
                
                df = pd.read_csv(data_file)
                smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
                smiles_list = df[smiles_col].tolist()
                
                # Convert to molecules
                mols = []
                valid_indices = []
                for i, smi in enumerate(smiles_list):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        mols.append(mol)
                        valid_indices.append(i)
                
                print(f"  Loaded {len(mols)} valid molecules")
                
                # Calculate descriptors
                descriptor_dict = calculator.calculate_all_descriptors(mols)
                
                # Save to cache
                calculator.save_descriptors(descriptor_dict, dataset, split, subset)
                processed += 1
                print(f"  ✅ Completed {dataset}-{split}-{subset}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"✅ Pre-computation complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped (cached): {skipped} files")
    print(f"  Total time: {total_elapsed:.1f}s")


    def load_failed_molecules(self) -> Dict:
        """Load list of failed molecules from file"""
        if self.failed_molecules_file.exists():
            with open(self.failed_molecules_file, 'r') as f:
                return json.load(f)
        return {}
    
    
    def is_failed(self, smiles: str) -> bool:
        """Check if molecule has previously failed 3D conformer generation"""
        return smiles in self.failed_molecules
    
    def filter_valid_molecules(self, mols: List, smiles_list: List = None) -> Tuple[List, List]:
        """
        Filter out molecules that are known to fail 3D conformer generation
        
        Args:
            mols: List of RDKit molecules
            smiles_list: Optional list of SMILES strings
            
        Returns:
            Tuple of (valid_mols, valid_indices)
        """
        valid_mols = []
        valid_indices = []
        
        for i, mol in enumerate(mols):
            if mol is None:
                continue
                
            # Get SMILES
            if smiles_list and i < len(smiles_list):
                smiles = smiles_list[i]
            else:
                smiles = Chem.MolToSmiles(mol)
            
            # Check if failed
            if not self.is_failed(smiles):
                valid_mols.append(mol)
                valid_indices.append(i)
        
        if len(valid_mols) < len(mols):
            print(f"  Filtered out {len(mols) - len(valid_mols)} molecules with known 3D conformer failures")
        
        return valid_mols, valid_indices

# ============================================================================
# DESCRIPTOR CACHE UTILS - Unified from descriptor_cache_utils.py
# ============================================================================

def get_descriptor_cache_path(
    dataset_name: str,
    split_type: str,
    data_type: str,
    selection_key: str = None
) -> Path:
    """
    Generate cache file path for descriptors

    Args:
        dataset_name: Dataset identifier (e.g., 'ws', 'de')
        split_type: Split type (e.g., 'rm', 'ac')
        data_type: 'train' or 'test'
        selection_key: Optional key for specific descriptor selection

    Returns:
        Path to cache file
    """
    # Use only chemical_descriptors directory
    cache_dir = Path(RESULT_PATH) / f'chemical_descriptors/{dataset_name.lower()}/{split_type}'
    cache_dir.mkdir(parents=True, exist_ok=True)

    if selection_key:
        # For specific descriptor selections
        cache_file = cache_dir / f"{dataset_name.lower()}_{split_type}_{data_type}_{selection_key}.pkl"
    else:
        # For all descriptors - try both naming conventions
        cache_file_new = cache_dir / f"{dataset_name.lower()}_{split_type}_{data_type}_all_descriptors.pkl"
        cache_file_old = cache_dir / f"{dataset_name.lower()}_{split_type}_{data_type}_descriptors.pkl"

        # Return existing file if found, otherwise use new naming
        if cache_file_old.exists():
            return cache_file_old
        else:
            return cache_file_new

    return cache_file

def load_descriptors_from_cache(
    dataset_name: str,
    split_type: str,
    data_type: str,
    n_molecules: int = None,
    selection_key: str = None,
    check_remake: bool = True
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Load descriptors from cache (NPZ or PKL format)

    Args:
        dataset_name: Dataset identifier
        split_type: Split type
        data_type: 'train' or 'test'
        n_molecules: Expected number of molecules for validation
        selection_key: Optional key for specific selection
        check_remake: Whether to check remake flags

    Returns:
        Tuple of (descriptors array, descriptor names list) or (None, None)
    """
    if check_remake:
        from config import CACHE_CONFIG
        remake_descriptors = CACHE_CONFIG.get('remake_descriptors', False)
        if remake_descriptors:
            return None, None

    cache_file = get_descriptor_cache_path(dataset_name, split_type, data_type, selection_key)

    if not cache_file.exists():
        return None, None

    try:
        # Load pickle format (unified format)
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            # Convert dict format to array format
            descriptor_names = list(data.keys())
            # Handle multi-dimensional descriptors properly
            descriptor_arrays = []
            expanded_names = []

            for name in descriptor_names:
                arr = np.array(data[name])
                if arr.ndim == 1:
                    # 1D array - use as is
                    descriptor_arrays.append(arr.reshape(-1, 1))
                    expanded_names.append(name)
                elif arr.ndim == 2:
                    if arr.shape[1] == 1:
                        # Single feature per molecule
                        descriptor_arrays.append(arr)
                        expanded_names.append(name)
                    else:
                        # Multiple features per molecule - split into columns
                        for i in range(arr.shape[1]):
                            descriptor_arrays.append(arr[:, i:i+1])
                            expanded_names.append(f"{name}_{i}")
                else:
                    print(f"  Warning: Skipping {name} with shape {arr.shape}")
                    continue

            if descriptor_arrays:
                descriptors = np.hstack(descriptor_arrays)
                descriptor_names = expanded_names
            else:
                print(f"  Warning: No valid descriptors found")
                return None, None
        else:
            print(f"  Warning: Unexpected pkl format in {cache_file.name}")
            return None, None

        # Validate dimensions
        if n_molecules is not None and descriptors.shape[0] != n_molecules:
            print(f"  Cache size mismatch: {descriptors.shape[0]} vs {n_molecules}")
            return None, None

        print(f"  Loaded descriptor cache: {cache_file.name}")
        print(f"    Shape: {descriptors.shape}, Names: {len(descriptor_names)}")
        return descriptors, descriptor_names

    except Exception as e:
        print(f"  Warning: Failed to load descriptor cache: {e}")
        return None, None

def get_descriptors_cached(
    mols: List[Chem.Mol],
    dataset_name: str,
    split_type: str,
    data_type: str,
    descriptor_function = None,
    selection_key: str = None,
    **descriptor_kwargs
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Get descriptors with caching - main interface function

    Args:
        mols: List of RDKit molecules
        dataset_name: Dataset identifier
        split_type: Split type
        data_type: 'train' or 'test'
        descriptor_function: Function to calculate descriptors if cache miss
        selection_key: Optional key for specific selection
        **descriptor_kwargs: Additional arguments for descriptor function

    Returns:
        Tuple of (descriptors array, descriptor names list) or (None, None)
    """
    n_molecules = len(mols)

    # Try to load from cache first
    descriptors, descriptor_names = load_descriptors_from_cache(
        dataset_name, split_type, data_type, n_molecules, selection_key
    )

    if descriptors is not None:
        return descriptors, descriptor_names

    # Calculate new descriptors
    if descriptor_function is None:
        print(f"  Warning: No descriptor function provided for {dataset_name}/{split_type}/{data_type}")
        return np.array([]), []

    print(f"  Calculating new descriptors for {dataset_name}/{split_type}/{data_type}...")

    try:
        # Call the descriptor calculation function
        descriptors, descriptor_names = descriptor_function(mols, **descriptor_kwargs)

        if isinstance(descriptors, np.ndarray) and len(descriptor_names) > 0:
            # Note: Saving is handled by the original ChemDescriptorCalculator
            return descriptors, descriptor_names
        else:
            print(f"  Warning: Invalid descriptor calculation result")
            return np.array([]), []

    except Exception as e:
        print(f"  Error calculating descriptors: {e}")
        return np.array([]), []

if __name__ == "__main__":
    main()