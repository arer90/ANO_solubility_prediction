"""
QSAR Feature Calculator Module - STABLE VERSION
No numba, with stable concurrent processing
"""
import types
from functools import lru_cache
import gc
from typing import Dict, List, Tuple, Optional
import numpy as np
import psutil
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import pickle
import os

from .base import RDKIT_AVAILABLE
from .config import CHUNK_SIZE, DESCRIPTOR_NAMES, DEFAULT_PARAMS

# RDKit imports with proper error handling
if RDKIT_AVAILABLE:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors


def process_molecule_batch(smiles_batch: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Process batch of molecules with caching - stable version"""
    if RDKIT_AVAILABLE:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        
    results = []
    descriptor_names = [
        'MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumRotatableBonds', 'NumValenceElectrons', 'NumAromaticRings',
        'NumSaturatedRings', 'NumAliphaticRings', 'RingCount', 'TPSA', 'LabuteASA',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha',
        'Kappa1', 'Kappa2', 'Kappa3', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3',
        'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3',
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'MaxEStateIndex', 'MinEStateIndex',
        'MaxAbsEStateIndex', 'MinAbsEStateIndex'
    ]
    
    for smiles in smiles_batch:
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Descriptor calculation
                    desc_vals = np.zeros(49, dtype=np.float32)
                    for i, desc_name in enumerate(descriptor_names[:49]):
                        try:
                            func = getattr(Descriptors, desc_name, None)
                            if func:
                                val = func(mol)
                                desc_vals[i] = float(val) if val is not None and not np.isnan(val) else 0.0
                        except:
                            desc_vals[i] = 0.0
                    
                    # Fingerprint calculation
                    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=2048)
                    fp_arr = np.zeros(2048, dtype=np.uint8)
                    nonzero_elements = fp.GetNonzeroElements()
                    for idx in nonzero_elements.keys():
                        if idx < 2048:
                            fp_arr[idx] = 1
                    
                    results.append((desc_vals, fp_arr))
                else:
                    results.append((np.zeros(49, dtype=np.float32), np.zeros(2048, dtype=np.uint8)))
            else:
                results.append((np.zeros(49, dtype=np.float32), np.zeros(2048, dtype=np.uint8)))
                
        except Exception as e:
            logging.error(f"Error processing molecule {smiles}: {e}")
            results.append((np.zeros(49, dtype=np.float32), np.zeros(2048, dtype=np.uint8)))
        
    return results


class OptimizedFeatureCalculator:
    """Stable feature calculator with optional concurrent processing"""
    
    def __init__(self, performance_mode: bool = True, n_jobs: int = -1, 
                 parallel_timeout: int = None, cache_enabled: bool = True):
        """Initialize with optimization parameters"""
        self.performance_mode = performance_mode
        self.chunk_size = DEFAULT_PARAMS.get('chunk_size', 100)
        self.cache_enabled = DEFAULT_PARAMS.get('cache_enabled', cache_enabled)
        self.rdkit_available = RDKIT_AVAILABLE
        
        # Parallel processing setup
        if n_jobs == -1:
            self.n_jobs = max(1, mp.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)
            
        self.parallel_timeout = parallel_timeout or DEFAULT_PARAMS.get('parallel_timeout', 60)
        
        # Cache initialization
        self._cache = {} if self.cache_enabled else None
        self._cache_file = '.feature_cache.pkl' if self.cache_enabled else None
        
        if self.cache_enabled:
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_enabled and self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"    Loaded feature cache with {len(self._cache)} entries")
            except:
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        if self.cache_enabled and self._cache_file and self._cache:
            try:
                with open(self._cache_file, 'wb') as f:
                    pickle.dump(self._cache, f)
            except:
                pass
    
    def calculate_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """Calculate features with flexible processing modes"""
        print(f"  Calculating features for {len(smiles_list)} molecules...")
        
        # Sampling for performance mode
        max_molecules = DEFAULT_PARAMS.get('max_samples_analysis', 50000)
        if self.performance_mode and len(smiles_list) > max_molecules:
            print(f"    Performance mode sampling: {max_molecules}/{len(smiles_list)}")
            sampled_indices = np.random.choice(len(smiles_list), max_molecules, replace=False)
            sampled_smiles = [smiles_list[i] for i in sampled_indices]
        else:
            sampled_smiles = smiles_list
            sampled_indices = None
        
        # Choose processing method
        if len(sampled_smiles) < 500 or self.performance_mode:
            # Sequential for small datasets or performance mode
            descriptors, fingerprints = self._calculate_sequential_fast(sampled_smiles)
        else:
            # Try parallel for larger datasets
            try:
                descriptors, fingerprints = self._calculate_parallel_safe(sampled_smiles)
            except Exception as e:
                print(f"    ⚠️ Parallel processing failed: {str(e)}")
                print("    Falling back to sequential processing...")
                descriptors, fingerprints = self._calculate_sequential_fast(sampled_smiles)
        
        # Map back to full data if sampled
        if sampled_indices is not None:
            full_descriptors = np.zeros((len(smiles_list), 49), dtype=np.float32)
            full_fingerprints = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
            
            full_descriptors[sampled_indices] = descriptors
            full_fingerprints[sampled_indices] = fingerprints
            
            # Fill non-sampled with mean values
            mean_desc = np.mean(descriptors, axis=0)
            non_sampled = np.setdiff1d(np.arange(len(smiles_list)), sampled_indices)
            full_descriptors[non_sampled] = mean_desc
            
            descriptors = full_descriptors
            fingerprints = full_fingerprints
        
        # Combine features
        features = np.hstack([descriptors, fingerprints]).astype(np.float32)
        
        # Save cache
        if self.cache_enabled:
            self._save_cache()
        
        return {
            'features': features,
            'descriptors': descriptors,
            'fingerprints': fingerprints
        }
    
    def _calculate_sequential_fast(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Fast sequential calculation"""
        n = len(smiles_list)
        descriptors = np.zeros((n, 49), dtype=np.float32)
        fingerprints = np.zeros((n, 2048), dtype=np.uint8)
        
        if self.rdkit_available:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            
            # Simple descriptors for fast calculation
            simple_descriptors = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 
                                'NumRotatableBonds', 'TPSA', 'NumAromaticRings']
            
            for i, smi in enumerate(smiles_list):
                if i % 100 == 0 and i > 0:
                    print(f"        Processed {i}/{n} molecules")
                
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        # Fast descriptor calculation
                        for j, desc_name in enumerate(simple_descriptors):
                            func = getattr(Descriptors, desc_name, None)
                            if func:
                                try:
                                    val = func(mol)
                                    descriptors[i, j] = float(val) if val is not None else 0.0
                                except:
                                    descriptors[i, j] = 0.0
                        
                        # Fast fingerprint
                        try:
                            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                            arr = np.zeros(512, dtype=np.uint8)
                            DataStructs.ConvertToNumpyArray(fp, arr)
                            fingerprints[i, :512] = arr
                        except:
                            pass
                except:
                    pass
        else:
            # Simulation without RDKit
            np.random.seed(42)
            descriptors = np.random.randn(n, 49).astype(np.float32)
            fingerprints = np.random.randint(0, 2, (n, 2048), dtype=np.uint8)
        
        return descriptors, fingerprints
    
    def _calculate_parallel_safe(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Safe parallel calculation with fallback"""
        try:
            # Check cached results
            cached_results = self._get_cached_results(smiles_list)
            
            # Process uncached molecules
            if cached_results:
                uncached_indices = np.where(~cached_results['mask'])[0]
                uncached_smiles = [smiles_list[i] for i in uncached_indices]
            else:
                uncached_smiles = smiles_list
                uncached_indices = np.arange(len(smiles_list))
            
            if not uncached_smiles:
                return cached_results['descriptors'], cached_results['fingerprints']
            
            # Parallel processing
            return self._calculate_parallel_optimized(smiles_list, cached_results)
            
        except Exception as e:
            print(f"    ⚠️ Parallel processing failed: {str(e)}")
            # Fallback to sequential
            return self._calculate_sequential_fast(smiles_list)
    
    def _get_cached_results(self, smiles_list: List[str]) -> Optional[Dict]:
        """Get cached results for SMILES"""
        if not self._cache:
            return None
        
        n = len(smiles_list)
        descriptors = np.zeros((n, 49), dtype=np.float32)
        fingerprints = np.zeros((n, 2048), dtype=np.uint8)
        mask = np.zeros(n, dtype=bool)
        
        for i, smi in enumerate(smiles_list):
            if smi in self._cache:
                desc, fp = self._cache[smi]
                descriptors[i] = desc
                fingerprints[i] = fp
                mask[i] = True
        
        return {
            'descriptors': descriptors,
            'fingerprints': fingerprints,
            'mask': mask
        }
    
    def _calculate_parallel_optimized(self, smiles_list: List[str], 
                                    cached_results: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized parallel processing"""
        n = len(smiles_list)
        
        # Initialize with cached results or zeros
        if cached_results:
            descriptors = cached_results['descriptors']
            fingerprints = cached_results['fingerprints']
            mask = cached_results['mask']
            
            # Only process uncached molecules
            uncached_indices = np.where(~mask)[0]
            uncached_smiles = [smiles_list[i] for i in uncached_indices]
            
            if not uncached_smiles:
                return descriptors, fingerprints
        else:
            descriptors = np.zeros((n, 49), dtype=np.float32)
            fingerprints = np.zeros((n, 2048), dtype=np.uint8)
            uncached_smiles = smiles_list
            uncached_indices = np.arange(n)
        
        print(f"    Processing {len(uncached_smiles)} uncached molecules...")
        
        # Dynamic batch sizing
        memory_per_mol = 0.01  # MB estimate
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        max_batch_size = int(available_memory_mb / (self.n_jobs * memory_per_mol))
        batch_size = min(self.chunk_size, max_batch_size, len(uncached_smiles) // self.n_jobs)
        batch_size = max(10, batch_size)
        
        # Create batches
        batches = []
        for i in range(0, len(uncached_smiles), batch_size):
            batch = uncached_smiles[i:i+batch_size]
            batch_indices = uncached_indices[i:i+batch_size]
            batches.append((batch, batch_indices))
        
        print(f"    Processing {len(uncached_smiles)} molecules in {len(batches)} batches")
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_molecule_batch, batch): (batch, indices)
                    for batch, indices in batches
                }
                
                # Process results
                for future in as_completed(future_to_batch, timeout=self.parallel_timeout):
                    batch, indices = future_to_batch[future]
                    try:
                        results = future.result(timeout=10)
                        
                        # Store results
                        for j, (desc, fp) in enumerate(results):
                            idx = indices[j]
                            descriptors[idx] = desc
                            fingerprints[idx] = fp
                            
                            # Update cache
                            if self.cache_enabled:
                                original_smi = smiles_list[idx]
                                self._cache[original_smi] = (desc, fp)
                        
                    except Exception as e:
                        print(f"      Warning: Batch processing failed: {str(e)}")
                        # Process failed batch sequentially
                        for j, smi in enumerate(batch):
                            idx = indices[j]
                            result = process_molecule_batch([smi])[0]
                            descriptors[idx] = result[0]
                            fingerprints[idx] = result[1]
        
        except Exception as e:
            print(f"    Parallel processing failed: {str(e)}")
            # Final fallback to sequential
            return self._calculate_sequential_fast(smiles_list)
        
        return descriptors, fingerprints
    
    def _calculate_tanimoto_similarity(self, train_smiles: List[str], 
                                     test_smiles: List[str]) -> Dict:
        """Calculate Tanimoto similarity using RDKit"""
        if not RDKIT_AVAILABLE:
            return self._calculate_similarity_fallback(train_smiles, test_smiles)
        
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            
            # Calculate fingerprints
            train_fps = []
            test_fps = []
            
            # Process training set
            for smi in train_smiles:
                try:
                    mol = Chem.MolFromSmiles(smi) if smi else None
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        train_fps.append(fp)
                except:
                    continue
            
            # Process test set
            for smi in test_smiles:
                try:
                    mol = Chem.MolFromSmiles(smi) if smi else None
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        test_fps.append(fp)
                except:
                    continue
            
            if not train_fps or not test_fps:
                return self._calculate_similarity_fallback(train_smiles, test_smiles)
            
            # Calculate similarities
            similarities = []
            for test_fp in test_fps:
                max_sim = 0.0
                for train_fp in train_fps:
                    sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
                    max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            
            similarities = np.array(similarities)
            
            # Statistics
            stats = {
                'mean': float(np.mean(similarities)),
                'median': float(np.median(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'std': float(np.std(similarities))
            }
            
            # Quality assessment
            mean_sim = stats['mean']
            if mean_sim < 0.2:
                quality = 'Excellent'
            elif mean_sim < 0.4:
                quality = 'Good'
            elif mean_sim < 0.6:
                quality = 'Acceptable'
            elif mean_sim < 0.75:
                quality = 'Risky'
            else:
                quality = 'Dangerous'
            
            return {
                'tanimoto': {
                    'stats': stats,
                    'quality': quality,
                    'values': similarities.tolist()
                }
            }
            
        except Exception as e:
            print(f"RDKit similarity calculation error: {str(e)}")
            return self._calculate_similarity_fallback(train_smiles, test_smiles)
    
    def calculate_similarity(self, train_smiles: List[str], 
                           test_smiles: List[str]) -> Dict:
        """Calculate Tanimoto similarity - main method"""
        try:
            if not self.rdkit_available:
                return self._calculate_similarity_fallback(train_smiles, test_smiles)
            
            # Use RDKit for actual Tanimoto calculation
            return self._calculate_tanimoto_similarity(train_smiles, test_smiles)
            
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return self._calculate_similarity_fallback(train_smiles, test_smiles)
    
    def _calculate_similarity_fallback(self, train_smiles: List[str], 
                                     test_smiles: List[str]) -> Dict:
        """Fallback similarity calculation without RDKit"""
        try:
            # Simple Jaccard similarity based implementation
            similarities = []
            for test_smi in test_smiles:
                max_sim = 0.0
                test_set = set(test_smi) if test_smi else set()
                for train_smi in train_smiles:
                    train_set = set(train_smi) if train_smi else set()
                    if len(test_set | train_set) > 0:
                        sim = len(test_set & train_set) / len(test_set | train_set)
                        max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            
            similarities = np.array(similarities)
            
            stats = {
                'mean': float(np.mean(similarities)) if len(similarities) > 0 else 0.0,
                'median': float(np.median(similarities)) if len(similarities) > 0 else 0.0,
                'min': float(np.min(similarities)) if len(similarities) > 0 else 0.0,
                'max': float(np.max(similarities)) if len(similarities) > 0 else 0.0,
                'std': float(np.std(similarities)) if len(similarities) > 0 else 0.0
            }
            
            return {
                'tanimoto': {
                    'stats': stats,
                    'quality': 'Unknown',
                    'values': similarities.tolist()
                }
            }
        except Exception as e:
            print(f"Fallback similarity calculation failed: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear all caches"""
        if self._cache:
            self._cache.clear()
        if self._cache_file and os.path.exists(self._cache_file):
            os.remove(self._cache_file)
        gc.collect()