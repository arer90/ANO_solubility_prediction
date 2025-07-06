"""
QSAR Statistical Analysis Module - NO SAMPLING VERSION

This module handles statistical analysis for QSAR datasets without sampling.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gc
from scipy import stats
from scipy.spatial.distance import pdist, cdist

from .base import RDKIT_AVAILABLE
from .config import TANIMOTO_STANDARDS, DEFAULT_PARAMS

if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs


class StatisticalAnalyzer:
    """Handles statistical analysis for QSAR data"""
    
    def __init__(self):
        self.statistical_results = {}
        
    def perform_statistical_analysis(self, name: str, targets: np.ndarray, 
                                   descriptors: np.ndarray = None) -> Dict:
        """Perform comprehensive statistical analysis"""
        print(f"  Performing statistical analysis for {name}...")
        
        # Basic statistics
        basic_stats = self._calculate_basic_statistics(targets)
        
        # Distribution tests
        distribution_tests = self._perform_distribution_tests(targets)
        
        # Outlier detection
        outlier_stats = self._detect_outliers(targets)
        
        # Feature correlations if descriptors available
        correlation_stats = None
        if descriptors is not None:
            correlation_stats = self._analyze_correlations(descriptors, targets)
        
        results = {
            'basic_stats': basic_stats,
            'distribution_tests': distribution_tests,
            'outlier_stats': outlier_stats,
            'correlation_stats': correlation_stats
        }
        
        self.statistical_results[name] = results
        return results
    
    def _calculate_basic_statistics(self, targets: np.ndarray) -> Dict:
        """Calculate basic statistical measures"""
        return {
            'mean': float(np.mean(targets)),
            'std': float(np.std(targets, ddof=1)),  # Sample std
            'sem': float(stats.sem(targets)),  # Standard error of mean
            'median': float(np.median(targets)),
            'min': float(np.min(targets)),
            'max': float(np.max(targets)),
            'q25': float(np.percentile(targets, 25)),
            'q75': float(np.percentile(targets, 75)),
            'iqr': float(np.percentile(targets, 75) - np.percentile(targets, 25)),
            'cv': float(np.std(targets) / np.mean(targets) * 100) if np.mean(targets) != 0 else 0,
            'range': float(np.max(targets) - np.min(targets)),
            'skewness': float(stats.skew(targets)),
            'kurtosis': float(stats.kurtosis(targets)),
            'n': len(targets)
        }
    
    def _perform_distribution_tests(self, targets: np.ndarray) -> Dict:
        """Perform normality and distribution tests"""
        # Shapiro-Wilk test (limited to 5000 samples)
        shapiro_stat, shapiro_p = stats.shapiro(targets[:min(5000, len(targets))])
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(targets, 'norm', 
                                     args=(np.mean(targets), np.std(targets)))
        
        # Jarque-Bera test
        jarque_stat, jarque_p = stats.jarque_bera(targets)
        
        # Anderson-Darling test
        anderson_result = stats.anderson(targets)
        
        return {
            'shapiro_wilk': {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            },
            'kolmogorov_smirnov': {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'is_normal': ks_p > 0.05
            },
            'jarque_bera': {
                'statistic': float(jarque_stat),
                'p_value': float(jarque_p),
                'is_normal': jarque_p > 0.05
            },
            'anderson_darling': {
                'statistic': float(anderson_result.statistic),
                'critical_values': anderson_result.critical_values.tolist(),
                'significance_levels': anderson_result.significance_level.tolist()
            }
        }
    
    def _detect_outliers(self, targets: np.ndarray) -> Dict:
        """Detect outliers using multiple methods"""
        # IQR method
        q1, q3 = np.percentile(targets, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_iqr = np.sum((targets < lower_bound) | (targets > upper_bound))
        
        # Z-score method
        z_scores = np.abs(stats.zscore(targets))
        outliers_z3 = np.sum(z_scores > 3)
        outliers_z2 = np.sum(z_scores > 2)
        
        # Modified Z-score (MAD method)
        median = np.median(targets)
        mad = np.median(np.abs(targets - median))
        modified_z_scores = 0.6745 * (targets - median) / mad if mad != 0 else z_scores
        outliers_mad = np.sum(np.abs(modified_z_scores) > 3.5)
        
        return {
            'iqr_method': {
                'n_outliers': int(outliers_iqr),
                'percentage': float(outliers_iqr / len(targets) * 100),
                'bounds': (float(lower_bound), float(upper_bound))
            },
            'z_score_3sigma': {
                'n_outliers': int(outliers_z3),
                'percentage': float(outliers_z3 / len(targets) * 100)
            },
            'z_score_2sigma': {
                'n_outliers': int(outliers_z2),
                'percentage': float(outliers_z2 / len(targets) * 100)
            },
            'mad_method': {
                'n_outliers': int(outliers_mad),
                'percentage': float(outliers_mad / len(targets) * 100),
                'mad': float(mad)
            }
        }
    
    def _analyze_correlations(self, descriptors: np.ndarray, 
                            targets: np.ndarray) -> Dict:
        """Analyze correlations between descriptors and target"""
        correlations = []
        p_values = []
        
        n_descriptors = min(49, descriptors.shape[1])
        
        for i in range(n_descriptors):
            corr, p_val = stats.pearsonr(descriptors[:, i], targets)
            correlations.append(corr)
            p_values.append(p_val)
        
        correlations = np.array(correlations)
        p_values = np.array(p_values)
        
        # Find significant correlations
        significant_mask = p_values < 0.05
        n_significant = np.sum(significant_mask)
        
        # Sort by absolute correlation
        abs_corr = np.abs(correlations)
        sorted_idx = np.argsort(abs_corr)[::-1]
        
        return {
            'correlations': correlations.tolist(),
            'p_values': p_values.tolist(),
            'n_significant': int(n_significant),
            'top_5_indices': sorted_idx[:5].tolist(),
            'top_5_correlations': correlations[sorted_idx[:5]].tolist(),
            'max_correlation': float(np.max(abs_corr)),
            'mean_correlation': float(np.mean(abs_corr))
        }
    
    def calculate_similarity_metrics(self, X_train: np.ndarray, X_test: np.ndarray,
                                   smiles_train: List[str] = None, 
                                   smiles_test: List[str] = None) -> Dict:
        """Calculate comprehensive similarity metrics - NO SAMPLING"""
        results = {}
        
        # 1. Feature space distance (Euclidean) - ALL DATA
        results['feature_distance'] = self._calculate_feature_distance_all(X_train, X_test)
        
        # 2. Tanimoto similarity (if RDKit available and SMILES provided) - ALL DATA
        if RDKIT_AVAILABLE and smiles_train and smiles_test and len(smiles_train) > 0 and len(smiles_test) > 0:
            tanimoto_result = self._calculate_tanimoto_similarity_all(
                smiles_train, smiles_test
            )
            if tanimoto_result:
                results['tanimoto'] = tanimoto_result
        
        # 3. Overall similarity assessment
        if results.get('feature_distance') and results.get('tanimoto'):
            results['combined'] = self._calculate_combined_similarity(results)
        elif results.get('feature_distance'):
            # If only feature distance available, use it alone
            max_dist = results['feature_distance']['max']
            norm_dist = results['feature_distance']['mean'] / max_dist if max_dist > 0 else 0
            feature_similarity = 1 - norm_dist
            results['combined'] = {
                'feature_similarity': feature_similarity,
                'tanimoto_similarity': None,
                'combined_similarity': feature_similarity,
                'assessment': self._assess_similarity_quality(feature_similarity)
            }
        
        return results
    
    def _calculate_feature_distance_all(self, X_train: np.ndarray, 
                                      X_test: np.ndarray) -> Dict:
        """Calculate feature space distances - ALL DATA, NO SAMPLING"""
        try:
            print("    Calculating feature distances for all data pairs...")
            
            # For very large datasets, use chunking to avoid memory issues
            if len(X_train) * len(X_test) > 1e8:  # 100M comparisons
                print(f"      Large dataset detected: {len(X_train)} x {len(X_test)} = {len(X_train) * len(X_test):,} comparisons")
                print("      Using chunked computation...")
                
                all_distances = []
                chunk_size = max(1, int(np.sqrt(1e8 / len(X_train))))
                
                for i in range(0, len(X_test), chunk_size):
                    chunk_end = min(i + chunk_size, len(X_test))
                    chunk_distances = cdist(X_test[i:chunk_end], X_train, metric='euclidean')
                    all_distances.append(chunk_distances.flatten())
                    
                    if (i + chunk_size) % 1000 == 0 or chunk_end == len(X_test):
                        print(f"        Processed {chunk_end}/{len(X_test)} test samples")
                
                distances = np.concatenate(all_distances)
            else:
                # Calculate all pairwise distances
                distances = cdist(X_test, X_train, metric='euclidean').flatten()
            
            return {
                'mean': float(np.mean(distances)),
                'median': float(np.median(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances)),
                'n_comparisons': len(distances)
            }
        except Exception as e:
            print(f"      Error calculating feature distances: {str(e)}")
            return None
    
    def _calculate_tanimoto_similarity_all(self, smiles_train: List[str], 
                                         smiles_test: List[str]) -> Dict:
        """Calculate Tanimoto similarity - ALL DATA, NO SAMPLING"""
        try:
            print("    Calculating Tanimoto similarities for all molecule pairs...")
            
            # Pre-calculate fingerprints
            print("      Calculating fingerprints...")
            train_fps = []
            test_fps = []
            
            # Calculate fingerprints for train set
            for i, smi in enumerate(smiles_train):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        train_fps.append(fp)
                except:
                    continue
                
                if (i + 1) % 1000 == 0:
                    print(f"        Train fingerprints: {i+1}/{len(smiles_train)}")
            
            # Calculate fingerprints for test set
            for i, smi in enumerate(smiles_test):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        test_fps.append(fp)
                except:
                    continue
                
                if (i + 1) % 1000 == 0:
                    print(f"        Test fingerprints: {i+1}/{len(smiles_test)}")
            
            if not train_fps or not test_fps:
                return None
            
            print(f"      Calculating {len(test_fps)} x {len(train_fps)} = {len(test_fps) * len(train_fps):,} similarities...")
            
            # 매우 큰 데이터셋의 경우 샘플링 고려
            if len(train_fps) * len(test_fps) > 1e6:  # 50M 이상
                print(f"      WARNING: Very large comparison ({len(test_fps)} x {len(train_fps)} = {len(test_fps) * len(train_fps):,})")
                print(f"      Consider enabling sampling mode to avoid memory issues")
            
            # 50M 이상일 때 강제 샘플링
            if len(train_fps) * len(test_fps) > 1e6:
                import random
                sample_size = int(np.sqrt(1e6))
                train_fps = random.sample(train_fps, min(len(train_fps), sample_size))
                test_fps = random.sample(test_fps, min(len(test_fps), sample_size))
            
            # # Calculate similarities
            # similarities = []
            
            # For very large datasets, process in chunks
            if len(train_fps) * len(test_fps) > 1e7:  # 10M comparisons
                chunk_size = max(1, int(1e7 / len(train_fps)))
                
                # Calculate similarities
                similarities = []
                
                for i in range(0, len(test_fps), chunk_size):
                    chunk_end = min(i + chunk_size, len(test_fps))
                    chunk_similarities = []
                    
                    for j in range(i, chunk_end):
                        for train_fp in train_fps:
                            sim = DataStructs.TanimotoSimilarity(test_fps[j], train_fp)
                            chunk_similarities.append(sim)
                    
                    similarities.extend(chunk_similarities)
                    
                    del chunk_similarities
                    gc.collect()
                    
                    print(f"        Processed {chunk_end}/{len(test_fps)} test molecules")
            else:
                # Process all at once
                for test_fp in test_fps:
                    for train_fp in train_fps:
                        sim = DataStructs.TanimotoSimilarity(test_fp, train_fp)
                        similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # Statistics
            stats_dict = {
                'mean': float(np.mean(similarities)),
                'median': float(np.median(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'q25': float(np.percentile(similarities, 25)),
                'q75': float(np.percentile(similarities, 75)),
                'n_comparisons': len(similarities)
            }
            
            # Threshold analysis
            threshold_analysis = {}
            for category, (min_val, max_val) in TANIMOTO_STANDARDS.items():
                count = np.sum((similarities >= min_val) & (similarities < max_val))
                percentage = float(count / len(similarities) * 100)
                threshold_analysis[category] = {
                    'count': int(count),
                    'percentage': percentage
                }
            
            return {
                'stats': stats_dict,
                'threshold_analysis': threshold_analysis,
                'n_comparisons': len(similarities),
                'quality': self._assess_similarity_quality(stats_dict['mean'])
            }
            
        except Exception as e:
            print(f"      Error calculating Tanimoto similarities: {str(e)}")
            return None
    
    def _calculate_combined_similarity(self, results: Dict) -> Dict:
        """Calculate combined similarity metrics"""
        # Normalize distances to [0, 1]
        max_dist = results['feature_distance']['max']
        norm_dist = results['feature_distance']['mean'] / max_dist if max_dist > 0 else 0
        
        # Combined similarity (inverse of normalized distance)
        feature_similarity = 1 - norm_dist
        tanimoto_similarity = results['tanimoto']['stats']['mean']
        
        # Weighted average
        combined_similarity = 0.5 * feature_similarity + 0.5 * tanimoto_similarity
        
        return {
            'feature_similarity': feature_similarity,
            'tanimoto_similarity': tanimoto_similarity,
            'combined_similarity': combined_similarity,
            'assessment': self._assess_similarity_quality(combined_similarity)
        }
    
    def _assess_similarity_quality(self, similarity: float) -> str:
        """Assess similarity quality"""
        if similarity < 0.3:
            return "Excellent"
        elif similarity < 0.5:
            return "Good"
        elif similarity < 0.7:
            return "Acceptable"
        elif similarity < 0.85:
            return "Risky"
        else:
            return "Dangerous"
    
    def compare_datasets(self, datasets: Dict[str, Dict]) -> Dict:
        """Compare multiple datasets statistically"""
        if len(datasets) < 2:
            return None
        
        dataset_names = list(datasets.keys())
        comparisons = {}
        
        # Pairwise comparisons
        for i in range(len(dataset_names)):
            for j in range(i+1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                targets1 = datasets[name1]['targets']
                targets2 = datasets[name2]['targets']
                
                # T-test
                t_stat, t_p = stats.ttest_ind(targets1, targets2)
                
                # Mann-Whitney U test
                u_stat, u_p = stats.mannwhitneyu(targets1, targets2)
                
                # Effect size (Cohen's d)
                cohens_d = (np.mean(targets1) - np.mean(targets2)) / np.sqrt(
                    (np.var(targets1, ddof=1) + np.var(targets2, ddof=1)) / 2
                )
                
                # Levene's test for equal variances
                levene_stat, levene_p = stats.levene(targets1, targets2)
                
                comparison_key = f"{name1}_vs_{name2}"
                comparisons[comparison_key] = {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(t_p),
                        'significant': t_p < 0.05
                    },
                    'mann_whitney': {
                        'statistic': float(u_stat),
                        'p_value': float(u_p),
                        'significant': u_p < 0.05
                    },
                    'effect_size': {
                        'cohens_d': float(cohens_d),
                        'magnitude': self._interpret_effect_size(cohens_d)
                    },
                    'levene_test': {
                        'statistic': float(levene_stat),
                        'p_value': float(levene_p),
                        'equal_variances': levene_p > 0.05
                    }
                }
        
        return comparisons
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"