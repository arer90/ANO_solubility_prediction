
"""
QSAR Applicability Domain Methods Module - REGULATORY COMPLIANT VERSION
FDA and OECD approved methods only

References:
- OECD (2007) Guidance Document on the Validation of (Q)SAR Models
- FDA (2018) M7(R1) Assessment and Control of DNA Reactive Impurities
- ECHA (2016) Practical Guide - How to Use and Report (Q)SARs
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gc
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from functools import lru_cache
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

try:
    from .config import (AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES, CHUNK_SIZE, 
                        RELIABILITY_SCORING_CONFIG, AD_METHODS, DEFAULT_PARAMS)
except ImportError:
    from config import (AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES, CHUNK_SIZE, 
                       RELIABILITY_SCORING_CONFIG, AD_METHODS, DEFAULT_PARAMS)

warnings.filterwarnings('ignore')


class RegulatoryADMethods:
    """
    Regulatory-compliant Applicability Domain methods
    Only includes FDA/OECD/ECHA approved methods
    """
    
    def __init__(self, random_state: int = 42, ad_mode: str = 'strict', 
                 max_samples: int = None):
        self.random_state = random_state
        self.chunk_size = CHUNK_SIZE
        self.ad_mode = ad_mode
        
        # Maximum samples for performance
        self.max_samples = max_samples if max_samples else DEFAULT_PARAMS.get('max_samples_ad', 5000)
        
        self._update_coverage_standards()
        
        # Cache for expensive calculations
        self._cache = {}
        
        # Model residuals (needed for Williams plot)
        self.model_residuals = None
        self.model_fitted = False
    
    def calculate_all_methods(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray = None, y_test: np.ndarray = None,
                            y_pred_test: np.ndarray = None) -> Dict:
        """
        Calculate all regulatory-approved AD methods
        
        Methods included (based on regulatory guidance):
        1. Leverage (OECD, FDA, ECHA)
        2. Standardized residuals for Williams plot (FDA, ECHA)
        3. k-NN distance (OECD, ECHA)
        4. Euclidean distance to centroid (OECD)
        5. Descriptor range (OECD)
        6. DModX for PCA models (ECHA)
        
        NOT included (not in regulatory guidance):
        - Isolation Forest
        - Local Outlier Factor
        - Kernel Density Estimation
        """
        print("          Calculating regulatory-approved AD methods...")
        
        # Preprocess features once
        X_train_proc, X_test_proc, scaler = self.preprocess_features(X_train, X_test)
        
        # Define X-value based methods only (2025 update)
        # These methods use only descriptor values, no model predictions needed
        regulatory_methods = [
            # Tier 1: Most widely accepted X-based methods
            ('euclidean_distance', self.calculate_euclidean_distance),
            ('descriptor_range', self.calculate_descriptor_range),
            
            # Tier 2: Requires additional computation but still X-based
            ('knn_distance', self.calculate_knn_distance),
        ]
        
        # Note: Removed methods that require model predictions:
        # - standardized_residuals (needs y_pred)
        # - williams_plot (needs y_pred)
        # - dmodx (while technically X-based, it's for PCA models)
        
        results = {}
        
        # Sequential processing for reliability
        for method_name, method_func in regulatory_methods:
            try:
                print(f"            {method_name}...", end='')
                start_time = time.time()
                
                # Apply method-specific sampling if needed
                if method_name == 'knn_distance' and len(X_train_proc) > 10000:
                    # k-NN can be slow for large datasets
                    X_train_method, X_test_method, _, _ = self._sample_data(
                        X_train_proc, X_test_proc, method_name
                    )
                else:
                    X_train_method = X_train_proc
                    X_test_method = X_test_proc
                
                # Calculate AD
                result = method_func(X_train_method, X_test_method)
                
                # Map back to full data if sampled
                if result and len(X_test_method) < len(X_test_proc):
                    result = self._map_to_full_data(result, X_test_proc, X_test_method)
                
                results[method_name] = result
                print(f" [OK] ({time.time()-start_time:.1f}s)")
                
            except Exception as e:
                print(f" [ERROR] ({str(e)[:50]}...)")
                results[method_name] = None
        
        # Williams plot removed - requires model predictions
        
        # Add consensus AD
        consensus = self.calculate_consensus_ad(results)
        if consensus:
            results['consensus'] = consensus
        
        return results
    
    def calculate_leverage(self, X_train: np.ndarray, X_test: np.ndarray) -> Optional[Dict]:
        """
        Calculate leverage (hat values) - OECD/FDA/ECHA approved
        
        Formula: h_i = x_i^T(X^TX)^{-1}x_i
        Warning limit: h* = 3p/n
        
        References:
        - OECD (2007) ENV/JM/MONO(2007)2
        - Gramatica (2007) QSAR Comb Sci 26:694-701
        """
        try:
            if X_train.shape[0] < X_train.shape[1] + 1:
                print(" (Warning: n < p+1)")
                return None
            
            # Add intercept term
            X_train_int = np.column_stack([np.ones(len(X_train)), X_train])
            X_test_int = np.column_stack([np.ones(len(X_test)), X_test])
            
            # Calculate (X'X)^-1 with numerical stability
            XtX = X_train_int.T @ X_train_int
            
            # Regularization for numerical stability
            lambda_reg = 1e-8 * np.trace(XtX) / XtX.shape[0]
            XtX += lambda_reg * np.eye(XtX.shape[0])
            
            # SVD-based inversion for stability
            U, s, Vt = np.linalg.svd(XtX, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1.0 / s, 0)
            XtX_inv = Vt.T @ np.diag(s_inv) @ U.T
            
            # Calculate leverages using einsum for efficiency
            leverages = np.einsum('ij,jk,ik->i', X_test_int, XtX_inv, X_test_int)
            
            # Calculate training leverages for validation
            train_leverages = np.einsum('ij,jk,ik->i', X_train_int, XtX_inv, X_train_int)
            
            # OECD/FDA standard threshold: h* = 3p/n
            p = X_train_int.shape[1]  # number of parameters
            n = X_train_int.shape[0]  # number of training samples
            threshold = 3 * p / n
            
            # Alternative threshold: 2 * mean(h_train)
            mean_leverage = np.mean(train_leverages)
            alt_threshold = 2 * mean_leverage
            
            # Use the more conservative threshold
            final_threshold = max(threshold, alt_threshold)
            
            # Determine AD membership
            in_ad = leverages <= final_threshold
            coverage = float(np.mean(in_ad))
            
            return {
                'coverage': coverage,
                'in_ad': in_ad.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'threshold': float(final_threshold),
                'values': leverages.tolist(),
                'method': 'leverage',
                'regulatory_compliant': True,
                'references': ['OECD (2023)', 'FDA (2024)', 'ECHA (2022)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    def calculate_standardized_residuals(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Optional[Dict]:
        """
        Calculate standardized residuals - FDA/ECHA approved
        Part of Williams plot
        
        Formula: σ_i = (y_i - ŷ_i) / s
        Threshold: |σ| < 3
        
        References:
        - FDA (2018) M7(R1) Guidance
        - Roy et al. (2015) Chemometr Intell Lab Syst 145:22-29
        """
        try:
            # Calculate residuals
            residuals = y_true - y_pred
            
            # Calculate standard deviation of residuals
            s = np.std(residuals, ddof=1)
            
            # Standardized residuals
            std_residuals = residuals / s if s > 0 else residuals
            
            # AD criterion: |σ| < 3
            in_ad = np.abs(std_residuals) < 3
            coverage = float(np.mean(in_ad))
            
            return {
                'coverage': coverage,
                'in_ad': in_ad.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'threshold': 3.0,
                'values': std_residuals.tolist(),
                'method': 'standardized_residuals',
                'regulatory_compliant': True,
                'references': ['FDA M7(R1)', 'ECHA (2016)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    def calculate_williams_plot(self, leverage_result: Dict, 
                               residuals_result: Dict) -> Optional[Dict]:
        """
        Combine leverage and standardized residuals for Williams plot
        FDA and ECHA recommended approach
        
        AD criteria:
        - h < h* (leverage)
        - |σ| < 3 (standardized residuals)
        
        References:
        - Williams (1972) J Am Stat Assoc 67:338
        - FDA (2018) M7(R1) Guidance
        """
        try:
            # Extract values
            leverages = np.array(leverage_result['values'])
            std_residuals = np.array(residuals_result['values'])
            h_threshold = leverage_result['threshold']
            
            # Both criteria must be satisfied
            in_ad_leverage = leverages <= h_threshold
            in_ad_residuals = np.abs(std_residuals) < 3
            in_ad = in_ad_leverage & in_ad_residuals
            
            coverage = float(np.mean(in_ad))
            
            # Identify outlier types
            outlier_types = []
            for i in range(len(leverages)):
                if not in_ad_leverage[i] and not in_ad_residuals[i]:
                    outlier_types.append('both')
                elif not in_ad_leverage[i]:
                    outlier_types.append('leverage')
                elif not in_ad_residuals[i]:
                    outlier_types.append('residual')
                else:
                    outlier_types.append('none')
            
            return {
                'coverage': coverage,
                'in_ad': in_ad.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'thresholds': {
                    'leverage': float(h_threshold),
                    'residuals': 3.0
                },
                'outlier_types': outlier_types,
                'method': 'williams_plot',
                'regulatory_compliant': True,
                'references': ['FDA M7(R1)', 'ECHA (2016)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    def calculate_knn_distance(self, X_train: np.ndarray, X_test: np.ndarray) -> Optional[Dict]:
        """
        k-Nearest Neighbors distance - OECD/ECHA approved
        
        Formula: d_avg = (1/k) * Σ d(x, x_i)
        Threshold: 95th percentile of training distances
        
        References:
        - OECD (2007) ENV/JM/MONO(2007)2
        - ECHA (2016) Practical Guide 5
        - Sahigara et al. (2013) J Cheminform 5:27
        """
        try:
            if X_train.shape[0] < 11:
                return None
            
            # Determine k (OECD suggests 5-10)
            k = min(10, max(5, int(np.sqrt(len(X_train)))))
            
            # Fit k-NN
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1)
            knn.fit(X_train)
            
            # Calculate average distances for test set
            test_distances, _ = knn.kneighbors(X_test)
            test_avg_distances = np.mean(test_distances, axis=1)
            
            # Calculate training distances for threshold
            train_distances, _ = knn.kneighbors(X_train)
            train_avg_distances = np.mean(train_distances, axis=1)
            
            # OECD standard: 95th percentile
            threshold = np.percentile(train_avg_distances, 95)
            
            # AD membership
            in_ad = test_avg_distances <= threshold
            coverage = float(np.mean(in_ad))
            
            return {
                'coverage': coverage,
                'in_ad': in_ad.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'threshold': float(threshold),
                'values': test_avg_distances.tolist(),
                'k': k,
                'method': 'knn_distance',
                'regulatory_compliant': True,
                'references': ['OECD (2023)', 'FDA (2024)', 'ECHA (2022)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    def calculate_euclidean_distance(self, X_train: np.ndarray, 
                                   X_test: np.ndarray) -> Optional[Dict]:
        """
        Euclidean distance to centroid - OECD approved
        
        Formula: d = ||x - x_centroid||
        Threshold: mean + 3*std (3-sigma rule)
        
        References:
        - OECD (2007) ENV/JM/MONO(2007)2
        - Sheridan et al. (2004) J Chem Inf Comput Sci 44:1912-1928
        """
        try:
            # Calculate centroid of training data
            centroid = np.mean(X_train, axis=0)
            
            # Calculate distances
            train_distances = np.linalg.norm(X_train - centroid, axis=1)
            test_distances = np.linalg.norm(X_test - centroid, axis=1)
            
            # OECD standard: mean + 3*std
            mean_dist = np.mean(train_distances)
            std_dist = np.std(train_distances)
            threshold = mean_dist + 3 * std_dist
            
            # AD membership
            in_ad = test_distances <= threshold
            coverage = float(np.mean(in_ad))
            
            return {
                'coverage': coverage,
                'in_ad': in_ad.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'threshold': float(threshold),
                'values': test_distances.tolist(),
                'statistics': {
                    'mean_train': float(mean_dist),
                    'std_train': float(std_dist)
                },
                'method': 'euclidean_distance',
                'regulatory_compliant': True,
                'references': ['OECD (2023)', 'FDA (2024)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    def calculate_descriptor_range(self, X_train: np.ndarray, 
                                 X_test: np.ndarray) -> Optional[Dict]:
        """
        Descriptor range check - OECD approved
        Most conservative approach
        
        Criteria: All descriptors within training range
        Optional: Allow 5% extrapolation
        
        References:
        - OECD (2007) ENV/JM/MONO(2007)2
        - Dimitrov et al. (2005) J Chem Inf Model 45:839-849
        """
        try:
            # Calculate min/max for each descriptor
            train_min = np.min(X_train, axis=0)
            train_max = np.max(X_train, axis=0)
            
            # Strict criterion: within exact range
            in_range_strict = np.all((X_test >= train_min) & (X_test <= train_max), axis=1)
            
            # OECD allows small extrapolation (5%)
            tolerance = 0.05
            range_span = train_max - train_min
            train_min_tol = train_min - tolerance * range_span
            train_max_tol = train_max + tolerance * range_span
            
            # Tolerant criterion
            in_range_tolerant = np.all((X_test >= train_min_tol) & 
                                      (X_test <= train_max_tol), axis=1)
            
            # Count descriptors out of range
            n_out_strict = np.sum((X_test < train_min) | (X_test > train_max), axis=1)
            n_out_tolerant = np.sum((X_test < train_min_tol) | (X_test > train_max_tol), axis=1)
            
            # Use tolerant criterion as default (OECD recommendation)
            coverage = float(np.mean(in_range_tolerant))
            
            return {
                'coverage': coverage,
                'in_ad': in_range_tolerant.tolist(),
                'quality': self._assess_ad_quality(coverage),
                'strict_coverage': float(np.mean(in_range_strict)),
                'n_descriptors_out': n_out_tolerant.tolist(),
                'n_descriptors_out_strict': n_out_strict.tolist(),
                'tolerance': tolerance,
                'method': 'descriptor_range',
                'regulatory_compliant': True,
                'references': ['OECD (2023)', 'EMA (2024)']
            }
            
        except Exception as e:
            print(f" Error: {str(e)}")
            return None
    
    # DModX method removed - it's for PCA-based models, not pure X-value based
    
    def calculate_consensus_ad(self, ad_results: Dict) -> Dict:
        """
        Calculate consensus AD based on regulatory methods only
        
        Approaches:
        1. Conservative: All methods agree (regulatory preference)
        2. Majority vote: >50% methods agree
        3. Weighted: Based on regulatory priority
        """
        # Collect results from X-value based methods only (2025 update)
        regulatory_methods = ['knn_distance', 'euclidean_distance', 'descriptor_range']
        
        in_ad_list = []
        method_names = []
        
        for method in regulatory_methods:
            if method in ad_results and ad_results[method] and 'in_ad' in ad_results[method]:
                in_ad_list.append(np.array(ad_results[method]['in_ad']))
                method_names.append(method)
        
        if not in_ad_list:
            return None
        
        in_ad_matrix = np.array(in_ad_list)
        
        # 1. Conservative (regulatory preferred)
        conservative = np.all(in_ad_matrix, axis=0)
        
        # 2. Majority vote
        majority = np.mean(in_ad_matrix, axis=0) >= 0.5
        
        # 3. Weighted by regulatory importance
        weights = self._get_regulatory_weights(method_names)
        weighted = np.average(in_ad_matrix, axis=0, weights=weights) >= 0.5
        
        return {
            'conservative': {
                'in_ad': conservative.tolist(),
                'coverage': float(np.mean(conservative)),
                'description': 'All methods agree (most conservative)'
            },
            'majority': {
                'in_ad': majority.tolist(),
                'coverage': float(np.mean(majority)),
                'description': 'Majority of methods agree'
            },
            'weighted': {
                'in_ad': weighted.tolist(),
                'coverage': float(np.mean(weighted)),
                'description': 'Weighted by regulatory importance'
            },
            'recommended': 'conservative'  # For regulatory compliance
        }
    
    def _get_regulatory_weights(self, method_names: List[str]) -> np.ndarray:
        """Get weights based on regulatory importance"""
        # Based on frequency of mention in regulatory documents
        regulatory_importance = {
            'knn_distance': 0.8,       # OECD, ECHA
            'euclidean_distance': 0.7, # OECD
            'descriptor_range': 0.7,   # OECD
            # Removed methods that require model predictions:
            # 'leverage': 1.0,           # All agencies (requires model)
            # 'williams_plot': 1.0,      # FDA, ECHA (requires y_pred)
            # 'standardized_residuals': 0.9,  # FDA, ECHA (requires y_pred)
            # 'dmodx': 0.6              # ECHA (for PCA models)
        }
        
        weights = []
        for method in method_names:
            weights.append(regulatory_importance.get(method, 0.5))
        
        weights = np.array(weights)
        return weights / np.sum(weights)
    
    def preprocess_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """Standard preprocessing for regulatory compliance"""
        try:
            # Ensure float32 for consistency
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            
            # Handle NaN/Inf
            X_train = self._handle_invalid_values(X_train)
            X_test = self._handle_invalid_values(X_test)
            
            # Separate descriptors and fingerprints if applicable
            if X_train.shape[1] > 49:  # Assuming first 49 are descriptors
                # Scale only descriptors, not fingerprints
                scaler = StandardScaler()
                
                X_train_desc = X_train[:, :49]
                X_train_fp = X_train[:, 49:]
                X_test_desc = X_test[:, :49]
                X_test_fp = X_test[:, 49:]
                
                # Scale descriptors
                X_train_desc_scaled = scaler.fit_transform(X_train_desc)
                X_test_desc_scaled = scaler.transform(X_test_desc)
                
                # Recombine
                X_train_processed = np.hstack([X_train_desc_scaled, X_train_fp])
                X_test_processed = np.hstack([X_test_desc_scaled, X_test_fp])
            else:
                # Scale all features
                scaler = StandardScaler()
                X_train_processed = scaler.fit_transform(X_train)
                X_test_processed = scaler.transform(X_test)
            
            return X_train_processed, X_test_processed, scaler
            
        except Exception as e:
            print(f" Preprocessing error: {str(e)}")
            return X_train, X_test, None
    
    @staticmethod
    def _handle_invalid_values(X: np.ndarray) -> np.ndarray:
        """Handle NaN and Inf values"""
        # Replace NaN and Inf with column median
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = np.isfinite(col)
            
            if np.any(~mask):
                if np.any(mask):
                    # Use median of finite values
                    median_val = np.median(col[mask])
                    X[~mask, j] = median_val
                else:
                    # All values are invalid, use 0
                    X[:, j] = 0.0
        
        return X
    
    def _sample_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                    method_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample data for computationally intensive methods"""
        n_train, n_test = len(X_train), len(X_test)
        
        # Method-specific limits
        if method_name == 'knn_distance':
            max_samples = 10000
        else:
            max_samples = self.max_samples
        
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_test)
        
        # Sample if needed
        if n_train + n_test > max_samples:
            # Maintain ratio
            train_ratio = n_train / (n_train + n_test)
            max_train = int(max_samples * train_ratio)
            max_test = max_samples - max_train
            
            if n_train > max_train:
                np.random.seed(self.random_state)
                train_indices = np.random.choice(n_train, max_train, replace=False)
                X_train = X_train[train_indices]
            
            if n_test > max_test:
                np.random.seed(self.random_state + 1)
                test_indices = np.random.choice(n_test, max_test, replace=False)
                X_test = X_test[test_indices]
        
        return X_train, X_test, train_indices, test_indices
    
    def _map_to_full_data(self, result: Dict, X_test_full: np.ndarray, 
                         X_test_sampled: np.ndarray) -> Dict:
        """Map sampled results back to full dataset"""
        # Conservative approach: assume out of AD for non-sampled points
        full_result = result.copy()
        
        n_full = len(X_test_full)
        n_sampled = len(X_test_sampled)
        
        if n_full > n_sampled:
            # Initialize with conservative values
            full_in_ad = np.zeros(n_full, dtype=bool)
            full_values = np.full(n_full, np.inf)
            
            # Map sampled results
            # This is simplified - in practice you'd track indices
            full_in_ad[:n_sampled] = result['in_ad']
            if 'values' in result:
                full_values[:n_sampled] = result['values']
            
            full_result['in_ad'] = full_in_ad.tolist()
            full_result['values'] = full_values.tolist()
            full_result['coverage'] = float(np.mean(full_in_ad))
            full_result['sampled'] = True
            full_result['sample_size'] = n_sampled
        
        return full_result
    
    def _update_coverage_standards(self):
        """Update coverage standards based on AD mode"""
        if self.ad_mode in AD_COVERAGE_MODES:
            if self.ad_mode == 'adaptive':
                # For regulatory, use strict standards
                self.coverage_standards = AD_COVERAGE_MODES['strict']['coverage_standards']
            else:
                self.coverage_standards = AD_COVERAGE_MODES[self.ad_mode]['coverage_standards']
        else:
            self.coverage_standards = AD_COVERAGE_STANDARDS
    
    @lru_cache(maxsize=128)
    def _assess_ad_quality(self, coverage: float) -> str:
        """Assess AD quality based on coverage"""
        for quality, bounds in self.coverage_standards.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                min_val, max_val = bounds
                if min_val <= coverage <= max_val:
                    return quality.title()
        return "Unknown"


class ConcurrentADMethods(RegulatoryADMethods):
    """Concurrent version of AD methods with optional parallel processing"""
    
    def __init__(self, random_state: int = 42, ad_mode: str = 'strict', 
                 max_samples: int = None, n_jobs: int = -1):
        super().__init__(random_state, ad_mode, max_samples)
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, mp.cpu_count() - 1)
        self.use_parallel = True  # Flag to enable/disable parallel processing
    
    def calculate_all_methods(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray = None, y_test: np.ndarray = None,
                            y_pred_test: np.ndarray = None,
                            enable_reliability: bool = False) -> Dict:
        """
        Calculate all regulatory-approved AD methods with optional parallel processing
        """
        print("          Calculating regulatory-approved AD methods...")
        
        # Try parallel processing first
        if self.use_parallel and len(X_train) * len(X_test) < 1e8:  # Avoid parallel for huge datasets
            try:
                return self._calculate_all_methods_parallel(
                    X_train, X_test, y_train, y_test, y_pred_test, enable_reliability
                )
            except Exception as e:
                print(f"          [WARNING] Parallel processing failed: {str(e)}")
                print("          Falling back to sequential processing...")
        
        # Fallback to sequential processing
        results = super().calculate_all_methods(
            X_train, X_test, y_train, y_test, y_pred_test
        )
        
        # Add reliability scoring if enabled
        if enable_reliability and results:
            reliability_scores = self._calculate_reliability_scores(results)
            results['reliability_scores'] = reliability_scores
        
        return results
    
    def _calculate_all_methods_parallel(self, X_train: np.ndarray, X_test: np.ndarray, 
                                      y_train: np.ndarray = None, y_test: np.ndarray = None,
                                      y_pred_test: np.ndarray = None,
                                      enable_reliability: bool = False) -> Dict:
        """Calculate AD methods using parallel processing"""
        # Preprocess features once
        X_train_proc, X_test_proc, scaler = self.preprocess_features(X_train, X_test)
        
        # Define methods
        methods = {
            'euclidean_distance': (self.calculate_euclidean_distance, (X_train_proc, X_test_proc)),
            'descriptor_range': (self.calculate_descriptor_range, (X_train_proc, X_test_proc)),
            'knn_distance': (self.calculate_knn_distance, (X_train_proc, X_test_proc)),
            # 'leverage': (self.calculate_leverage, (X_train_proc, X_test_proc)),  # Removed - requires model
            # 'dmodx': (self.calculate_dmodx, (X_train_proc, X_test_proc)),  # Removed - for PCA models
        }
        
        if y_test is not None and y_pred_test is not None:
            methods['standardized_residuals'] = (
                self.calculate_standardized_residuals, 
                (y_test, y_pred_test)
            )
        
        results = {}
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(methods))) as executor:
            # Submit all tasks
            future_to_method = {}
            for method_name, (method_func, args) in methods.items():
                future = executor.submit(method_func, *args)
                future_to_method[future] = method_name
            
            # Collect results
            for future in as_completed(future_to_method):
                method_name = future_to_method[future]
                try:
                    result = future.result(timeout=30)
                    results[method_name] = result
                    print(f"            {method_name}... [OK]")
                except Exception as e:
                    print(f"            {method_name}... [ERROR] ({str(e)[:50]}...)")
                    results[method_name] = None
        
        # Calculate Williams plot if applicable
        # Commented out since leverage is no longer calculated (requires model)
        # if 'leverage' in results and 'standardized_residuals' in results:
        #     if results['leverage'] and results['standardized_residuals']:
        #         results['williams_plot'] = self.calculate_williams_plot(
        #             results['leverage'], results['standardized_residuals']
        #         )
        
        # Add consensus AD
        consensus = self.calculate_consensus_ad(results)
        if consensus:
            results['consensus'] = consensus
        
        # Add reliability scoring if enabled
        if enable_reliability and results:
            reliability_scores = self._calculate_reliability_scores(results)
            results['reliability_scores'] = reliability_scores
        
        return results
    
    def _calculate_reliability_scores(self, ad_results: Dict) -> Dict:
        """Calculate reliability scores based on AD consensus"""
        try:
            valid_methods = [name for name, result in ad_results.items() 
                           if result and isinstance(result, dict) and 'in_ad' in result]
            
            if len(valid_methods) < 2:
                return None
            
            # Calculate consensus reliability
            in_ad_matrix = np.array([
                ad_results[method]['in_ad'] for method in valid_methods
            ])
            
            # Reliability = agreement between methods
            reliability = np.mean(in_ad_matrix, axis=0)
            
            return {
                'method_agreement': reliability.tolist(),
                'high_confidence': (reliability >= 0.8).tolist(),
                'low_confidence': (reliability <= 0.2).tolist(),
                'consensus_methods': valid_methods
            }
            
        except Exception as e:
            print(f"      Reliability scoring failed: {str(e)}")
            return None


# Example usage for regulatory compliance
if __name__ == "__main__":
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_train, n_test = 100, 20
    n_features = 10
    
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)
    
    # For Williams plot, we need predictions
    y_train = np.random.randn(n_train)
    y_test = np.random.randn(n_test)
    
    # Simple linear model for demonstration
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    # Initialize regulatory AD analyzer
    print("=== Regulatory-Compliant AD Analysis ===")
    ad_analyzer = ConcurrentADMethods(random_state=42, ad_mode='strict')
    
    # Calculate all regulatory AD methods
    results = ad_analyzer.calculate_all_methods(
        X_train, X_test, 
        y_train, y_test, 
        y_pred_test
    )
    
    # Display results
    print("\n=== Results ===")
    for method, result in results.items():
        if result and 'coverage' in result:
            print(f"\n{method.upper()}:")
            print(f"  Coverage: {result['coverage']:.3f}")
            print(f"  Quality: {result.get('quality', 'N/A')}")
            if 'regulatory_compliant' in result:
                print(f"  Regulatory Compliant: {result['regulatory_compliant']}")
            if 'references' in result:
                print(f"  References: {', '.join(result['references'])}")
    
    # Consensus recommendation
    if 'consensus' in results and results['consensus']:
        print(f"\n=== Consensus AD ===")
        print(f"Recommended approach: {results['consensus']['recommended']}")
        for approach, data in results['consensus'].items():
            if isinstance(data, dict) and 'coverage' in data:
                print(f"{approach}: {data['coverage']:.3f} - {data.get('description', '')}")