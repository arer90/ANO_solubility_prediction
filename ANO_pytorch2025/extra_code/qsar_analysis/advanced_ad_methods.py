"""
Advanced Applicability Domain Methods Module
Additional AD methods for enhanced domain assessment

References:
- Sahigara et al. (2012) Comparison of Different Approaches to Define the Applicability Domain
- Jaworska et al. (2005) QSAR Applicability Domain Estimation by Projection
- Netzeva et al. (2005) Current Status of Methods for Defining the Applicability Domain
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import warnings

warnings.filterwarnings('ignore')


class AdvancedADMethods:
    """
    Advanced Applicability Domain methods for comprehensive assessment
    """
    
    def __init__(self, ad_mode: str = 'strict'):
        self.ad_mode = ad_mode
        self.scaler = StandardScaler()
        
    def calculate_mahalanobis_distance(self, X_train: np.ndarray, 
                                      X_test: np.ndarray) -> Dict:
        """
        Calculate Mahalanobis distance for AD assessment
        
        Advantages:
        - Accounts for correlation between descriptors
        - Scale-invariant
        - Better for ellipsoidal AD boundaries
        """
        try:
            # Standardize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Calculate robust covariance matrix using Minimum Covariance Determinant
            robust_cov = MinCovDet(random_state=42, support_fraction=0.8)
            robust_cov.fit(X_train_scaled)
            
            # Calculate Mahalanobis distances
            train_distances = robust_cov.mahalanobis(X_train_scaled)
            test_distances = robust_cov.mahalanobis(X_test_scaled)
            
            # Define threshold using chi-square distribution
            # For multivariate normal data, Mahalanobis distance^2 follows chi-square
            n_features = X_train.shape[1]
            
            if self.ad_mode == 'strict':
                threshold = np.sqrt(stats.chi2.ppf(0.95, n_features))
            elif self.ad_mode == 'flexible':
                threshold = np.sqrt(stats.chi2.ppf(0.975, n_features))
            else:  # adaptive
                threshold = np.sqrt(stats.chi2.ppf(0.99, n_features))
            
            # Alternative: use training data quantile
            train_threshold = np.percentile(np.sqrt(train_distances), 95)
            threshold = min(threshold, train_threshold * 1.2)
            
            # Calculate AD membership
            test_distances_sqrt = np.sqrt(test_distances)
            in_ad = test_distances_sqrt <= threshold
            
            return {
                'train_distances': np.sqrt(train_distances),
                'test_distances': test_distances_sqrt,
                'threshold': threshold,
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'method': 'mahalanobis_distance',
                'chi2_critical': threshold,
                'robust_center': robust_cov.location_,
                'n_features': n_features
            }
            
        except Exception as e:
            print(f"            [WARNING] Mahalanobis distance failed: {str(e)}")
            return None
    
    def calculate_hotellings_t2(self, X_train: np.ndarray, 
                                X_test: np.ndarray) -> Dict:
        """
        Calculate Hotelling's T² statistic for AD assessment
        
        Common in chemometrics and regulatory applications
        """
        try:
            # Standardize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Calculate mean and covariance
            mean_train = np.mean(X_train_scaled, axis=0)
            cov_train = np.cov(X_train_scaled.T)
            
            # Add regularization to avoid singular matrix
            cov_train_reg = cov_train + 1e-6 * np.eye(cov_train.shape[0])
            inv_cov = np.linalg.inv(cov_train_reg)
            
            # Calculate T² for training data
            n_train = X_train_scaled.shape[0]
            t2_train = []
            for i in range(n_train):
                diff = X_train_scaled[i] - mean_train
                t2_train.append(diff @ inv_cov @ diff.T)
            t2_train = np.array(t2_train)
            
            # Calculate T² for test data
            n_test = X_test_scaled.shape[0]
            t2_test = []
            for i in range(n_test):
                diff = X_test_scaled[i] - mean_train
                t2_test.append(diff @ inv_cov @ diff.T)
            t2_test = np.array(t2_test)
            
            # Calculate critical value using F-distribution
            n_features = X_train.shape[1]
            alpha = 0.05 if self.ad_mode == 'strict' else 0.01
            
            # T² critical value formula
            t2_crit = ((n_train - 1) * n_features / (n_train - n_features)) * \
                     stats.f.ppf(1 - alpha, n_features, n_train - n_features)
            
            # Alternative: use training data quantile
            train_threshold = np.percentile(t2_train, 95)
            threshold = min(t2_crit, train_threshold * 1.2)
            
            # Determine AD membership
            in_ad = t2_test <= threshold
            
            return {
                'train_t2': t2_train,
                'test_t2': t2_test,
                'threshold': threshold,
                't2_critical': t2_crit,
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'method': 'hotellings_t2',
                'n_features': n_features,
                'n_train': n_train
            }
            
        except Exception as e:
            print(f"            [WARNING] Hotelling's T² failed: {str(e)}")
            return None
    
    def calculate_bounding_box(self, X_train: np.ndarray, 
                               X_test: np.ndarray) -> Dict:
        """
        Simple bounding box (hyper-rectangle) method
        
        Advantages:
        - Very interpretable
        - Fast computation
        - Good for regulatory submissions
        """
        try:
            # Calculate min/max for each descriptor
            min_vals = np.min(X_train, axis=0)
            max_vals = np.max(X_train, axis=0)
            
            # Add margin based on mode
            if self.ad_mode == 'strict':
                margin = 0.0  # No extrapolation
            elif self.ad_mode == 'flexible':
                margin = 0.1  # 10% extrapolation
            else:  # adaptive
                margin = 0.05  # 5% extrapolation
            
            ranges = max_vals - min_vals
            min_bounds = min_vals - margin * ranges
            max_bounds = max_vals + margin * ranges
            
            # Check if test samples are within bounds
            in_ad = np.all((X_test >= min_bounds) & (X_test <= max_bounds), axis=1)
            
            # Calculate how many descriptors are out of bounds for each sample
            n_out_of_bounds = np.sum((X_test < min_bounds) | (X_test > max_bounds), axis=1)
            
            # Calculate interpolation score (0 = fully outside, 1 = fully inside)
            n_descriptors = X_train.shape[1]
            interpolation_score = 1 - (n_out_of_bounds / n_descriptors)
            
            return {
                'min_bounds': min_bounds,
                'max_bounds': max_bounds,
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'method': 'bounding_box',
                'n_out_of_bounds': n_out_of_bounds,
                'interpolation_score': interpolation_score,
                'margin': margin
            }
            
        except Exception as e:
            print(f"            [WARNING] Bounding box failed: {str(e)}")
            return None
    
    def calculate_confidence_index(self, ad_results: Dict) -> np.ndarray:
        """
        Calculate confidence index based on multiple AD methods
        
        Returns confidence score [0, 1] for each test sample
        """
        if not ad_results:
            return None
        
        # Collect all AD predictions
        ad_matrix = []
        weights = {
            'mahalanobis_distance': 1.2,  # Higher weight for sophisticated methods
            'hotellings_t2': 1.2,
            'euclidean_distance': 1.0,
            'knn_distance': 1.0,
            'descriptor_range': 0.8,
            'bounding_box': 0.8
        }
        
        method_weights = []
        for method, result in ad_results.items():
            if result and 'in_ad' in result:
                ad_matrix.append(result['in_ad'].astype(float))
                method_weights.append(weights.get(method, 1.0))
        
        if not ad_matrix:
            return None
        
        ad_matrix = np.array(ad_matrix)
        method_weights = np.array(method_weights) / np.sum(method_weights)
        
        # Weighted average of AD predictions
        confidence = np.average(ad_matrix, axis=0, weights=method_weights)
        
        return confidence
    
    def calculate_reliability_index(self, ad_results: Dict, 
                                   X_test: np.ndarray) -> Dict:
        """
        Calculate AD reliability index combining multiple factors
        
        Factors:
        - Method agreement
        - Distance from training domain center
        - Density of training samples nearby
        """
        if not ad_results:
            return None
        
        # 1. Method agreement score
        confidence = self.calculate_confidence_index(ad_results)
        if confidence is None:
            return None
        
        # 2. Distance-based reliability
        distance_scores = []
        for method in ['euclidean_distance', 'mahalanobis_distance', 'knn_distance']:
            if method in ad_results and ad_results[method]:
                result = ad_results[method]
                if 'test_distances' in result and 'threshold' in result:
                    # Normalize distances to [0, 1] where 0 is at threshold
                    norm_dist = 1 - (result['test_distances'] / result['threshold'])
                    norm_dist = np.clip(norm_dist, 0, 1)
                    distance_scores.append(norm_dist)
        
        if distance_scores:
            avg_distance_score = np.mean(distance_scores, axis=0)
        else:
            avg_distance_score = confidence  # Fallback to confidence
        
        # 3. Combined reliability index
        reliability = (confidence + avg_distance_score) / 2
        
        # Categorize reliability
        reliability_categories = []
        for r in reliability:
            if r >= 0.8:
                reliability_categories.append('High')
            elif r >= 0.6:
                reliability_categories.append('Medium')
            elif r >= 0.4:
                reliability_categories.append('Low')
            else:
                reliability_categories.append('Very Low')
        
        return {
            'confidence_scores': confidence,
            'distance_reliability': avg_distance_score,
            'overall_reliability': reliability,
            'reliability_categories': reliability_categories,
            'high_reliability_fraction': np.mean(reliability >= 0.8),
            'low_reliability_fraction': np.mean(reliability < 0.4)
        }
    
    def create_ad_profile(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """
        Create comprehensive AD profile using multiple methods
        """
        print("            Calculating advanced AD methods...")
        
        results = {}
        
        # Calculate all advanced methods
        methods = [
            ('mahalanobis_distance', self.calculate_mahalanobis_distance),
            ('hotellings_t2', self.calculate_hotellings_t2),
            ('bounding_box', self.calculate_bounding_box)
        ]
        
        for method_name, method_func in methods:
            print(f"              {method_name}...", end=' ')
            result = method_func(X_train, X_test)
            if result:
                results[method_name] = result
                print(f"[OK] (coverage: {result['coverage']:.2%})")
            else:
                print("[FAILED]")
        
        # Calculate reliability metrics
        if results:
            reliability = self.calculate_reliability_index(results, X_test)
            if reliability:
                results['reliability_index'] = reliability
                print(f"              Reliability index calculated: "
                     f"{reliability['high_reliability_fraction']:.1%} high reliability")
        
        return results


class ADEnsemble:
    """
    Ensemble approach combining standard and advanced AD methods
    """
    
    def __init__(self, ad_mode: str = 'strict'):
        self.ad_mode = ad_mode
        self.advanced_methods = AdvancedADMethods(ad_mode)
        
    def create_ensemble_ad(self, X_train: np.ndarray, X_test: np.ndarray,
                          standard_ad_results: Dict) -> Dict:
        """
        Combine standard and advanced AD methods for robust assessment
        """
        # Get advanced AD results
        advanced_results = self.advanced_methods.create_ad_profile(X_train, X_test)
        
        # Combine all results
        all_results = {**standard_ad_results}
        all_results.update(advanced_results)
        
        # Create ensemble prediction
        all_in_ad = []
        method_names = []
        
        for method, result in all_results.items():
            if result and 'in_ad' in result and method != 'reliability_index':
                all_in_ad.append(result['in_ad'])
                method_names.append(method)
        
        if all_in_ad:
            all_in_ad = np.array(all_in_ad)
            
            # Different ensemble strategies based on mode
            if self.ad_mode == 'strict':
                # Conservative: all methods must agree
                ensemble_in_ad = np.all(all_in_ad, axis=0)
            elif self.ad_mode == 'flexible':
                # Majority vote
                ensemble_in_ad = np.mean(all_in_ad, axis=0) >= 0.5
            else:  # adaptive
                # Weighted voting based on method reliability
                weights = self._get_method_weights(method_names)
                weighted_votes = np.average(all_in_ad, axis=0, weights=weights)
                ensemble_in_ad = weighted_votes >= 0.6
            
            ensemble_result = {
                'in_ad': ensemble_in_ad,
                'coverage': np.mean(ensemble_in_ad),
                'method_agreement': np.mean(all_in_ad, axis=0),
                'n_methods': len(method_names),
                'methods_used': method_names
            }
            
            all_results['ensemble'] = ensemble_result
        
        return all_results
    
    def _get_method_weights(self, method_names: List[str]) -> np.ndarray:
        """
        Get reliability weights for different AD methods
        """
        weight_map = {
            'mahalanobis_distance': 1.5,
            'hotellings_t2': 1.5,
            'leverage': 1.2,
            'euclidean_distance': 1.0,
            'knn_distance': 1.0,
            'descriptor_range': 0.8,
            'bounding_box': 0.8
        }
        
        weights = [weight_map.get(method, 1.0) for method in method_names]
        return np.array(weights) / np.sum(weights)