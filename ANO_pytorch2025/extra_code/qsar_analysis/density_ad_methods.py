"""
Density-based Applicability Domain Methods
Advanced methods using density and anomaly detection

References:
- Breunig et al. (2000) LOF: Identifying Density-Based Local Outliers
- Liu et al. (2008) Isolation Forest
- Schölkopf et al. (2001) Estimating the Support of a High-Dimensional Distribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class DensityBasedADMethods:
    """
    Density and anomaly detection based AD methods
    """
    
    def __init__(self, ad_mode: str = 'strict', random_state: int = 42):
        self.ad_mode = ad_mode
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Configure contamination based on mode
        if ad_mode == 'strict':
            self.contamination = 0.05  # Expect 5% outliers
        elif ad_mode == 'flexible':
            self.contamination = 0.1   # Expect 10% outliers
        else:  # adaptive
            self.contamination = 0.075 # Expect 7.5% outliers
    
    def calculate_lof(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """
        Local Outlier Factor for AD assessment
        
        Advantages:
        - Detects local anomalies
        - Good for non-uniform density distributions
        - Considers neighborhood context
        """
        try:
            # Standardize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Configure LOF
            n_neighbors = min(20, len(X_train) // 10)
            
            # Train LOF on training data
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                novelty=True,  # Enable prediction on new data
                n_jobs=-1
            )
            lof.fit(X_train_scaled)
            
            # Predict on test data (-1 for outliers, 1 for inliers)
            predictions = lof.predict(X_test_scaled)
            in_ad = predictions == 1
            
            # Get anomaly scores (negative scores = more abnormal)
            scores = lof.score_samples(X_test_scaled)
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Get decision boundary
            decision_boundary = lof.offset_
            
            return {
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'anomaly_scores': -scores,  # Convert to positive anomaly scores
                'normalized_scores': normalized_scores,
                'decision_boundary': decision_boundary,
                'method': 'local_outlier_factor',
                'n_neighbors': n_neighbors,
                'contamination': self.contamination
            }
            
        except Exception as e:
            print(f"            [WARNING] LOF failed: {str(e)}")
            return None
    
    def calculate_isolation_forest(self, X_train: np.ndarray, 
                                   X_test: np.ndarray) -> Dict:
        """
        Isolation Forest for AD assessment
        
        Advantages:
        - Very fast
        - Works well with high-dimensional data
        - No distance calculations needed
        """
        try:
            # Standardize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Configure Isolation Forest
            n_estimators = 100 if self.ad_mode == 'strict' else 50
            
            iso_forest = IsolationForest(
                n_estimators=n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Fit on training data
            iso_forest.fit(X_train_scaled)
            
            # Predict on test data
            predictions = iso_forest.predict(X_test_scaled)
            in_ad = predictions == 1
            
            # Get anomaly scores (lower = more abnormal)
            scores = iso_forest.score_samples(X_test_scaled)
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Get path lengths for interpretation
            path_lengths = iso_forest.decision_function(X_test_scaled)
            
            return {
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'anomaly_scores': -scores,
                'normalized_scores': normalized_scores,
                'path_lengths': path_lengths,
                'method': 'isolation_forest',
                'n_estimators': n_estimators,
                'contamination': self.contamination
            }
            
        except Exception as e:
            print(f"            [WARNING] Isolation Forest failed: {str(e)}")
            return None
    
    def calculate_one_class_svm(self, X_train: np.ndarray, 
                                X_test: np.ndarray) -> Dict:
        """
        One-Class SVM for AD assessment
        
        Advantages:
        - Robust to outliers
        - Works well for high-dimensional data
        - Flexible boundary shapes with RBF kernel
        """
        try:
            # Standardize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Reduce dimensionality if needed
            n_features = X_train_scaled.shape[1]
            if n_features > 20:
                pca = PCA(n_components=min(20, n_features), random_state=self.random_state)
                X_train_scaled = pca.fit_transform(X_train_scaled)
                X_test_scaled = pca.transform(X_test_scaled)
                variance_explained = pca.explained_variance_ratio_.sum()
            else:
                variance_explained = 1.0
            
            # Configure One-Class SVM
            nu = self.contamination  # Nu parameter ≈ outlier fraction
            
            ocsvm = OneClassSVM(
                kernel='rbf',
                nu=nu,
                gamma='auto'
            )
            
            # Fit on training data
            ocsvm.fit(X_train_scaled)
            
            # Predict on test data
            predictions = ocsvm.predict(X_test_scaled)
            in_ad = predictions == 1
            
            # Get decision scores
            scores = ocsvm.decision_function(X_test_scaled)
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            return {
                'in_ad': in_ad,
                'coverage': np.mean(in_ad),
                'decision_scores': scores,
                'normalized_scores': normalized_scores,
                'method': 'one_class_svm',
                'nu': nu,
                'variance_explained': variance_explained
            }
            
        except Exception as e:
            print(f"            [WARNING] One-Class SVM failed: {str(e)}")
            return None
    
    def calculate_combined_density_ad(self, X_train: np.ndarray, 
                                      X_test: np.ndarray) -> Dict:
        """
        Combine multiple density-based methods for robust AD assessment
        """
        results = {}
        
        # Calculate all density methods
        print("            Calculating density-based AD methods...")
        
        methods = [
            ('lof', self.calculate_lof),
            ('isolation_forest', self.calculate_isolation_forest),
            ('one_class_svm', self.calculate_one_class_svm)
        ]
        
        for method_name, method_func in methods:
            print(f"              {method_name}...", end=' ')
            result = method_func(X_train, X_test)
            if result:
                results[method_name] = result
                print(f"[OK] (coverage: {result['coverage']:.2%})")
            else:
                print("[FAILED]")
        
        # Create ensemble prediction if multiple methods succeeded
        if len(results) >= 2:
            all_predictions = []
            for result in results.values():
                if 'in_ad' in result:
                    all_predictions.append(result['in_ad'])
            
            if all_predictions:
                all_predictions = np.array(all_predictions)
                
                # Majority voting
                ensemble_in_ad = np.mean(all_predictions, axis=0) >= 0.5
                
                # Confidence based on agreement
                agreement = np.mean(all_predictions, axis=0)
                
                results['density_ensemble'] = {
                    'in_ad': ensemble_in_ad,
                    'coverage': np.mean(ensemble_in_ad),
                    'method_agreement': agreement,
                    'confidence': agreement,
                    'n_methods': len(all_predictions)
                }
                
                print(f"              Density ensemble: {results['density_ensemble']['coverage']:.2%} coverage")
        
        return results


class TemporalDriftDetector:
    """
    Detect temporal and concept drift in AD
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.scaler = StandardScaler()
    
    def detect_drift(self, X_train: np.ndarray, X_test: np.ndarray,
                     timestamps_train: Optional[np.ndarray] = None,
                     timestamps_test: Optional[np.ndarray] = None) -> Dict:
        """
        Detect temporal drift between training and test sets
        """
        # If no timestamps, use sequential ordering
        if timestamps_train is None:
            timestamps_train = np.arange(len(X_train))
        if timestamps_test is None:
            timestamps_test = np.arange(len(X_train), len(X_train) + len(X_test))
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate distribution statistics over time windows
        drift_metrics = {}
        
        # 1. Feature-wise drift (Kolmogorov-Smirnov test)
        from scipy import stats
        ks_statistics = []
        ks_pvalues = []
        
        for feature_idx in range(X_train.shape[1]):
            ks_stat, p_value = stats.ks_2samp(
                X_train_scaled[:, feature_idx],
                X_test_scaled[:, feature_idx]
            )
            ks_statistics.append(ks_stat)
            ks_pvalues.append(p_value)
        
        # 2. Multivariate drift (Maximum Mean Discrepancy)
        mmd = self._calculate_mmd(X_train_scaled, X_test_scaled)
        
        # 3. Centroid shift
        train_centroid = np.mean(X_train_scaled, axis=0)
        test_centroid = np.mean(X_test_scaled, axis=0)
        centroid_shift = np.linalg.norm(test_centroid - train_centroid)
        
        # 4. Covariance change
        train_cov = np.cov(X_train_scaled.T)
        test_cov = np.cov(X_test_scaled.T)
        cov_change = np.linalg.norm(test_cov - train_cov, 'fro')
        
        # Determine drift severity
        significant_features = np.sum(np.array(ks_pvalues) < 0.05)
        drift_severity = significant_features / len(ks_pvalues)
        
        if drift_severity < 0.1:
            drift_level = 'None'
        elif drift_severity < 0.3:
            drift_level = 'Low'
        elif drift_severity < 0.5:
            drift_level = 'Medium'
        else:
            drift_level = 'High'
        
        return {
            'ks_statistics': ks_statistics,
            'ks_pvalues': ks_pvalues,
            'significant_features': significant_features,
            'drift_severity': drift_severity,
            'drift_level': drift_level,
            'mmd': mmd,
            'centroid_shift': centroid_shift,
            'covariance_change': cov_change,
            'feature_importance_for_drift': np.argsort(ks_statistics)[::-1][:5]  # Top 5 drifting features
        }
    
    def _calculate_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate Maximum Mean Discrepancy between two distributions
        """
        n_x, n_y = len(X), len(Y)
        
        # Use RBF kernel
        gamma = 1.0 / X.shape[1]
        
        # Kernel matrices
        XX = np.exp(-gamma * np.sum((X[:, None] - X[None, :]) ** 2, axis=2))
        YY = np.exp(-gamma * np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2))
        XY = np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))
        
        # MMD
        mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        return mmd
    
    def visualize_drift(self, drift_results: Dict) -> None:
        """
        Create drift visualization (placeholder for actual implementation)
        """
        print(f"\nDrift Detection Results:")
        print(f"  Drift Level: {drift_results['drift_level']}")
        print(f"  Drift Severity: {drift_results['drift_severity']:.2%}")
        print(f"  Significant Features: {drift_results['significant_features']}")
        print(f"  MMD: {drift_results['mmd']:.4f}")
        print(f"  Centroid Shift: {drift_results['centroid_shift']:.4f}")