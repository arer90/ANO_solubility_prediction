# ad_performance_analysis.py (SVD 에러 수정 버전)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import warnings
import gc
import time
import pickle
from dataclasses import dataclass

# XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Using only Random Forest.")

# LightGBM import with fallback
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for model training"""
    rf_params: Dict
    xgb_params: Dict
    lgb_params: Dict
    n_jobs: int


class ADPerformanceAnalyzer:
    """
    Sequential AD Performance Analyzer - SVD Error Fixed Version
    
    Features:
    - Vectorized NumPy operations with error handling
    - Sequential model training and evaluation
    - Memory-efficient chunking
    - Support for RF, XGBoost, and LightGBM
    - Split-type based analysis
    """
    
    def __init__(self, output_dir: Path, n_jobs: int = -1, 
             chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.performance_dir = output_dir / 'ad_analysis' / 'performance'
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Create consistent structure
        self.by_split_dir = self.performance_dir / 'by_split'
        self.overall_dir = self.performance_dir / 'overall'
        
        self.by_split_dir.mkdir(parents=True, exist_ok=True)
        self.overall_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split-specific directories - 이름 매칭 수정
        self.split_dirs = {
            'random': self.by_split_dir / 'random_split',
            'scaffold': self.by_split_dir / 'scaffold_split',
            'chemical_space': self.by_split_dir / 'chemical_space_split',  # 일관성 위해 _split 추가
            'chemical_space_coverage': self.by_split_dir / 'chemical_space_split',  # 별칭 추가
            'cluster': self.by_split_dir / 'cluster_split',
            'physchem': self.by_split_dir / 'physchem_split',  # 일관성 위해 _split 추가
            'activity_cliff': self.by_split_dir / 'activity_cliff_split',  # 일관성 위해 _split 추가
            'solubility_aware': self.by_split_dir / 'solubility_aware_split',  # 일관성 위해 _split 추가
            'time': self.by_split_dir / 'time_split',
            'time_series': self.by_split_dir / 'time_split',  # time_series를 time으로 매핑
            'ensemble': self.by_split_dir / 'ensemble_split',  # 일관성 위해 _split 추가
            'test_only': self.by_split_dir / 'test_only',
            'unknown': self.by_split_dir / 'unknown'  # 알 수 없는 split type을 위한 기본값
        }
        
        # 모든 디렉토리 생성 (중복 제거)
        unique_dirs = set(self.split_dirs.values())
        for split_dir in unique_dirs:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # 메모리 제한 설정
        try:
            from config import MAX_MEMORY_MB
            self.memory_limit = MAX_MEMORY_MB * 1024 * 1024
        except ImportError:
            self.memory_limit = 8192 * 1024 * 1024  # Default 8GB
        except Exception:
            import psutil
            self.memory_limit = int(psutil.virtual_memory().total * 0.6)
        
        # Set number of jobs for models (not for parallel processing)
        self.n_jobs = 1  # Sequential processing
        
        # Initialize model configurations
        self.model_config = self._init_model_config()
        
        # Chunk size for batch processing
        self.chunk_size = chunk_size
        
    def _init_model_config(self) -> ModelConfig:
        """Initialize optimized model configurations"""
        return ModelConfig(
            rf_params={
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'random_state': 42,
                'n_jobs': 1,
                'warm_start': False
            },
            xgb_params={
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'n_jobs': 1,
                'tree_method': 'hist',
                'predictor': 'cpu_predictor',
                'enable_categorical': False
            },
            lgb_params={
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0,
                'lambda_l2': 1,
                'min_data_in_leaf': 20,
                'random_state': 42,
                'n_jobs': 1,
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1
            },
            n_jobs=1
        )
    
    def _validate_and_clean_data(self, X: np.ndarray, y: np.ndarray, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and clean data to prevent SVD errors"""
        # Check for NaN or Inf values
        X_valid = np.isfinite(X).all(axis=1)
        y_valid = np.isfinite(y)
        valid_mask = X_valid & y_valid
        
        if not valid_mask.all():
            print(f"    Warning: Removing {(~valid_mask).sum()} invalid samples from {name}")
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Check for constant features
        if len(X) > 0:
            std_features = np.std(X, axis=0)
            constant_features = std_features == 0
            if constant_features.any():
                print(f"    Warning: Found {constant_features.sum()} constant features")
                # Add small noise to constant features
                X[:, constant_features] += np.random.normal(0, 1e-8, size=(len(X), constant_features.sum()))
        
        return X, y
    
    def analyze_ad_performance_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray, 
                                ad_results: Dict, dataset_name: str,
                                split_type: str = None,
                                models: List[str] = None,
                                ad_mode: str = 'flexible') -> Dict[str, pd.DataFrame]:
        """Analyze AD performance with split type and mode organization"""
        if models is None:
            models = ['rf']
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')
        
        # Extract split type from dataset_name if not provided
        if split_type is None:
            # Check all possible split types
            split_mappings = {
                'random': 'random',
                'scaffold': 'scaffold',
                'chemical_space_coverage': 'chemical_space',
                'chemical_space': 'chemical_space',
                'cluster': 'cluster',
                'physchem': 'physchem',
                'activity_cliff': 'activity_cliff',
                'solubility_aware': 'solubility_aware',
                'time_series': 'time',
                'time': 'time',
                'ensemble': 'ensemble',
                'test_only': 'test_only'
            }
            
            for split_name, mapped_split in split_mappings.items():
                if split_name in dataset_name.lower():
                    split_type = mapped_split
                    break
            
            if split_type is None:
                split_type = 'unknown'
        
        print(f"\nAnalyzing AD performance for {dataset_name} (Split: {split_type}, Mode: {ad_mode}) using {models}...")
    
        start_time = time.time()
        
        results = {}
        
        # Validate and clean data
        X_train, y_train = self._validate_and_clean_data(X_train, y_train, "training data")
        X_test, y_test = self._validate_and_clean_data(X_test, y_test, "test data")
        
        # Validate data consistency
        if len(X_test) != len(y_test):
            print(f"  ❌ Test data size mismatch: X_test={len(X_test)}, y_test={len(y_test)}")
            return results
        
        # Sequential model analysis
        results = self._sequential_model_analysis(
            X_train, y_train, X_test, y_test, ad_results, dataset_name, split_type, models, ad_mode
        )
        
        # Create comparison visualization if multiple models
        if len(results) > 1:
            try:
                self._create_model_comparison_plot_optimized(results, dataset_name, split_type, ad_mode)
            except Exception as e:
                print(f"  Warning: Failed to create model comparison plot: {e}")
        
        # Create split-specific summary
        try:
            self._create_split_type_summary(results, dataset_name, split_type, ad_mode)
        except Exception as e:
            print(f"  Warning: Failed to create split summary: {e}")
        
        elapsed = time.time() - start_time
        print(f"  Total analysis time: {elapsed:.2f}s")
        
        return results
    
    def _sequential_model_analysis(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 ad_results: Dict, dataset_name: str,
                                 split_type: str, models: List[str],
                                 ad_mode: str = 'flexible') -> Dict[str, pd.DataFrame]:
        """Sequential model analysis"""
        results = {}
        
        for model_name in models:
            try:
                if model_name == 'rf':
                    result = self._analyze_single_model(
                        X_train, y_train, X_test, y_test,
                        ad_results, dataset_name, split_type, 'RandomForest',
                        self._train_rf_model_optimized, ad_mode
                    )
                    results['rf'] = result
                elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                    result = self._analyze_single_model(
                        X_train, y_train, X_test, y_test,
                        ad_results, dataset_name, split_type, 'XGBoost',
                        self._train_xgb_model_optimized, ad_mode
                    )
                    results['xgboost'] = result
                elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    result = self._analyze_single_model(
                        X_train, y_train, X_test, y_test,
                        ad_results, dataset_name, split_type, 'LightGBM',
                        self._train_lgb_model_optimized, ad_mode
                    )
                    results['lightgbm'] = result
            except Exception as e:
                print(f"  ❌ {model_name} analysis failed: {str(e)}")
        
        return results
    
    def _analyze_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            ad_results: Dict, dataset_name: str,
                            split_type: str, model_type: str, train_func,
                            ad_mode: str = 'flexible') -> pd.DataFrame:
        """Analyze single model"""
        start_time = time.time()
        print(f"  Training {model_type} model...")
        
        # Train model (no cache)
        model = train_func(X_train, y_train)
        
        # Vectorized predictions
        y_pred_test = self._predict_chunked(model, X_test)
        
        # Vectorized performance analysis
        results = self._vectorized_ad_analysis(
            ad_results, y_test, y_pred_test, dataset_name
        )
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('rmse_ratio', ascending=True)
        
        # Save results in split-specific directory with mode and dataset structure
        save_dir = self.split_dirs.get(split_type, self.performance_dir) / ad_mode / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        excel_path = save_dir / f'ad_performance_{model_type.lower()}_{ad_mode}.xlsx'
        self._save_results_to_excel_optimized(df_results, excel_path, dataset_name, model_type)
        
        # Create visualization in split-specific directory with mode and dataset structure
        plot_path = save_dir / f'ad_performance_{model_type.lower()}_{ad_mode}.png'
        self._plot_ad_performance_analysis_optimized(df_results, dataset_name, model_type, plot_path)
        
        elapsed = time.time() - start_time
        print(f"  {model_type} analysis completed in {elapsed:.2f}s")
        
        return df_results
    
    def _create_split_type_summary(self, results: Dict[str, pd.DataFrame], 
                                 dataset_name: str, split_type: str, ad_mode: str = 'flexible'):
        """Create summary specific to split type"""
        save_dir = self.split_dirs.get(split_type, self.performance_dir) / ad_mode / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split-type specific summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison for this split
        ax1 = axes[0, 0]
        self._plot_model_comparison_for_split(ax1, results, split_type)
        
        # 2. AD method performance for this split
        ax2 = axes[0, 1]
        self._plot_method_performance_for_split(ax2, results, split_type)
        
        # 3. Coverage vs RMSE ratio
        ax3 = axes[1, 0]
        self._plot_coverage_vs_rmse_for_split(ax3, results, split_type)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        self._plot_split_summary_stats(ax4, results, dataset_name, split_type)
        
        plt.suptitle(f'{dataset_name} - {split_type.upper()} Split Analysis ({ad_mode} mode)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = save_dir / f'split_summary_{ad_mode}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()
        
        print(f"    Split summary saved to: {save_path.name}")
    
    def _plot_model_comparison_for_split(self, ax, results: Dict, split_type: str):
        """Plot model comparison for specific split type"""
        if not results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'Model Comparison - {split_type.upper()} Split')
            return
        
        models = list(results.keys())
        metrics = ['Mean RMSE Inside', 'Mean RMSE Outside', 'Mean RMSE Ratio']
        
        data = []
        for model in models:
            df = results[model]
            if len(df) > 0:  # Check if DataFrame is not empty
                data.append([
                    df['rmse_inside_ad'].mean(),
                    df['rmse_outside_ad'].mean(),
                    df['rmse_ratio'].mean()
                ])
        
        if not data:  # If no valid data
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(f'Model Comparison - {split_type.upper()} Split')
            return
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [d[i] for d in data]
            ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Value')
        ax.set_title(f'Model Comparison - {split_type.upper()} Split')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_method_performance_for_split(self, ax, results: Dict, split_type: str):
        """Plot AD method performance for specific split"""
        if not results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'AD Method Performance - {split_type.upper()} Split')
            return
        
        # Combine results from all models
        all_methods = set()
        for df in results.values():
            if len(df) > 0:  # Check DataFrame is not empty
                all_methods.update(df['method'].values)
        
        method_data = {method: {'rmse_ratios': [], 'coverages': []} for method in all_methods}
        
        for df in results.values():
            if len(df) > 0:  # Check DataFrame is not empty
                for _, row in df.iterrows():
                    method = row['method']
                    method_data[method]['rmse_ratios'].append(row['rmse_ratio'])
                    method_data[method]['coverages'].append(row['coverage'])
        
        # Calculate means and sort by RMSE ratio
        method_stats = []
        for method, data in method_data.items():
            if data['rmse_ratios']:
                method_stats.append({
                    'method': method,
                    'mean_rmse_ratio': np.mean(data['rmse_ratios']),
                    'mean_coverage': np.mean(data['coverages'])
                })
        
        if not method_stats:
            ax.text(0.5, 0.5, 'No valid statistics', ha='center', va='center')
            ax.set_title(f'AD Method Performance - {split_type.upper()} Split')
            return
        
        method_stats.sort(key=lambda x: x['mean_rmse_ratio'])
        
        # Plot
        methods = [s['method'] for s in method_stats]
        rmse_ratios = [s['mean_rmse_ratio'] for s in method_stats]
        coverages = [s['mean_coverage'] for s in method_stats]
        
        x = np.arange(len(methods))
        
        # Create bar plot with proper colors
        bars = ax.bar(x, rmse_ratios, alpha=0.8)
        
        # Color by performance
        for bar, ratio in zip(bars, rmse_ratios):
            if ratio < 1.5:
                bar.set_facecolor('darkgreen')
            elif ratio < 2.0:
                bar.set_facecolor('orange')
            else:
                bar.set_facecolor('red')
        
        ax.set_xlabel('AD Method')
        ax.set_ylabel('Mean RMSE Ratio')
        ax.set_title(f'AD Method Performance - {split_type.upper()} Split')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_coverage_vs_rmse_for_split(self, ax, results: Dict, split_type: str):
        """Plot coverage vs RMSE ratio for split"""
        if not results:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(f'Coverage vs Performance - {split_type.upper()} Split')
            return
        
        # Collect all data points
        coverages = []
        rmse_ratios = []
        methods = []
        
        for model_name, df in results.items():
            for _, row in df.iterrows():
                coverages.append(row['coverage'] * 100)
                rmse_ratios.append(row['rmse_ratio'])
                methods.append(row['method'])
        
        if not coverages:
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            ax.set_title(f'Coverage vs Performance - {split_type.upper()} Split')
            return
        
        # Create scatter plot
        scatter = ax.scatter(coverages, rmse_ratios, alpha=0.6, s=100)
        
        # Add trend line with error handling
        if len(coverages) > 2:
            try:
                # Use robust polynomial fitting
                valid_mask = np.isfinite(coverages) & np.isfinite(rmse_ratios)
                if valid_mask.sum() > 2:
                    z = np.polyfit(np.array(coverages)[valid_mask], 
                                 np.array(rmse_ratios)[valid_mask], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(coverages), max(coverages), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2)
            except Exception as e:
                print(f"      Warning: Could not fit trend line: {e}")
        
        ax.set_xlabel('AD Coverage (%)')
        ax.set_ylabel('RMSE Ratio')
        ax.set_title(f'Coverage vs Performance - {split_type.upper()} Split')
        ax.axhline(1.0, color='green', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_split_summary_stats(self, ax, results: Dict, dataset_name: str, split_type: str):
        """Plot summary statistics for split"""
        summary_text = f"SPLIT ANALYSIS SUMMARY\n"
        summary_text += f"{'='*30}\n\n"
        summary_text += f"Dataset: {dataset_name}\n"
        summary_text += f"Split Type: {split_type.upper()}\n"
        summary_text += f"Models Analyzed: {', '.join(results.keys())}\n\n"
        
        # Calculate overall statistics
        all_rmse_ratios = []
        all_coverages = []
        
        for df in results.values():
            all_rmse_ratios.extend(df['rmse_ratio'].values)
            all_coverages.extend(df['coverage'].values)
        
        if all_rmse_ratios:
            summary_text += "OVERALL METRICS:\n"
            summary_text += f"• Mean RMSE Ratio: {np.mean(all_rmse_ratios):.3f}\n"
            summary_text += f"• Mean Coverage: {np.mean(all_coverages):.3f}\n"
            summary_text += f"• Best RMSE Ratio: {np.min(all_rmse_ratios):.3f}\n"
            summary_text += f"• Methods Analyzed: {len(set().union(*[set(df['method']) for df in results.values()]))}\n\n"
            
            # Best performing method
            best_method = None
            best_ratio = float('inf')
            for df in results.values():
                idx = df['rmse_ratio'].idxmin()
                if df.loc[idx, 'rmse_ratio'] < best_ratio:
                    best_ratio = df.loc[idx, 'rmse_ratio']
                    best_method = df.loc[idx, 'method']
            
            summary_text += f"BEST METHOD: {best_method}\n"
            summary_text += f"• RMSE Ratio: {best_ratio:.3f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
        ax.axis('off')
    
    # Keep other methods unchanged...
    def _train_rf_model_optimized(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train optimized Random Forest model"""
        rf_model = RandomForestRegressor(**self.model_config.rf_params)
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def _train_xgb_model_optimized(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train optimized XGBoost model"""
        if XGBOOST_AVAILABLE:
            params = self.model_config.xgb_params.copy()
            params['tree_method'] = 'hist'
            params['predictor'] = 'cpu_predictor'
            
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(X_train, y_train)
            return xgb_model
    
    def _train_lgb_model_optimized(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train optimized LightGBM model"""
        if LIGHTGBM_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(**self.model_config.lgb_params)
            lgb_model.fit(X_train, y_train)
            return lgb_model
    
    def _predict_chunked(self, model, X: np.ndarray) -> np.ndarray:
        """Make predictions in chunks for memory efficiency"""
        if len(X) <= self.chunk_size:
            return model.predict(X)
        
        predictions = []
        for i in range(0, len(X), self.chunk_size):
            chunk = X[i:i+self.chunk_size]
            chunk_pred = model.predict(chunk)
            predictions.extend(chunk_pred)
        
        return np.array(predictions)
    
    @staticmethod
    def _calculate_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """NumPy-based metric calculation"""
        # Remove any NaN or Inf values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            return np.nan, np.nan, np.nan
        
        # RMSE
        squared_errors = (y_true - y_pred) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        
        # MAE
        absolute_errors = np.abs(y_true - y_pred)
        mae = np.mean(absolute_errors)
        
        # R2
        mean_y = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return rmse, mae, r2
    
    def _vectorized_ad_analysis(self, ad_results: Dict, y_test: np.ndarray,
                          y_pred_test: np.ndarray, dataset_name: str) -> Dict:
        """Vectorized AD analysis with index validation"""
        # Overall metrics (기존 코드 유지)
        rmse_total, mae_total, r2_total = self._calculate_metrics_numpy(y_test, y_pred_test)
        
        print(f"    Overall - RMSE: {rmse_total:.3f}, MAE: {mae_total:.3f}, R²: {r2_total:.3f}")
        
        # Initialize results storage
        results = {
            'dataset': [],
            'method': [],
            'n_total': [],
            'n_inside_ad': [],
            'n_outside_ad': [],
            'coverage': [],
            'rmse_total': [],
            'rmse_inside_ad': [],
            'rmse_outside_ad': [],
            'rmse_ratio': [],
            'mae_inside_ad': [],
            'mae_outside_ad': [],
            'mae_ratio': [],
            'r2_inside_ad': [],
            'r2_outside_ad': [],
            'error_reduction': []
        }
        
        # Analyze each AD method
        for method, ad_data in ad_results.items():
            if ad_data and 'in_ad' in ad_data:
                # Vectorized boolean operations
                in_ad_mask = np.array(ad_data['in_ad'], dtype=bool)
                
                # Validate array length
                if len(in_ad_mask) != len(y_test):
                    print(f"      Warning: AD result size mismatch for {method}. " +
                        f"Expected {len(y_test)}, got {len(in_ad_mask)}. Adjusting...")
                    
                    # 크기 조정 - 원본 데이터 보호를 위해 복사본 사용
                    min_len = min(len(in_ad_mask), len(y_test))
                    in_ad_mask = in_ad_mask[:min_len]
                    y_test_adj = y_test[:min_len]
                    y_pred_test_adj = y_pred_test[:min_len]
                    
                    # 조정된 데이터로 전체 메트릭 재계산
                    rmse_total_adj, mae_total_adj, r2_total_adj = self._calculate_metrics_numpy(
                        y_test_adj, y_pred_test_adj
                    )
                    rmse_total_use = rmse_total_adj
                else:
                    # 크기가 맞으면 원본 사용
                    y_test_adj = y_test
                    y_pred_test_adj = y_pred_test
                    rmse_total_use = rmse_total
                
                out_ad_mask = ~in_ad_mask
                
                n_inside = np.sum(in_ad_mask)
                n_outside = np.sum(out_ad_mask)
                
                # Append basic info
                results['dataset'].append(dataset_name)
                results['method'].append(method)
                results['n_total'].append(len(y_test_adj))
                results['n_inside_ad'].append(int(n_inside))
                results['n_outside_ad'].append(int(n_outside))
                results['coverage'].append(ad_data.get('coverage', n_inside / len(y_test_adj)))
                results['rmse_total'].append(rmse_total_use)
                
                # Inside AD metrics
                if n_inside > 0:
                    y_true_in = y_test_adj[in_ad_mask]
                    y_pred_in = y_pred_test_adj[in_ad_mask]
                    rmse_in, mae_in, r2_in = self._calculate_metrics_numpy(y_true_in, y_pred_in)
                else:
                    rmse_in = mae_in = r2_in = np.nan
                
                results['rmse_inside_ad'].append(rmse_in)
                results['mae_inside_ad'].append(mae_in)
                results['r2_inside_ad'].append(r2_in)
                
                # Outside AD metrics
                if n_outside > 0:
                    y_true_out = y_test_adj[out_ad_mask]
                    y_pred_out = y_pred_test_adj[out_ad_mask]
                    rmse_out, mae_out, r2_out = self._calculate_metrics_numpy(y_true_out, y_pred_out)
                else:
                    rmse_out = mae_out = r2_out = np.nan
                
                results['rmse_outside_ad'].append(rmse_out)
                results['mae_outside_ad'].append(mae_out)
                results['r2_outside_ad'].append(r2_out)
                
                # Calculate ratios
                rmse_ratio = rmse_out / rmse_in if rmse_in > 0 and not np.isnan(rmse_out) else np.nan
                mae_ratio = mae_out / mae_in if mae_in > 0 and not np.isnan(mae_out) else np.nan
                error_reduction = ((rmse_out - rmse_in) / rmse_out * 100) if rmse_out > 0 and not np.isnan(rmse_in) else np.nan
                
                results['rmse_ratio'].append(rmse_ratio)
                results['mae_ratio'].append(mae_ratio)
                results['error_reduction'].append(error_reduction)
        
        return results
    
    def _save_results_to_excel_optimized(self, df_results: pd.DataFrame, excel_path: Path,
                                       dataset_name: str, model_name: str):
        """Save results to Excel with optimized formatting"""
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results
            df_results.to_excel(writer, sheet_name='AD_Performance', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': [
                    'Model Type',
                    'Mean RMSE Inside AD',
                    'Mean RMSE Outside AD', 
                    'Mean RMSE Ratio',
                    'Mean Error Reduction (%)',
                    'Best AD Method',
                    'Worst AD Method',
                    'Best RMSE Ratio',
                    'Methods Analyzed',
                    'Total Samples'
                ],
                'Value': [
                    model_name,
                    df_results['rmse_inside_ad'].mean(),
                    df_results['rmse_outside_ad'].mean(),
                    df_results['rmse_ratio'].mean(),
                    df_results['error_reduction'].mean(),
                    df_results.iloc[0]['method'] if len(df_results) > 0 else 'N/A',
                    df_results.iloc[-1]['method'] if len(df_results) > 0 else 'N/A',
                    df_results.iloc[0]['rmse_ratio'] if len(df_results) > 0 else np.nan,
                    len(df_results),
                    df_results['n_total'].iloc[0] if len(df_results) > 0 else 0
                ]
            }
            summary = pd.DataFrame(summary_data)
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Method ranking
            if len(df_results) > 0:
                ranking = df_results[['method', 'coverage', 'rmse_ratio', 'error_reduction', 
                                    'r2_inside_ad', 'r2_outside_ad']].copy()
                ranking['rank'] = range(1, len(ranking) + 1)
                ranking['score'] = (1/ranking['rmse_ratio'] + ranking['coverage'] + 
                                  ranking['error_reduction']/100).fillna(0)
                ranking = ranking.sort_values('score', ascending=False)
                ranking.to_excel(writer, sheet_name='Method_Ranking', index=False)
            
            # Detailed metrics
            if len(df_results) > 0:
                detailed = df_results.copy()
                detailed['rmse_improvement'] = (df_results['rmse_outside_ad'] - 
                                               df_results['rmse_inside_ad'])
                detailed['mae_improvement'] = (df_results['mae_outside_ad'] - 
                                             df_results['mae_inside_ad'])
                detailed.to_excel(writer, sheet_name='Detailed_Metrics', index=False)
        gc.collect()
        print(f"    Results saved to: {excel_path.name}")
    
    def _plot_ad_performance_analysis_optimized(self, df_results: pd.DataFrame, 
                                              dataset_name: str, model_name: str,
                                              save_path: Path):
        """Create optimized AD performance visualization"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Color palette
        n_methods = len(df_results)
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
        
        # 1. RMSE comparison
        self._plot_rmse_comparison_optimized(axes[0, 0], df_results, colors)
        
        # 2. RMSE Ratio
        self._plot_rmse_ratio_optimized(axes[0, 1], df_results, colors)
        
        # 3. Coverage vs Performance
        self._plot_coverage_vs_performance_optimized(axes[0, 2], df_results)
        
        # 4. Sample distribution and R²
        self._plot_sample_distribution_optimized(axes[1, 0], df_results)
        
        # 5. Error metrics comparison
        self._plot_error_metrics_optimized(axes[1, 1], df_results)
        
        # 6. Performance summary
        self._plot_performance_summary_optimized(axes[1, 2], df_results, model_name)
        
        # Overall title
        fig.suptitle(f'{dataset_name}: AD Performance Analysis ({model_name})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        gc.collect()
        
        print(f"    Plot saved to: {save_path.name}")
    
    def _plot_rmse_comparison_optimized(self, ax, df_results, colors):
        """Optimized RMSE comparison plot"""
        methods = df_results['method'].values
        x = np.arange(len(methods))
        width = 0.35
        
        # Vectorized data
        rmse_inside = df_results['rmse_inside_ad'].values
        rmse_outside = df_results['rmse_outside_ad'].values
        
        # Create bars
        bars1 = ax.bar(x - width/2, rmse_inside, width, 
                       label='Inside AD', color='green', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, rmse_outside, width, 
                       label='Outside AD', color='red', alpha=0.7, edgecolor='black')
        
        # Vectorized value labels
        for bars, values in [(bars1, rmse_inside), (bars2, rmse_outside)]:
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.annotate(f'{val:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8)
        
        ax.set_xlabel('AD Method', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('RMSE Comparison: Inside vs Outside AD', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_rmse_ratio_optimized(self, ax, df_results, colors):
        """Optimized RMSE ratio plot with fixed color normalization"""
        methods = df_results['method'].values
        x = np.arange(len(methods))
        rmse_ratios = df_results['rmse_ratio'].values
        
        # Handle edge cases for color normalization
        valid_ratios = rmse_ratios[~np.isnan(rmse_ratios)]
        
        # Create colors for each bar
        bar_colors = []
        
        if len(valid_ratios) > 0:
            # Ensure vmin != vmax
            vmin = 1.0
            vmax = np.nanmax(valid_ratios)
            if vmax <= vmin:
                vmax = vmin + 0.1  # Add small offset to prevent error
            
            # Create gradient colors based on ratio values
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.RdYlGn_r
            
            # Generate colors for each ratio
            for ratio in rmse_ratios:
                if np.isnan(ratio):
                    bar_colors.append([0.5, 0.5, 0.5, 0.8])  # Gray as RGBA tuple
                else:
                    # Get color from colormap and convert to RGBA
                    rgba = cmap(norm(ratio))
                    bar_colors.append(rgba)
        else:
            # Use default gray if no valid ratios
            bar_colors = [[0.5, 0.5, 0.5, 0.8]] * len(rmse_ratios)
        
        bars = ax.bar(x, rmse_ratios, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, ratio in zip(bars, rmse_ratios):
            if not np.isnan(ratio):
                height = bar.get_height()
                ax.annotate(f'{ratio:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('AD Method', fontsize=12)
        ax.set_ylabel('RMSE Ratio (Outside/Inside)', fontsize=12)
        ax.set_title('RMSE Ratio by AD Method', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation zones
        ylim = ax.get_ylim()
        ax.axhspan(1.0, 1.5, alpha=0.1, color='green', zorder=0)
        ax.axhspan(1.5, 2.0, alpha=0.1, color='yellow', zorder=0)
        ax.axhspan(2.0, ylim[1], alpha=0.1, color='red', zorder=0)
        ax.set_ylim(ylim)
    
    def _plot_coverage_vs_performance_optimized(self, ax, df_results):
        """Optimized coverage vs performance plot with fixed color handling"""
        # Filter valid data
        valid_mask = ~df_results['rmse_ratio'].isna()
        coverage = df_results.loc[valid_mask, 'coverage'].values * 100
        rmse_ratio = df_results.loc[valid_mask, 'rmse_ratio'].values
        n_inside = df_results.loc[valid_mask, 'n_inside_ad'].values
        methods = df_results.loc[valid_mask, 'method'].values
        
        if len(coverage) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('AD Coverage vs Performance Trade-off')
            return
        
        # Handle color normalization edge cases
        vmin = rmse_ratio.min() if len(rmse_ratio) > 0 else 1.0
        vmax = rmse_ratio.max() if len(rmse_ratio) > 0 else 2.0
        if vmin >= vmax:
            vmax = vmin + 0.1
        
        # Create scatter plot with size based on sample count
        scatter = ax.scatter(coverage, rmse_ratio, 
                        s=np.sqrt(n_inside) * 20,  # Size based on sqrt of samples
                        c=rmse_ratio, cmap='RdYlGn_r', 
                        alpha=0.7, edgecolors='black', linewidth=1,
                        vmin=vmin, vmax=vmax)
        
        # Add trend line if enough data
        if len(coverage) > 2:
            try:
                # Robust polynomial fit
                valid_fit = np.isfinite(coverage) & np.isfinite(rmse_ratio)
                if valid_fit.sum() > 2:
                    z = np.polyfit(coverage[valid_fit], rmse_ratio[valid_fit], 2, 
                                 w=np.sqrt(n_inside[valid_fit]))
                    p = np.poly1d(z)
                    x_trend = np.linspace(coverage.min(), coverage.max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2,
                        label=f'Trend: y={z[0]:.3e}x²+{z[1]:.3f}x+{z[2]:.3f}')
            except:
                pass  # Skip trend line if fitting fails
        
        # Reference lines
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.5)
        ax.axvline(x=80, color='blue', linestyle=':', alpha=0.5)
        
        # Add method labels for outliers
        for i, (cov, ratio, method) in enumerate(zip(coverage, rmse_ratio, methods)):
            if ratio > 2.5 or ratio < 0.8 or cov < 50 or cov > 95:
                ax.annotate(method, (cov, ratio), fontsize=8, alpha=0.7)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RMSE Ratio', fontsize=11)
        
        ax.set_xlabel('AD Coverage (%)', fontsize=12)
        ax.set_ylabel('RMSE Ratio (Outside/Inside)', fontsize=12)
        ax.set_title('AD Coverage vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(coverage) > 2:
            ax.legend(fontsize=9)
    
    def _plot_sample_distribution_optimized(self, ax, df_results):
        """Optimized sample distribution plot"""
        methods = df_results['method'].values
        x = np.arange(len(methods))
        
        n_inside = df_results['n_inside_ad'].values
        n_outside = df_results['n_outside_ad'].values
        n_total = df_results['n_total'].values
        
        # Stacked bar plot
        bars1 = ax.bar(x, n_inside, label='Inside AD', color='green', alpha=0.7)
        bars2 = ax.bar(x, n_outside, bottom=n_inside, label='Outside AD', color='red', alpha=0.7)
        
        # Add R² values as text
        r2_inside = df_results['r2_inside_ad'].values
        r2_outside = df_results['r2_outside_ad'].values
        
        for i, (r2_in, r2_out) in enumerate(zip(r2_inside, r2_outside)):
            if not np.isnan(r2_in):
                ax.text(i, n_inside[i]/2, f'R²:{r2_in:.2f}', 
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            if not np.isnan(r2_out):
                ax.text(i, n_inside[i] + n_outside[i]/2, f'R²:{r2_out:.2f}', 
                       ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        ax.set_xlabel('AD Method', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Distribution and R² Values', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_error_metrics_optimized(self, ax, df_results):
        """Optimized error metrics comparison"""
        methods = df_results['method'].values
        x = np.arange(len(methods))
        
        # Get metrics
        error_reduction = df_results['error_reduction'].values
        mae_ratio = df_results['mae_ratio'].values
        
        # Replace NaN with 0
        error_reduction = np.nan_to_num(error_reduction)
        mae_ratio = np.nan_to_num(mae_ratio)
        
        # Create grouped bar plot
        width = 0.35
        bars1 = ax.bar(x - width/2, error_reduction, width, 
                       label='Error Reduction (%)', alpha=0.8)
        bars2 = ax.bar(x + width/2, mae_ratio * 10, width, 
                       label='MAE Ratio (×10)', alpha=0.8)
        
        # Color bars based on performance
        for bar, val in zip(bars1, error_reduction):
            if val > 30:
                bar.set_color('darkgreen')
            elif val > 20:
                bar.set_color('green')
            elif val > 10:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('AD Method', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Error Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add reference line
        ax.axhline(y=20, color='black', linestyle='--', alpha=0.5, label='20% Error Reduction')
    
    def _plot_performance_summary_optimized(self, ax, df_results, model_name):
        """Optimized performance summary text"""
        # Calculate summary statistics
        mean_rmse_ratio = df_results['rmse_ratio'].mean()
        mean_coverage = df_results['coverage'].mean()
        mean_error_reduction = df_results['error_reduction'].mean()
        
        best_method_idx = df_results['rmse_ratio'].idxmin()
        best_method = df_results.loc[best_method_idx, 'method']
        best_rmse_ratio = df_results.loc[best_method_idx, 'rmse_ratio']
        
        worst_method_idx = df_results['rmse_ratio'].idxmax()
        worst_method = df_results.loc[worst_method_idx, 'method']
        worst_rmse_ratio = df_results.loc[worst_method_idx, 'rmse_ratio']
        
        # Create summary text
        summary_text = f"""
AD PERFORMANCE SUMMARY
======================
Model: {model_name}
Number of AD Methods: {len(df_results)}

OVERALL PERFORMANCE:
• Mean RMSE Ratio: {mean_rmse_ratio:.3f}
• Mean Coverage: {mean_coverage:.3f}
• Mean Error Reduction: {mean_error_reduction:.1f}%

BEST METHOD: {best_method}
• RMSE Ratio: {best_rmse_ratio:.3f}
• Coverage: {df_results.loc[best_method_idx, 'coverage']:.3f}
• Error Reduction: {df_results.loc[best_method_idx, 'error_reduction']:.1f}%

WORST METHOD: {worst_method}
• RMSE Ratio: {worst_rmse_ratio:.3f}
• Coverage: {df_results.loc[worst_method_idx, 'coverage']:.3f}
• Error Reduction: {df_results.loc[worst_method_idx, 'error_reduction']:.1f}%

RECOMMENDATION:
"""
        
        # Add recommendation
        if mean_rmse_ratio < 1.5 and mean_coverage > 0.8:
            summary_text += "✓ Excellent AD definition!\n"
            summary_text += "The model shows strong reliability\n"
            summary_text += "within its applicability domain."
        elif mean_rmse_ratio < 2.0 and mean_coverage > 0.7:
            summary_text += "✓ Good AD definition.\n"
            summary_text += "Consider using the best methods\n"
            summary_text += "for production deployment."
        else:
            summary_text += "⚠ AD definition needs improvement.\n"
            summary_text += "Consider using more conservative\n"
            summary_text += "AD methods or expanding training data."
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        ax.axis('off')
    
    def create_combined_report(self, all_results: Dict[str, Dict[str, pd.DataFrame]], 
                             ad_mode: str = 'flexible'):
        """Create combined report with split-type and mode organization"""
        print(f"\n📊 Creating combined AD performance report for {ad_mode} mode...")
        
        # Organize results by split type
        split_organized = {split: {} for split in self.split_dirs.keys()}
        
        for dataset_name, dataset_results in all_results.items():
            for split_method, model_results in dataset_results.items():
                # Determine split type
                split_type = None
                for split in self.split_dirs.keys():
                    if split in split_method.lower() or split in dataset_name.lower():
                        split_type = split
                        break
                
                if split_type is None:
                    split_type = 'unknown'
                
                if split_type not in split_organized:
                    split_organized[split_type] = {}
                
                split_organized[split_type][f"{dataset_name}_{split_method}"] = model_results
        
        # Create reports for each split type
        for split_type, split_results in split_organized.items():
            if split_results:
                save_dir = self.split_dirs.get(split_type, self.performance_dir) / ad_mode
                save_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    self._create_split_specific_report(split_results, split_type, save_dir, ad_mode)
                except Exception as e:
                    print(f"  Warning: Failed to create {split_type} report: {e}")
        
        # Create overall combined report
        try:
            self._create_overall_combined_report(all_results, ad_mode)
        except Exception as e:
            print(f"  Warning: Failed to create overall report: {e}")
        
        # Clean up
        gc.collect()
        
        print(f"  ✓ Combined report created for {ad_mode} mode")
    
    def _create_split_specific_report(self, split_results: Dict, split_type: str, 
                                    save_dir: Path, ad_mode: str = 'flexible'):
        """Create report specific to a split type with mode"""
        print(f"\n  Creating {split_type.upper()} split report for {ad_mode} mode...")
        
        # Combine all results for this split
        all_dfs = []
        for dataset_name, model_results in split_results.items():
            for model_name, df in model_results.items():
                df_copy = df.copy()
                df_copy['dataset_split'] = dataset_name
                df_copy['model'] = model_name
                all_dfs.append(df_copy)
        
        if not all_dfs:
            return
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Method performance across datasets
        ax1 = axes[0, 0]
        method_summary = combined_df.groupby('method')['rmse_ratio'].agg(['mean', 'std']).reset_index()
        method_summary = method_summary.sort_values('mean')
        
        ax1.bar(method_summary['method'], method_summary['mean'], 
               yerr=method_summary['std'], capsize=5, alpha=0.8)
        ax1.set_xlabel('AD Method')
        ax1.set_ylabel('Mean RMSE Ratio')
        ax1.set_title(f'{split_type.upper()} Split: Method Performance')
        ax1.set_xticklabels(method_summary['method'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Coverage distribution
        ax2 = axes[0, 1]
        ax2.hist(combined_df['coverage'] * 100, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Coverage (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{split_type.upper()} Split: Coverage Distribution')
        ax2.axvline(combined_df['coverage'].mean() * 100, color='red', 
                   linestyle='--', label=f'Mean: {combined_df["coverage"].mean():.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. RMSE ratio distribution
        ax3 = axes[0, 2]
        ax3.hist(combined_df['rmse_ratio'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('RMSE Ratio')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{split_type.upper()} Split: RMSE Ratio Distribution')
        ax3.axvline(combined_df['rmse_ratio'].mean(), color='red', 
                   linestyle='--', label=f'Mean: {combined_df["rmse_ratio"].mean():.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Dataset comparison
        ax4 = axes[1, 0]
        dataset_summary = combined_df.groupby('dataset_split')['rmse_ratio'].mean().sort_values()
        if len(dataset_summary) > 10:
            dataset_summary = dataset_summary.head(10)
        
        ax4.barh(dataset_summary.index, dataset_summary.values, alpha=0.8)
        ax4.set_xlabel('Mean RMSE Ratio')
        ax4.set_ylabel('Dataset')
        ax4.set_title(f'{split_type.upper()} Split: Top Datasets')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Model comparison
        ax5 = axes[1, 1]
        model_summary = combined_df.groupby('model')['rmse_ratio'].agg(['mean', 'std']).reset_index()
        
        x = np.arange(len(model_summary))
        ax5.bar(x, model_summary['mean'], yerr=model_summary['std'], 
               capsize=5, alpha=0.8)
        ax5.set_xlabel('Model')
        ax5.set_ylabel('Mean RMSE Ratio')
        ax5.set_title(f'{split_type.upper()} Split: Model Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_summary['model'])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        summary_text = f"""
{split_type.upper()} SPLIT SUMMARY ({ad_mode} mode)
==============================
Total Analyses: {len(combined_df)}
Datasets: {combined_df['dataset_split'].nunique()}
AD Methods: {combined_df['method'].nunique()}
Models: {combined_df['model'].nunique()}

PERFORMANCE METRICS:
• Mean RMSE Ratio: {combined_df['rmse_ratio'].mean():.3f}
• Mean Coverage: {combined_df['coverage'].mean():.3f}
• Mean Error Reduction: {combined_df['error_reduction'].mean():.1f}%

BEST PERFORMING:
• Method: {method_summary.iloc[0]['method']}
  (RMSE Ratio: {method_summary.iloc[0]['mean']:.3f})
• Dataset: {dataset_summary.index[0]}
  (RMSE Ratio: {dataset_summary.iloc[0]:.3f})
"""
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        ax6.axis('off')
        
        plt.suptitle(f'{split_type.upper()} Split AD Performance Report ({ad_mode} mode)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure in split/mode directory (not dataset-specific)
        save_path = save_dir / f'{split_type}_split_report_{ad_mode}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save Excel report in split/mode directory
        excel_path = save_dir / f'{split_type}_split_report_{ad_mode}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            combined_df.to_excel(writer, sheet_name='All_Results', index=False)
            method_summary.to_excel(writer, sheet_name='Method_Summary', index=False)
            dataset_summary.to_frame('mean_rmse_ratio').to_excel(writer, sheet_name='Dataset_Summary')
            model_summary.to_excel(writer, sheet_name='Model_Summary', index=False)
        
        print(f"    ✓ {split_type.upper()} split report saved")
    
    def _create_overall_combined_report(self, all_results: Dict[str, Dict[str, pd.DataFrame]], 
                                      ad_mode: str = 'flexible'):
        """Create overall combined report"""
        # Combine all results
        all_dfs = []
        for dataset_name, dataset_results in all_results.items():
            for split_method, model_results in dataset_results.items():
                for model_name, df in model_results.items():
                    df_copy = df.copy()
                    df_copy['dataset'] = dataset_name
                    df_copy['split'] = split_method
                    df_copy['model'] = model_name
                    all_dfs.append(df_copy)
        
        if not all_dfs:
            return
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Create overall summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall method performance
        ax1 = axes[0, 0]
        method_summary = combined_df.groupby('method')['rmse_ratio'].agg(['mean', 'std', 'count']).reset_index()
        method_summary = method_summary.sort_values('mean').head(10)
        
        ax1.bar(method_summary['method'], method_summary['mean'], 
               yerr=method_summary['std'], capsize=5, alpha=0.8)
        ax1.set_xlabel('AD Method')
        ax1.set_ylabel('Mean RMSE Ratio')
        ax1.set_title('Top 10 AD Methods Overall')
        ax1.set_xticklabels(method_summary['method'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Split type comparison
        ax2 = axes[0, 1]
        split_summary = combined_df.groupby('split')['rmse_ratio'].agg(['mean', 'std']).reset_index()
        
        ax2.bar(split_summary['split'], split_summary['mean'], 
               yerr=split_summary['std'], capsize=5, alpha=0.8)
        ax2.set_xlabel('Split Type')
        ax2.set_ylabel('Mean RMSE Ratio')
        ax2.set_title('Performance by Split Type')
        ax2.set_xticklabels(split_summary['split'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Coverage vs RMSE Ratio scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(combined_df['coverage'] * 100, combined_df['rmse_ratio'], 
                            alpha=0.5, s=30)
        ax3.set_xlabel('Coverage (%)')
        ax3.set_ylabel('RMSE Ratio')
        ax3.set_title('Overall Coverage vs Performance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        summary_text = f"""
OVERALL AD PERFORMANCE SUMMARY
Mode: {ad_mode}
==============================
Total Analyses: {len(combined_df)}
Datasets: {combined_df['dataset'].nunique()}
Split Types: {combined_df['split'].nunique()}
AD Methods: {combined_df['method'].nunique()}
Models: {combined_df['model'].nunique()}

PERFORMANCE METRICS:
• Mean RMSE Ratio: {combined_df['rmse_ratio'].mean():.3f}
• Mean Coverage: {combined_df['coverage'].mean():.3f}
• Mean Error Reduction: {combined_df['error_reduction'].mean():.1f}%

BEST OVERALL:
• Method: {method_summary.iloc[0]['method']}
  (RMSE Ratio: {method_summary.iloc[0]['mean']:.3f})
"""
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
               verticalalignment='top', fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9))
        ax4.axis('off')
        
        plt.suptitle(f'Overall AD Performance Summary ({ad_mode} mode)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to performance_dir root with mode subdirectory
        save_path = self.performance_dir / ad_mode / f'overall_combined_report_{ad_mode}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Overall combined report saved for {ad_mode} mode")