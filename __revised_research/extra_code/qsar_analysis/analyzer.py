"""
QSAR Main Analyzer Module - STABLE CONCURRENT VERSION
"""
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import gc
import time
import psutil
import logging
import types
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import polars as pl

# Try to import joblib, but make it optional
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ Joblib not available, using ThreadPoolExecutor instead")

from .base import BaseAnalyzer
from .config import DEFAULT_PARAMS, RELIABILITY_SCORING_CONFIG
from .data_loader import DataLoader
from .splitters import DataSplitter
from .features import OptimizedFeatureCalculator
from .ad_methods import ConcurrentADMethods
from .statistics import StatisticalAnalyzer
from .visualizations import (
    ADVisualizer,
    MetaVisualizer, 
    StatisticalVisualizer, 
    SummaryVisualizer,
    ADPerformanceAnalyzer,
)
from .utils import (
    save_results_json, create_summary_excel, 
    generate_decision_report, print_analysis_summary,
    save_analysis_summary
)

try:
    from .config import (AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES, CHUNK_SIZE, 
                        RELIABILITY_SCORING_CONFIG, AD_METHODS, DEFAULT_PARAMS, AD_METHOD_MAX_SAMPLES)
except ImportError:
    from config import (AD_COVERAGE_STANDARDS, AD_COVERAGE_MODES, CHUNK_SIZE, 
                       RELIABILITY_SCORING_CONFIG, AD_METHODS, DEFAULT_PARAMS, AD_METHOD_MAX_SAMPLES)


class ConcurrentQSARAnalyzer(BaseAnalyzer):
    def __init__(self, output_dir: str = "result/1_preprocess", 
                 random_state: int = DEFAULT_PARAMS['random_state'],
                 performance_mode: bool = DEFAULT_PARAMS['performance_mode'],
                 max_samples_analysis: int = DEFAULT_PARAMS['max_samples_analysis'],
                 ad_mode: str = DEFAULT_PARAMS['ad_mode'],
                 enable_reliability_scoring: Optional[bool] = None,
                 n_jobs: int = -1):
        """Initialize with stable concurrent processing"""
        super().__init__(output_dir, random_state)
        
        self.performance_mode = performance_mode
        self.max_samples_analysis = max_samples_analysis
        self.ad_mode = ad_mode
        self.original_ad_mode = ad_mode  # Store original mode
        self.logger = logging.getLogger(__name__)
        
        # Parallel processing setup
        self.n_jobs = n_jobs if n_jobs != -1 else max(1, mp.cpu_count() - 1)
        
        # Choose backend based on availability
        if JOBLIB_AVAILABLE:
            self.parallel_backend = 'threading'
            self.batch_size = 'auto'
        else:
            self.parallel_backend = None
            self.batch_size = 1000
        
        print(f"🚀 Concurrent QSAR Analyzer Initialized")
        print(f"📈 Performance Mode: {'✓ Enabled' if performance_mode else '✗ Disabled'}")
        print(f"🔧 Parallel Jobs: {self.n_jobs}")
        print(f"🔧 Backend: {self.parallel_backend or 'ThreadPoolExecutor'}")
        print(f"💾 Max Samples: {max_samples_analysis}")
        print(f"💾 Memory Limit: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Reliability scoring setup
        if enable_reliability_scoring is None:
            self.enable_reliability_scoring = RELIABILITY_SCORING_CONFIG['enabled']
        else:
            self.enable_reliability_scoring = enable_reliability_scoring
        
        # Initialize components
        self.data_loader = DataLoader(
            output_dir, random_state, 
            performance_mode, max_samples_analysis
        )
        self.splitter = DataSplitter(self.output_dir, random_state)
        self.feature_calculator = OptimizedFeatureCalculator(
            performance_mode, n_jobs=self.n_jobs
        )
        self.ad_methods = ConcurrentADMethods(
            random_state, ad_mode=ad_mode, n_jobs=self.n_jobs
        )
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Initialize visualizers
        self.ad_visualizer = ADVisualizer(
            self.output_dir, ad_mode=ad_mode, n_jobs=self.n_jobs
        )
        self.meta_visualizer = MetaVisualizer(self.output_dir)
        self.stat_visualizer = StatisticalVisualizer(self.output_dir)
        self.summary_visualizer = SummaryVisualizer(self.output_dir)
        
        # Data storage
        self.datasets = {}
        self.splits = {}
        self.features = {}
        self.ad_analysis = {}
        self.statistical_results = {}
        
        # All mode results storage
        self.all_mode_results = {}
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
    def _track_memory(self, checkpoint: str):
        """Track memory usage at checkpoint"""
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.performance_stats['memory_usage'].append(current_mem)
        self.memory_monitor.checkpoint(checkpoint)
        gc.collect()
        
    def _ensure_calculate_similarity_method(self):
        """Ensure calculate_similarity method exists on feature calculator"""
        if not hasattr(self.feature_calculator, 'calculate_similarity'):
            def calculate_similarity_method(self, train_smiles, test_smiles):
                try:
                    from .features import calculate_similarity
                    return calculate_similarity(self, train_smiles, test_smiles)
                except ImportError:
                    return self._calculate_similarity_fallback(train_smiles, test_smiles)
            
            self.feature_calculator.calculate_similarity = types.MethodType(
                calculate_similarity_method, self.feature_calculator
            )
    
    def _calculate_features_sequential(self):
        """Sequential feature calculation"""
        for name in self.datasets.keys():
            try:
                print(f"    Processing {name}...", end='')
                start_time = time.time()
                
                feature_data = self.feature_calculator.calculate_features(
                    self.splits[name]['smiles']
                )
                
                if feature_data is not None:
                    self.features[name] = {
                        **feature_data,
                        'smiles': self.splits[name]['smiles']
                    }
                    print(f" ✓ ({feature_data['features'].shape}) in {time.time()-start_time:.1f}s")
                else:
                    print(f" ❌ Failed")
                    
            except Exception as e:
                print(f" ❌ Error: {str(e)}")

    
    def _calculate_features_safe(self):
        """Calculate features with fallback to sequential"""
        print("\n🧬 Step 3: Calculating molecular features")
        
        # Ensure calculate_similarity method exists
        self._ensure_calculate_similarity_method()
        
        if JOBLIB_AVAILABLE and self.parallel_backend and not self.performance_mode:
            try:
                self._calculate_features_parallel()
            except Exception as e:
                print(f"    ⚠️ Parallel feature calculation failed: {str(e)}")
                print("    Falling back to sequential processing...")
                self._calculate_features_sequential()
        else:
            self._calculate_features_sequential()
        
        print(f"✓ Features calculated for {len(self.features)} datasets")
        gc.collect()
    
    
    def _analyze_single_split_optimized(self, name: str, split_name: str, 
                                   split_data: dict, features: np.ndarray):
        try:
            if split_name == 'test_only':
                return self._perform_test_only_ad_analysis(name, split_data, features)
            
            # Get indices
            train_idx = split_data['train_idx']
            test_idx = split_data['test_idx']
            
            # 동적 샘플링 - cluster split은 특별 처리
            if split_name == 'cluster':
                # cluster split은 더 큰 테스트 세트를 허용
                max_train = 2000
                max_test = min(1000, len(test_idx))  # 최대 1000까지 허용
            else:
                max_train = 2000
                max_test = min(500, len(test_idx))
            
            # 메모리 체크
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                max_train = int(max_train * 0.5)
                max_test = int(max_test * 0.5)
                print(f"      ⚠️ High memory ({memory_percent}%), reducing to {max_train}/{max_test}")
            
            # 샘플링
            if len(train_idx) > max_train:
                np.random.seed(self.random_state)
                train_idx_sampled = np.random.choice(train_idx, max_train, replace=False)
            else:
                train_idx_sampled = train_idx
            
            if len(test_idx) > max_test:
                np.random.seed(self.random_state + 1)
                test_idx_sampled = np.random.choice(test_idx, max_test, replace=False)
            else:
                test_idx_sampled = test_idx
            
            # 실제 샘플링된 크기 기록
            actual_test_size = len(test_idx_sampled)
            
            X_train = features[train_idx_sampled].astype(np.float32)
            X_test = features[test_idx_sampled].astype(np.float32)
            
            print(f"      Sampled: {len(X_train)} train, {len(X_test)} test (actual test: {actual_test_size})")
            
            # AD 메소드 계산 - 실제 크기 전달
            ad_results = self.ad_methods.calculate_all_methods(
                X_train, X_test,
                enable_reliability=False
            )
            
            # 결과에 실제 크기 정보 추가
            if ad_results:
                for method, result in ad_results.items():
                    if result and isinstance(result, dict):
                        result['expected_test_size'] = actual_test_size
            
            # 메모리 정리
            del X_train, X_test
            gc.collect()
            
            # Consensus AD
            consensus_ad = self._calculate_simple_consensus(ad_results)
            
            return {
                'ad_results': ad_results,
                'consensus_ad': consensus_ad,
                'split_info': {
                    'train_size': len(train_idx_sampled),
                    'test_size': len(test_idx_sampled),
                    'actual_test_size': actual_test_size,
                    'type': split_name
                },
                'ad_mode': self.ad_mode
            }
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _perform_ad_analysis_safe(self):
        """Perform AD analysis with memory optimization"""
        print(f"\n🔬 Step 4: Performing AD analysis (mode: {self.ad_mode})")
        
        # Clear previous AD analysis results
        self.ad_analysis = {}
        
        # Prepare analysis tasks
        analysis_tasks = []
        for name in self.features:
            features = self.features[name]['features']
            dataset_splits = self.splits[name]['splits']
            
            for split_name, split_data in dataset_splits.items():
                if split_data is not None:
                    analysis_tasks.append((name, split_name, split_data, features))
        
        # Process sequentially to save memory
        for i, (name, split_name, split_data, features) in enumerate(analysis_tasks):
            try:
                print(f"    Processing {name}/{split_name}... ({i+1}/{len(analysis_tasks)})")
                
                # Check memory
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    print(f"    ⚠️ High memory usage: {memory_percent}%, running cleanup...")
                    gc.collect()
                    time.sleep(1)
                
                # Initialize storage
                if name not in self.ad_analysis:
                    self.ad_analysis[name] = {}
                
                # Perform AD analysis
                result = self._analyze_single_split_optimized(
                    name, split_name, split_data, features
                )
                
                self.ad_analysis[name][split_name] = result
                
                # Clean up after each analysis
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ {name}/{split_name} failed: {str(e)}")
                self.ad_analysis[name][split_name] = None
        
        print(f"    ✓ Completed {len(analysis_tasks)} AD analyses")
    
    def analyze_datasets(self, df_dict: Dict[str, pl.DataFrame], 
                        test_only_datasets: List[str] = None,
                        ad_analysis_mode: str = None):
        """
        Main analysis function - with stable error handling and 'all' mode support
        """
        if test_only_datasets is None:
            test_only_datasets = []
        
        if ad_analysis_mode is None:
            ad_analysis_mode = self.ad_mode
        
        print("\n🚀 STARTING CONCURRENT QSAR ANALYSIS")
        print("=" * 60)
        
        # Initialize performance stats
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.performance_stats = {
            'start_time': time.time(),
            'memory_usage': [initial_memory]
        }
        
        try:
            # Step-by-step execution with error handling
            with self.performance_tracker.track("Total Analysis"):
                
                # Step 1: Load and preprocess data
                with self.performance_tracker.track("Data Loading"):
                    self._load_and_preprocess_data(df_dict, test_only_datasets)
                
                self._track_memory("After data loading")
                
                # Step 2: Create and save splits
                with self.performance_tracker.track("Data Splitting"):
                    self._create_and_save_splits()
                
                self._track_memory("After splitting")
                self._cleanup_intermediate_data()
                
                # Step 3: Calculate features
                with self.performance_tracker.track("Feature Calculation"):
                    self._calculate_features_safe()
                
                self._track_memory("After feature calculation")
                
                # ===== Handle 'all' mode =====
                if ad_analysis_mode == 'all':
                    print("\n" + "="*60)
                    print("📊 Running analysis for ALL AD modes")
                    print("="*60)
                    
                    # Store original mode
                    original_mode = self.ad_mode
                    self.all_mode_results = {}
                    
                    # Analyze each mode
                    for mode in ['strict', 'flexible', 'adaptive']:
                        print(f"\n{'='*50}")
                        print(f"🔄 Analyzing with {mode.upper()} mode")
                        print(f"{'='*50}")
                        
                        # Update mode for all components
                        self.ad_mode = mode
                        self.ad_methods.ad_mode = mode
                        self.ad_methods._update_coverage_standards()
                        self.ad_visualizer.ad_mode = mode
                        self.ad_visualizer._update_coverage_standards()
                        
                        # AD analysis for this mode
                        with self.performance_tracker.track(f"AD Analysis ({mode})"):
                            self._perform_ad_analysis_safe()
                        
                        # Store results for this mode
                        self.all_mode_results[mode] = {
                            'ad_analysis': self.ad_analysis.copy(),
                            'mode_info': AD_COVERAGE_MODES[mode]
                        }
                        
                        self._track_memory(f"After AD analysis ({mode})")
                        
                        # Statistical analysis for this mode
                        with self.performance_tracker.track(f"Statistical Analysis ({mode})"):
                            self._perform_statistical_analysis()
                        
                        self._track_memory(f"After statistical analysis ({mode})")
                        
                        # Visualizations for this mode
                        with self.performance_tracker.track(f"Visualization ({mode})"):
                            self._create_visualizations_sequential(mode)
                            # AD visualizations
                            self.ad_visualizer.create_all_ad_visualizations(
                                self.ad_analysis, self.features
                            )
                        
                        self._track_memory(f"After visualization ({mode})")
                        
                        # Save results for this mode
                        with self.performance_tracker.track(f"Saving Results ({mode})"):
                            self._save_all_results()
                        
                        self._track_memory(f"After saving results ({mode})")
                        
                        # Clean up between modes
                        gc.collect()
                    
                    # Create mode comparison visualizations
                    print("\n" + "="*50)
                    print("📊 Creating mode comparison visualizations")
                    print("="*50)
                    
                    with self.performance_tracker.track("Mode Comparison"):
                        self._create_mode_comparison_visualizations()
                        self._create_all_modes_summary_report()
                    
                    # Restore original mode
                    self.ad_mode = original_mode
                    self.ad_methods.ad_mode = original_mode
                    self.ad_visualizer.ad_mode = original_mode
                    
                else:
                    # ===== Single mode analysis =====
                    # Step 4: AD analysis
                    with self.performance_tracker.track("AD Analysis"):
                        self._perform_ad_analysis_safe()
                        
                    self._track_memory("After AD analysis")
                    
                    # Step 5: Statistical analysis
                    with self.performance_tracker.track("Statistical Analysis"):
                        self._perform_statistical_analysis()
                        
                    self._track_memory("After statistical analysis")
                    
                    # Step 6: Visualizations
                    with self.performance_tracker.track("Visualization"):
                        self._create_visualizations_sequential(ad_analysis_mode)
                        
                    self._track_memory("After visualization")
                    
                    # Step 7: Save results
                    with self.performance_tracker.track("Saving Results"):
                        self._save_all_results()
                        
                    self._track_memory("After saving results")
            
            # Final report
            self._generate_final_report()
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            # Final cleanup
            self._final_cleanup()
    
    def _create_mode_comparison_visualizations(self):
        """Create visualizations comparing all AD modes"""
        comparison_path = self.output_dir / 'ad_analysis' / 'mode_comparison'
        comparison_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        # 1. Average coverage by mode
        ax1 = axes[0]
        mode_coverages = {}
        for mode, results in self.all_mode_results.items():
            coverages = []
            for dataset_name, dataset_data in results['ad_analysis'].items():
                for split_name, split_data in dataset_data.items():
                    if split_data and 'ad_results' in split_data:
                        for method_result in split_data['ad_results'].values():
                            if method_result and 'coverage' in method_result:
                                coverages.append(method_result['coverage'])
            mode_coverages[mode] = {
                'mean': np.mean(coverages) if coverages else 0,
                'std': np.std(coverages) if coverages else 0,
                'n': len(coverages)
            }
        
        modes = list(mode_coverages.keys())
        means = [mode_coverages[m]['mean'] for m in modes]
        stds = [mode_coverages[m]['std'] for m in modes]
        
        bars = ax1.bar(modes, means, yerr=stds, capsize=5)
        colors = {'strict': 'red', 'flexible': 'green', 'adaptive': 'blue'}
        for bar, mode in zip(bars, modes):
            bar.set_color(colors.get(mode, 'gray'))
        
        ax1.set_ylabel('Mean Coverage')
        ax1.set_title('Average AD Coverage by Mode')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 2. Coverage distribution by mode
        ax2 = axes[1]
        coverage_data = []
        for mode, results in self.all_mode_results.items():
            for dataset_name, dataset_data in results['ad_analysis'].items():
                for split_name, split_data in dataset_data.items():
                    if split_data and 'ad_results' in split_data:
                        for method, method_result in split_data['ad_results'].items():
                            if method_result and 'coverage' in method_result:
                                coverage_data.append({
                                    'mode': mode,
                                    'coverage': method_result['coverage'],
                                    'method': method
                                })
        
        if coverage_data:
            import pandas as pd
            df_coverage = pd.DataFrame(coverage_data)
            
            positions = {'strict': 0, 'flexible': 1, 'adaptive': 2}
            for mode in modes:
                mode_data = df_coverage[df_coverage['mode'] == mode]['coverage']
                pos = positions[mode]
                parts = ax2.violinplot([mode_data], positions=[pos], widths=0.8)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[mode])
                    pc.set_alpha(0.7)
            
            ax2.set_xticks(list(positions.values()))
            ax2.set_xticklabels(list(positions.keys()))
            ax2.set_ylabel('Coverage Distribution')
            ax2.set_title('Coverage Distribution by Mode')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Method performance across modes
        ax3 = axes[2]
        method_performance = {}
        for mode, results in self.all_mode_results.items():
            for dataset_name, dataset_data in results['ad_analysis'].items():
                for split_name, split_data in dataset_data.items():
                    if split_data and 'ad_results' in split_data:
                        for method, method_result in split_data['ad_results'].items():
                            if method_result and 'coverage' in method_result:
                                if method not in method_performance:
                                    method_performance[method] = {}
                                if mode not in method_performance[method]:
                                    method_performance[method][mode] = []
                                method_performance[method][mode].append(method_result['coverage'])
        
        # Plot top 5 methods
        methods = list(method_performance.keys())[:5]
        n_methods = len(methods)
        width = 0.25
        x = np.arange(n_methods)
        
        for i, mode in enumerate(modes):
            means = []
            for method in methods:
                if mode in method_performance.get(method, {}):
                    means.append(np.mean(method_performance[method][mode]))
                else:
                    means.append(0)
            ax3.bar(x + i * width, means, width, label=mode.title(), 
                   color=colors[mode], alpha=0.8)
        
        ax3.set_xlabel('AD Methods')
        ax3.set_ylabel('Mean Coverage')
        ax3.set_title('Method Performance Across Modes')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Dataset performance across modes
        ax4 = axes[3]
        dataset_performance = {}
        for mode, results in self.all_mode_results.items():
            for dataset_name, dataset_data in results['ad_analysis'].items():
                if dataset_name not in dataset_performance:
                    dataset_performance[dataset_name] = {}
                
                coverages = []
                for split_name, split_data in dataset_data.items():
                    if split_data and 'ad_results' in split_data:
                        for method_result in split_data['ad_results'].values():
                            if method_result and 'coverage' in method_result:
                                coverages.append(method_result['coverage'])
                
                if coverages:
                    dataset_performance[dataset_name][mode] = np.mean(coverages)
        
        datasets = list(dataset_performance.keys())
        n_datasets = len(datasets)
        x = np.arange(n_datasets)
        
        for i, mode in enumerate(modes):
            values = [dataset_performance[d].get(mode, 0) for d in datasets]
            ax4.bar(x + i * width, values, width, label=mode.title(), 
                   color=colors[mode], alpha=0.8)
        
        ax4.set_xlabel('Datasets')
        ax4.set_ylabel('Mean Coverage')
        ax4.set_title('Dataset Performance Across Modes')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(datasets, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Mode characteristics
        ax5 = axes[4]
        mode_info_text = ""
        for mode in modes:
            mode_info = self.all_mode_results[mode]['mode_info']
            mode_info_text += f"{mode.upper()} MODE:\n"
            mode_info_text += f"  Name: {mode_info['name']}\n"
            mode_info_text += f"  Reference: {mode_info['reference'][:50]}...\n\n"
        
        ax5.text(0.05, 0.95, mode_info_text, transform=ax5.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax5.axis('off')
        ax5.set_title('Mode Characteristics')
        
        # 6. Summary statistics
        ax6 = axes[5]
        summary_text = "SUMMARY STATISTICS:\n\n"
        for mode, stats in mode_coverages.items():
            summary_text += f"{mode.upper()}:\n"
            summary_text += f"  Mean Coverage: {stats['mean']:.3f}\n"
            summary_text += f"  Std Coverage: {stats['std']:.3f}\n"
            summary_text += f"  N Analyses: {stats['n']}\n\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3))
        ax6.axis('off')
        ax6.set_title('Summary Statistics')
        
        plt.suptitle('AD Mode Comparison Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(comparison_path / 'mode_comparison_summary.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        gc.collect()
        
        print("  ✓ Mode comparison visualizations created")
    
    def _create_all_modes_summary_report(self):
        """Create comprehensive summary report for all modes"""
        report_path = self.output_dir / 'ad_analysis' / 'all_modes_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE AD MODE COMPARISON REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Duration: {time.time() - self.performance_stats['start_time']:.1f}s\n")
            f.write("="*80 + "\n\n")
            
            # Mode comparison summary
            f.write("MODE COMPARISON SUMMARY:\n")
            f.write("-"*50 + "\n")
            
            for mode, results in self.all_mode_results.items():
                mode_info = results['mode_info']
                f.write(f"\n{mode.upper()} MODE:\n")
                f.write(f"  Full Name: {mode_info['name']}\n")
                f.write(f"  Reference: {mode_info['reference']}\n")
                
                # Calculate statistics
                all_coverages = []
                method_coverages = {}
                dataset_coverages = {}
                
                for dataset_name, dataset_data in results['ad_analysis'].items():
                    dataset_coverages[dataset_name] = []
                    for split_name, split_data in dataset_data.items():
                        if split_data and 'ad_results' in split_data:
                            for method, method_result in split_data['ad_results'].items():
                                if method_result and 'coverage' in method_result:
                                    coverage = method_result['coverage']
                                    all_coverages.append(coverage)
                                    dataset_coverages[dataset_name].append(coverage)
                                    
                                    if method not in method_coverages:
                                        method_coverages[method] = []
                                    method_coverages[method].append(coverage)
                
                # Overall statistics
                if all_coverages:
                    f.write(f"\n  Overall Statistics:\n")
                    f.write(f"    Mean Coverage: {np.mean(all_coverages):.3f}\n")
                    f.write(f"    Std Coverage: {np.std(all_coverages):.3f}\n")
                    f.write(f"    Min Coverage: {np.min(all_coverages):.3f}\n")
                    f.write(f"    Max Coverage: {np.max(all_coverages):.3f}\n")
                    f.write(f"    Total Analyses: {len(all_coverages)}\n")
                
                # Method performance
                f.write(f"\n  Method Performance:\n")
                for method, coverages in sorted(method_coverages.items(), 
                                              key=lambda x: np.mean(x[1]), reverse=True):
                    f.write(f"    {method}: {np.mean(coverages):.3f} ± {np.std(coverages):.3f}\n")
                
                # Dataset performance
                f.write(f"\n  Dataset Performance:\n")
                for dataset, coverages in sorted(dataset_coverages.items(), 
                                               key=lambda x: np.mean(x[1]) if x[1] else 0, 
                                               reverse=True):
                    if coverages:
                        f.write(f"    {dataset}: {np.mean(coverages):.3f} ± {np.std(coverages):.3f}\n")
            
            # Recommendations
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*50 + "\n")
            
            # Find best mode for different criteria
            mode_means = {}
            for mode, results in self.all_mode_results.items():
                coverages = []
                for dataset_name, dataset_data in results['ad_analysis'].items():
                    for split_name, split_data in dataset_data.items():
                        if split_data and 'ad_results' in split_data:
                            for method_result in split_data['ad_results'].values():
                                if method_result and 'coverage' in method_result:
                                    coverages.append(method_result['coverage'])
                mode_means[mode] = np.mean(coverages) if coverages else 0
            
            # Regulatory compliance
            if mode_means.get('strict', 0) >= 0.9:
                f.write("\n✓ STRICT mode meets regulatory requirements (≥90% coverage)\n")
                f.write("  Recommended for: Regulatory submissions, safety-critical applications\n")
            else:
                f.write("\n✗ STRICT mode does not meet regulatory requirements\n")
                f.write("  Consider: Expanding training data or adjusting model scope\n")
            
            # Research applications
            best_research_mode = max(['flexible', 'adaptive'], key=lambda x: mode_means.get(x, 0))
            f.write(f"\n✓ {best_research_mode.upper()} mode recommended for research\n")
            f.write(f"  Coverage: {mode_means[best_research_mode]:.3f}\n")
            
            # Coverage standards summary
            f.write("\n" + "="*80 + "\n")
            f.write("COVERAGE STANDARDS BY MODE:\n")
            f.write("-"*50 + "\n")
            
            for mode in ['strict', 'flexible', 'adaptive']:
                if mode in AD_COVERAGE_MODES:
                    f.write(f"\n{mode.upper()} mode standards:\n")
                    standards = AD_COVERAGE_MODES[mode].get('coverage_standards', {})
                    
                    if mode == 'adaptive':
                        # Handle nested structure for adaptive mode
                        for context, context_standards in standards.items():
                            if isinstance(context_standards, dict):
                                f.write(f"  {context}:\n")
                                for quality, bounds in context_standards.items():
                                    if isinstance(bounds, tuple):
                                        f.write(f"    {quality}: {bounds[0]:.2f} - {bounds[1]:.2f}\n")
                    else:
                        for quality, bounds in standards.items():
                            if isinstance(bounds, tuple):
                                f.write(f"  {quality}: {bounds[0]:.2f} - {bounds[1]:.2f}\n")
        
        print(f"  ✓ All modes summary report saved: {report_path.name}")
    
    def _load_and_preprocess_data(self, df_dict: Dict[str, pl.DataFrame], 
                                 test_only_datasets: List[str]):
        """Load and preprocess data"""
        print("\n📊 Step 1: Loading and preprocessing data")
        
        self.datasets = self.data_loader.load_and_preprocess_data(
            df_dict, test_only_datasets
        )
        self.splits = self.data_loader.splits
        
        # Validate data
        if not self.data_loader.validate_data_structure():
            raise ValueError("Data validation failed")
    
    def _create_and_save_splits(self):
        """Create and save splits"""
        print("\n📂 Step 2: Creating and saving data splits")
        
        for name, dataset_info in self.datasets.items():
            smiles = self.splits[name]['smiles']
            targets = self.splits[name]['targets']
            is_test_only = dataset_info['is_test_only']
            
            # Create splits
            splits = self.splitter.create_all_splits(
                name, smiles, targets, is_test_only
            )
            
            # Save splits
            self.splits[name]['splits'] = splits
            
            # Clean up
            del smiles, targets
            gc.collect()
    
    def analyze_ad_performance_for_all_datasets(self, features_dict: Dict, 
                                            targets_dict: Dict, 
                                            ad_analysis_dict: Dict,
                                            splits_dict: Dict):
        """AD performance 분석 - targets 전달 수정"""
        if not self.ad_visualizer.performance_analyzer:
            print("  ❌ AD performance analyzer not available")
            return None
        
        # targets_dict 생성
        targets_dict = {}
        for name in self.features:
            if name in self.splits and 'targets' in self.splits[name]:
                targets_dict[name] = self.splits[name]['targets']
        
        # AD performance analysis 실행
        return self.ad_visualizer.analyze_ad_performance_for_all_datasets(
            self.features,  # features_dict
            targets_dict,   # targets_dict
            self.ad_analysis,  # ad_analysis_dict
            self.splits     # splits_dict (전체 splits 정보 전달)
        )

    def _calculate_similarity_fallback(self, train_smiles, test_smiles):
        """Fallback Tanimoto similarity calculation without RDKit"""
        try:
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
            print(f"        Fallback similarity calculation failed: {str(e)}")
            return None
    
    def _calculate_features_parallel(self):
        """Parallel feature calculation using joblib"""
        def calculate_features_for_dataset(name, smiles):
            try:
                feature_data = self.feature_calculator.calculate_features(smiles)
                return name, feature_data
            except Exception as e:
                print(f"    ❌ {name} failed: {str(e)}")
                return name, None
        
        # Parallel processing with joblib
        with Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend) as parallel:
            results = parallel(
                delayed(calculate_features_for_dataset)(name, self.splits[name]['smiles'])
                for name in self.datasets.keys()
            )
        
        # Store results
        for name, feature_data in results:
            if feature_data is not None:
                self.features[name] = {
                    **feature_data,
                    'smiles': self.splits[name]['smiles']
                }
                print(f"    ✓ {name}: {feature_data['features'].shape}")
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis"""
        print("\n📈 Step 5: Performing statistical analysis")
        
        self.statistical_results = {}
        
        for name in self.features:
            targets = self.splits[name]['targets']
            descriptors = self.features[name]['descriptors']
            
            # Statistical analysis
            stats_results = self.statistical_analyzer.perform_statistical_analysis(
                name, targets, descriptors
            )
            
            self.statistical_results[name] = stats_results
        
        # Dataset comparison
        if len(self.splits) > 1:
            dataset_comparison = self.statistical_analyzer.compare_datasets(
                self.splits
            )
            self.statistical_results['dataset_comparison'] = dataset_comparison
    
    def _create_visualizations_sequential(self, ad_analysis_mode: str):
        """Create visualizations sequentially"""
        print("\n🎨 Step 6: Creating visualizations")
        
        # AD Visualizations
        if hasattr(self, 'ad_visualizer'):
            self.ad_visualizer.create_all_ad_visualizations(
                self.ad_analysis, self.features
            )
            
        # 2. Meta Visualizations - 추가해야 함!
        print("  Creating meta visualizations...")
        if hasattr(self, 'meta_visualizer'):
            self.meta_visualizer.create_all_meta_visualizations(
                self.datasets, self.splits, self.features, self.ad_analysis
            )
        
        # 3. Statistical Visualizations - 추가해야 함!
        print("  Creating statistical visualizations...")
        if hasattr(self, 'stat_visualizer'):
            self.stat_visualizer.create_all_statistical_visualizations(
                self.splits, self.features, self.statistical_results
            )
        
        # 4. Summary Visualizations - 추가해야 함!
        print("  Creating summary visualizations...")
        if hasattr(self, 'summary_visualizer'):
            self.summary_visualizer.create_all_summary_visualizations(
                self.datasets, self.splits, self.ad_analysis, self.statistical_results
            )
        
        # AD Performance Analysis
        print("\n  Checking AD Performance Analysis availability...")
        
        if hasattr(self, 'ad_visualizer') and hasattr(self.ad_visualizer, 'performance_analyzer'):
            if self.ad_visualizer.performance_analyzer:
                try:
                    print("  ✓ AD Performance Analyzer is available")
                    
                    # Prepare data
                    targets_dict = {}
                    splits_dict = {}
                    
                    for name in self.features:
                        if name in self.splits and 'targets' in self.splits[name]:
                            targets_dict[name] = self.splits[name]['targets']
                        
                        if name in self.splits:
                            splits_dict[name] = {}
                            if 'splits' in self.splits[name]:
                                for split_name, split_data in self.splits[name]['splits'].items():
                                    if split_data:
                                        splits_dict[name][split_name] = {
                                            'train_idx': split_data.get('train_idx', []),
                                            'test_idx': split_data.get('test_idx', [])
                                        }
                    
                    # Run AD performance analysis
                    ad_perf_results = self.ad_visualizer.analyze_ad_performance_for_all_datasets(
                        self.features,
                        targets_dict,
                        self.ad_analysis,
                        splits_dict
                    )
                    
                    if ad_perf_results:
                        print(f"  ✓ AD performance analysis completed")
                    else:
                        print("  ⚠️ AD performance analysis returned no results")
                        
                except Exception as e:
                    print(f"  ❌ AD performance analysis failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("  ⚠️ AD Performance Analyzer not available")
        else:
            print("  ⚠️ ad_visualizer or performance_analyzer not found")
        
                
              
    def _save_all_results(self):
        """Save all results with consistent paths"""
        print("\n💾 Step 7: Saving results")
        
        # Save JSON in by_dataset folder structure
        json_path = self.output_dir / 'ad_analysis' / 'by_dataset' / 'all_datasets_results' / f'ad_results_{self.ad_mode}.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        save_results_json(
            self.ad_analysis,
            json_path
        )
        
        # Create Excel summary
        create_summary_excel(
            self.output_dir, self.datasets, self.splits,
            self.ad_analysis, self.statistical_results,
            self.ad_mode
        )
        
        # Save analysis summary
        save_analysis_summary(
            self.output_dir, self.datasets,
            self.performance_stats['start_time'],
            self.performance_stats['memory_usage'],
            self.ad_mode,
            self.enable_reliability_scoring
        )
        
        # Generate decision report
        generate_decision_report(
            self.output_dir, self.ad_analysis, self.ad_mode
        )
    
    def _generate_final_report(self):
        """Generate final report"""
        # Get mode info
        if self.ad_mode in AD_COVERAGE_MODES:
            mode_info = AD_COVERAGE_MODES[self.ad_mode]
        else:
            mode_info = {
                'name': 'Custom',
                'reference': 'User-defined standards'
            }
        
        print("\n" + "="*60)
        print("📊 ANALYSIS MODE INFORMATION:")
        print(f"  • {mode_info['name']} AD standards")
        print(f"  • Reference: {mode_info['reference']}")
        print("="*60)
        
        print_analysis_summary(
            self.performance_stats['start_time'],
            self.datasets,
            self.performance_stats['memory_usage'],
            self.output_dir,
            self.ad_mode,
            self.enable_reliability_scoring
        )
        
        # Performance report
        self.performance_tracker.print_report()
        
        # Memory report
        self.memory_monitor.print_report()
    
    def _cleanup_intermediate_data(self):
        """Clean up intermediate data"""
        # Remove unnecessary data
        if hasattr(self, 'data_loader'):
            if hasattr(self.data_loader, 'datasets'):
                del self.data_loader.datasets
            if hasattr(self.data_loader, 'splits'):
                del self.data_loader.splits
        
        # Clear caches
        if hasattr(self, 'feature_calculator') and hasattr(self.feature_calculator, '_cache'):
            self.feature_calculator._cache.clear()
        
        gc.collect()
    
    def _final_cleanup(self):
        """Final memory cleanup"""
        print("\n🧹 Final memory cleanup...")
        
        # Clear all caches
        if hasattr(self.feature_calculator, '_cache'):
            self.feature_calculator._cache.clear()
        
        if hasattr(self.ad_methods, '_cache'):
            self.ad_methods._cache.clear()
        
        # Close matplotlib
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Final garbage collection
        gc.collect()
        
        print("✓ Memory cleanup completed")
    
    def _analyze_single_split_memory_optimized(self, name: str, split_name: str, 
                                          split_data: dict, features: np.ndarray):
        """메모리 최적화된 단일 분할 AD 분석"""
        try:
            if split_name == 'test_only':
                return self._perform_test_only_ad_analysis(name, split_data, features)
            
            # Train/test indices
            train_idx = split_data['train_idx']
            test_idx = split_data['test_idx']
            
            # 매우 공격적인 샘플링
            max_train = 2000
            max_test = 500
            
            # 메모리 체크
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                max_train = 1000
                max_test = 300
                print(f"      ⚠️ High memory usage ({memory_percent}%), reducing samples")
            
            if len(train_idx) > max_train:
                np.random.seed(self.random_state)
                train_idx_sampled = np.random.choice(train_idx, max_train, replace=False)
            else:
                train_idx_sampled = train_idx
            
            if len(test_idx) > max_test:
                np.random.seed(self.random_state + 1)
                test_idx_sampled = np.random.choice(test_idx, max_test, replace=False)
            else:
                test_idx_sampled = test_idx
            
            # 데이터 타입 최적화
            X_train = features[train_idx_sampled].astype(np.float32)
            X_test = features[test_idx_sampled].astype(np.float32)
            
            print(f"      Sampled: {len(X_train)} train, {len(X_test)} test")
            
            # AD 메소드 계산
            ad_results = self.ad_methods.calculate_all_methods(
                X_train, X_test,
                enable_reliability=False  # Reliability scoring 비활성화 (메모리 절약)
            )
            
            # 메모리 정리
            del X_train, X_test
            gc.collect()
            
            # Consensus AD (간단한 버전)
            consensus_ad = self._calculate_simple_consensus(ad_results)
            
            return {
                'ad_results': ad_results,
                'consensus_ad': consensus_ad,
                'split_info': {
                    'train_size': len(train_idx_sampled),
                    'test_size': len(test_idx_sampled),
                    'type': split_name
                },
                'ad_mode': self.ad_mode
            }
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            return None
    
    def _perform_test_only_ad_analysis(self, name: str, split_data: dict, 
                                     features: np.ndarray):
        """Test-only AD analysis"""
        X_test = features[split_data['test_idx']]
        
        # Internal train/test split
        if len(X_test) > 10:
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X_test))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.2, random_state=self.random_state
            )
            
            X_train_internal = X_test[train_idx]
            X_test_internal = X_test[test_idx]
            
            # AD methods
            ad_results = self.ad_methods.calculate_all_methods(
                X_train_internal, X_test_internal,
                enable_reliability=self.enable_reliability_scoring
            )
            
            # Consensus
            consensus_ad = self._calculate_simple_consensus(ad_results)
            
            return {
                'ad_results': ad_results,
                'consensus_ad': consensus_ad,
                'split_info': {
                    'train_size': 0,
                    'test_size': len(X_test),
                    'type': 'test_only'
                },
                'ad_mode': self.ad_mode
            }
        
        return None
    
    def _calculate_simple_consensus(self, ad_results: Dict) -> Dict:
        """Simple consensus AD calculation"""
        valid_results = [r for r in ad_results.values() if r and 'in_ad' in r]
        
        if not valid_results:
            return None
        
        # Get sample size
        n_samples = len(valid_results[0]['in_ad'])
        
        # Majority vote
        votes = np.zeros(n_samples)
        for result in valid_results:
            votes += np.array(result['in_ad'])
        
        majority_vote = votes >= (len(valid_results) / 2)
        
        return {
            'majority_vote': {
                'in_ad': majority_vote.tolist(),
                'coverage': float(np.mean(majority_vote))
            }
        }


class MemoryMonitor:
    """Memory monitoring tool"""
    
    def __init__(self, threshold_mb: int = 3000):
        self.checkpoints = []
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.threshold_mb = threshold_mb
        self.warnings = []
    
    def checkpoint(self, label: str, force_gc: bool = False):
        """Memory checkpoint with automatic cleanup"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        delta = current_memory - self.initial_memory
        
        self.checkpoints.append({
            'label': label,
            'memory_mb': current_memory,
            'delta_mb': delta
        })
        
        # Warn and clean if threshold exceeded
        if current_memory > self.threshold_mb:
            warning = f"⚠️ High memory usage at {label}: {current_memory:.1f} MB"
            print(warning)
            self.warnings.append(warning)
            gc.collect()
            
            # Check memory after cleanup
            new_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if new_memory < current_memory:
                print(f"   ✓ Freed {current_memory - new_memory:.1f} MB")
        
        elif force_gc:
            gc.collect()
            
    def print_report(self):
        """Print memory usage report"""
        print("\n💾 MEMORY USAGE REPORT:")
        print("=" * 50)
        
        if not self.checkpoints:
            print("No memory checkpoints recorded.")
            return
        
        print(f"Initial memory: {self.initial_memory:.1f} MB")
        print(f"Threshold: {self.threshold_mb} MB")
        print("\nCheckpoints:")
        
        max_memory = self.initial_memory
        max_checkpoint = None
        
        for checkpoint in self.checkpoints:
            print(f"  • {checkpoint['label']}: {checkpoint['memory_mb']:.1f} MB "
                f"(+{checkpoint['delta_mb']:.1f} MB)")
            
            if checkpoint['memory_mb'] > max_memory:
                max_memory = checkpoint['memory_mb']
                max_checkpoint = checkpoint['label']
        
        # Final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_delta = final_memory - self.initial_memory
        
        print(f"\nFinal memory: {final_memory:.1f} MB")
        print(f"Total increase: {total_delta:.1f} MB")
        
        if max_checkpoint:
            print(f"Peak memory: {max_memory:.1f} MB at '{max_checkpoint}'")
        
        # Print warnings
        if self.warnings:
            print(f"\n⚠️ Memory warnings ({len(self.warnings)}):")
            for warning in self.warnings[-5:]:  # Show last 5 warnings
                print(f"  {warning}")


class PerformanceTracker:
    """Performance tracking tool"""
    
    def __init__(self):
        self.timings = {}
        self.current_task = None
        self.start_time = None
    
    def track(self, task_name: str):
        """Context manager for time tracking"""
        return self.TaskTimer(self, task_name)
    
    class TaskTimer:
        def __init__(self, tracker, task_name):
            self.tracker = tracker
            self.task_name = task_name
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            self.tracker.timings[self.task_name] = elapsed
    
    def print_report(self):
        """Print performance report"""
        print("\n⏱️ PERFORMANCE REPORT:")
        print("=" * 50)
        
        total_time = self.timings.get("Total Analysis", 0)
        
        for task, elapsed in self.timings.items():
            if task != "Total Analysis":
                percentage = (elapsed / total_time * 100) if total_time > 0 else 0
                print(f"{task}: {elapsed:.2f}s ({percentage:.1f}%)")
        
        print(f"\nTotal time: {total_time:.2f}s")


# Maintain backward compatibility
SequentialQSARAnalyzer = ConcurrentQSARAnalyzer