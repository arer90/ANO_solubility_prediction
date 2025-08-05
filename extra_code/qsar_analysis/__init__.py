__version__ = "1.0.0"
__author__ = "Seungjin Lee"

# Core imports
import os
import sys

# Apply visualization fixes immediately
try:
    from .apply_all_mode_fixes import apply_global_matplotlib_fixes, patch_ad_visualizer
    apply_global_matplotlib_fixes()
    patch_ad_visualizer()
    
    # Apply comprehensive fixes
    from .comprehensive_plot_fix import apply_comprehensive_fixes
    apply_comprehensive_fixes()
    
    # Apply final definitive fixes
    from .visualizations.ad_plots_final_fix import apply_final_fixes
    apply_final_fixes()
    
    print("[OK] All visualization fixes applied for all AD modes")
except Exception as e:
    print(f"[WARNING] Could not apply visualization fixes: {e}")

# Main analyzer class - Import carefully to avoid circular imports
try:
    from .analyzer import ConcurrentQSARAnalyzer
    # Backward compatibility
    SequentialQSARAnalyzer = ConcurrentQSARAnalyzer
    print("Successfully imported ConcurrentQSARAnalyzer")
except ImportError as e:
    print(f"Failed to import ConcurrentQSARAnalyzer: {e}")
    ConcurrentQSARAnalyzer = None
    SequentialQSARAnalyzer = None

# Configuration
try:
    from .config import (
        AD_METHODS,
        AD_COVERAGE_STANDARDS,
        AD_COVERAGE_MODES,
        TANIMOTO_STANDARDS,
        DESCRIPTOR_NAMES,
        DEFAULT_PARAMS,
        RELIABILITY_SCORING_CONFIG,
        SYSTEM_INFO,
        print_system_recommendations
    )
    print("Successfully imported configuration")
except ImportError as e:
    print(f"Failed to import configuration: {e}")
    raise

def run_enhanced_analysis(df_dict=None, test_only_datasets=None, 
                         output_dir="result/1_preprocess", 
                         performance_mode=True,  
                         ad_mode='flexible',
                         enable_reliability_scoring=False,
                         ad_analysis_mode=None,
                         max_samples=1000,
                         show_recommendations=True,
                         temp_dir=None,
                         enable_gc=True,
                         chunk_size=5000):
    """
    Run enhanced QSAR analysis with complete implementation
    
    Args:
        df_dict: Dictionary of Polars DataFrames with SMILES and logS values
        test_only_datasets: List of dataset names to treat as test-only
        output_dir: Output directory for results
        performance_mode: Enable performance optimizations
        ad_mode: AD coverage mode ('strict', 'flexible', 'adaptive')
        enable_reliability_scoring: Enable reliability scoring
        ad_analysis_mode: 'all' to analyze all modes, or None for single mode
        max_samples: Maximum samples for analysis (default: 1000)
        show_recommendations: Show system recommendations
        temp_dir: Temporary directory for intermediate files
        enable_gc: Enable garbage collection
        chunk_size: Chunk size for processing
    
    Returns:
        ConcurrentQSARAnalyzer: Analyzer instance with results
    
    Example:
        >>> import polars as pl
        >>> # Prepare QSAR data with SMILES and logS
        >>> df = pl.DataFrame({
        ...     'target_x': ['CCO', 'CC(C)O', 'CCCO'],  # SMILES
        ...     'target_y': [-0.77, -0.92, -0.66]       # logS values
        ... })
        >>> analyzer = run_enhanced_analysis({'dataset1': df})
    """
    if test_only_datasets is None:
        test_only_datasets = []
    
    # Check if analyzer is available
    if ConcurrentQSARAnalyzer is None:
        raise ImportError("ConcurrentQSARAnalyzer could not be imported. Please check your installation.")
    
    # 시스템 추천사항 표시
    # Import gc for memory management
    import gc
    import os
    
    if show_recommendations:
        print_system_recommendations()
        
        # 사용자에게 권장 설정 사용 여부 확인 - 자동 사용
        # use_recommended = input("\nUse recommended settings? [Y/n]: ").strip().lower()
        use_recommended = 'y'  # Automatically use recommended settings
        if use_recommended != 'n':
            if max_samples and max_samples != float('inf'):
                max_samples = min(max_samples, SYSTEM_INFO['recommended_max_samples'])
            DEFAULT_PARAMS['max_samples_ad'] = min(
                DEFAULT_PARAMS['max_samples_ad'], 
                SYSTEM_INFO['recommended_max_ad_samples']
            )
            DEFAULT_PARAMS['n_jobs'] = SYSTEM_INFO['max_workers']
            print(f"Using recommended settings: max_samples={max_samples}")
    
    # 성능 모드 설정
    if performance_mode:
        DEFAULT_PARAMS['max_samples_analysis'] = max_samples
        DEFAULT_PARAMS['max_samples_ad'] = min(max_samples // 2, 500)
        DEFAULT_PARAMS['enable_sampling'] = True
        DEFAULT_PARAMS['sampling_ratio'] = 0.1
    
    # Set up temp directory if provided
    if temp_dir:
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
    
    # Enable garbage collection
    if enable_gc:
        gc.enable()
        gc.collect()
    
    # Create analyzer with memory management options
    analyzer = ConcurrentQSARAnalyzer(
        output_dir=output_dir, 
        performance_mode=performance_mode,
        ad_mode=ad_mode,
        enable_reliability_scoring=enable_reliability_scoring,
        max_samples_analysis=max_samples
    )
    
    # Set chunk size if provided
    if chunk_size:
        DEFAULT_PARAMS['chunk_size'] = chunk_size
    
    try:
        # Run analysis with memory management
        analyzer.analyze_datasets(df_dict, test_only_datasets, ad_analysis_mode)
        
        # Force garbage collection after analysis
        if enable_gc:
            gc.collect()
        
        return analyzer
        
    finally:
        # Cleanup
        if enable_gc:
            gc.collect()

# Import new modules
try:
    from .metrics import QSARMetrics
    print("Successfully imported QSARMetrics")
except ImportError as e:
    print(f"Failed to import QSARMetrics: {e}")
    QSARMetrics = None

# Export all important classes and functions
__all__ = [
    # Main class
    'ConcurrentQSARAnalyzer',
    'SequentialQSARAnalyzer',
    
    # Metrics
    'QSARMetrics',
    
    # Configuration
    'AD_METHODS',
    'AD_COVERAGE_STANDARDS',
    'AD_COVERAGE_MODES',
    'TANIMOTO_STANDARDS',
    'DESCRIPTOR_NAMES',
    'DEFAULT_PARAMS',
    'RELIABILITY_SCORING_CONFIG',
    'SYSTEM_INFO',
    'print_system_recommendations',
    
    # Main function
    'run_enhanced_analysis'
]