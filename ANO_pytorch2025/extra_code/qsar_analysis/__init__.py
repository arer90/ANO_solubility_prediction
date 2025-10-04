__version__ = "1.0.0"
__author__ = "Seungjin Lee"

# Core imports
import os
import sys

# Apply visualization fixes immediately
try:
    # Module not found - commenting out for now
    # from .apply_all_mode_fixes import apply_global_matplotlib_fixes, patch_ad_visualizer
    # apply_global_matplotlib_fixes()
    # patch_ad_visualizer()
    pass
    
    # Apply comprehensive fixes (module not found - commenting out)
    # from .comprehensive_plot_fix import apply_comprehensive_fixes
    # apply_comprehensive_fixes()

    # Apply final definitive fixes (module not found - commenting out)
    # from .visualizations.ad_plots_final_fix import apply_final_fixes
    # apply_final_fixes()
    
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
                         skip_split_creation=False,
                         existing_splits=None,
                         enable_gc=True,
                         chunk_size=5000,
                         timeout_minutes=30,
                         test_size=0.2,
                         split_only_mode=False,
                         update_data_folder=True,
                         num_descriptors=7,
                         descriptor_list=None):
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
        timeout_minutes: Maximum runtime in minutes (default: 30)
        test_size: Fraction of data to use for testing (default: 0.2 for 80:20 split)
        split_only_mode: If True, only create splits and skip all analysis (default: False)
        update_data_folder: If True, update data/train and data/test folders (default: True)
        num_descriptors: Number of molecular descriptors to use (7 for fast, 49 for full, default: 7)
        descriptor_list: Custom list of descriptor names to use (overrides num_descriptors if provided)
    
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
    
    # Display system recommendations
    # Import gc for memory management
    import gc
    import os
    
    if show_recommendations:
        print_system_recommendations()
        
        # Auto-use recommended settings (skip user confirmation)
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
    
    # Performance mode configuration
    if performance_mode:
        DEFAULT_PARAMS['max_samples_analysis'] = max_samples
        DEFAULT_PARAMS['max_samples_ad'] = min(max_samples // 2, 500)
        DEFAULT_PARAMS['enable_sampling'] = True
        DEFAULT_PARAMS['sampling_ratio'] = 0.1
    
    # Set test_size parameter
    DEFAULT_PARAMS['test_size'] = test_size
    print(f"Using train/test split ratio: {int((1-test_size)*100)}:{int(test_size*100)}")
    
    # Set descriptor configuration
    from .config import DESCRIPTOR_NAMES_FAST, DESCRIPTOR_NAMES_FULL, ACTIVE_DESCRIPTORS
    import extra_code.qsar_analysis.config as qsar_config
    
    if descriptor_list is not None:
        # Process descriptor list (can be names or indices)
        processed_descriptors = []
        
        for item in descriptor_list:
            if isinstance(item, int):
                # If it's an index, get the descriptor name
                if 0 <= item < len(DESCRIPTOR_NAMES_FULL):
                    processed_descriptors.append(DESCRIPTOR_NAMES_FULL[item])
                    print(f"  Descriptor #{item}: {DESCRIPTOR_NAMES_FULL[item]}")
                else:
                    print(f"[WARNING] Invalid descriptor index {item}, skipping")
            elif isinstance(item, str):
                # If it's a name, use it directly
                processed_descriptors.append(item)
            else:
                print(f"[WARNING] Invalid descriptor item {item}, skipping")
        
        qsar_config.ACTIVE_DESCRIPTORS = processed_descriptors
        DEFAULT_PARAMS['num_descriptors'] = len(processed_descriptors)
        print(f"Using custom descriptor list with {len(processed_descriptors)} descriptors:")
        for i, desc in enumerate(processed_descriptors):
            print(f"  [{i}] {desc}")
    else:
        # Use predefined descriptor count
        if num_descriptors not in [7, 49]:
            print(f"[WARNING] Invalid num_descriptors={num_descriptors}. Must be 7 or 49. Using 7.")
            num_descriptors = 7
        
        if num_descriptors == 7:
            qsar_config.ACTIVE_DESCRIPTORS = DESCRIPTOR_NAMES_FAST
            print(f"Using {num_descriptors} fast molecular descriptors:")
            for i, desc in enumerate(DESCRIPTOR_NAMES_FAST):
                print(f"  [{i}] {desc}")
        else:
            qsar_config.ACTIVE_DESCRIPTORS = DESCRIPTOR_NAMES_FULL
            print(f"Using {num_descriptors} full molecular descriptors")
        
        DEFAULT_PARAMS['num_descriptors'] = num_descriptors
    
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
        max_samples_analysis=max_samples,
        split_only_mode=split_only_mode,
        update_data_folder=update_data_folder
    )
    
    # Set chunk size if provided
    if chunk_size:
        DEFAULT_PARAMS['chunk_size'] = chunk_size
    
    try:
        # Add timeout protection
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Analysis timed out after {timeout_minutes} minutes")
        
        # Set timeout (only on Unix systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_minutes * 60)
        
        try:
            # Run analysis with memory management
            analyzer.analyze_datasets(df_dict, test_only_datasets, ad_analysis_mode, skip_split_creation, existing_splits)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
        except TimeoutError as e:
            print(f"[ERROR] {e}")
            print("Consider reducing max_samples or enabling performance_mode")
            raise
        
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