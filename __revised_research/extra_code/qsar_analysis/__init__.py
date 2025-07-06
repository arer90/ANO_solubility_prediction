__version__ = "1.0.0"
__author__ = "Seungjin Lee"

# Core imports
import os
import sys

# Main analyzer class - Import carefully to avoid circular imports
try:
    from .analyzer import ConcurrentQSARAnalyzer
    # Backward compatibility
    SequentialQSARAnalyzer = ConcurrentQSARAnalyzer
    print("✓ Successfully imported ConcurrentQSARAnalyzer")
except ImportError as e:
    print(f"⚠️ Failed to import ConcurrentQSARAnalyzer: {e}")
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
    print("✓ Successfully imported configuration")
except ImportError as e:
    print(f"⚠️ Failed to import configuration: {e}")
    raise

def run_enhanced_analysis(df_dict=None, test_only_datasets=None, 
                         output_dir="result/1_preprocess", 
                         performance_mode=True,  
                         ad_mode='flexible',
                         enable_reliability_scoring=False,
                         ad_analysis_mode=None,
                         max_samples=1000,
                         show_recommendations=True):
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
    if show_recommendations:
        print_system_recommendations()
        
        # 사용자에게 권장 설정 사용 여부 확인
        use_recommended = input("\nUse recommended settings? [Y/n]: ").strip().lower()
        if use_recommended != 'n':
            max_samples = min(max_samples, SYSTEM_INFO['recommended_max_samples'])
            DEFAULT_PARAMS['max_samples_ad'] = min(
                DEFAULT_PARAMS['max_samples_ad'], 
                SYSTEM_INFO['recommended_max_ad_samples']
            )
            DEFAULT_PARAMS['n_jobs'] = SYSTEM_INFO['max_workers']
            print(f"✓ Using recommended settings: max_samples={max_samples}")
    
    # 성능 모드 설정
    if performance_mode:
        DEFAULT_PARAMS['max_samples_analysis'] = max_samples
        DEFAULT_PARAMS['max_samples_ad'] = min(max_samples // 2, 500)
        DEFAULT_PARAMS['enable_sampling'] = True
        DEFAULT_PARAMS['sampling_ratio'] = 0.1
    
    # Create analyzer
    analyzer = ConcurrentQSARAnalyzer(
        output_dir=output_dir, 
        performance_mode=performance_mode,
        ad_mode=ad_mode,
        enable_reliability_scoring=enable_reliability_scoring,
        max_samples_analysis=max_samples
    )
    
    # Run analysis
    analyzer.analyze_datasets(df_dict, test_only_datasets, ad_analysis_mode)
    
    return analyzer

# Export all important classes and functions
__all__ = [
    # Main class
    'ConcurrentQSARAnalyzer',
    'SequentialQSARAnalyzer',
    
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