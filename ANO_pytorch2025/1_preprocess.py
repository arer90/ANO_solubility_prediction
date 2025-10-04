#!/usr/bin/env python3
"""
Data Preprocessing and Splitting - ANO Framework Module 1
=========================================================

PURPOSE:
Module 1 prepares all molecular datasets for the ANO framework by creating
standardized train/test splits and performing data quality analysis.

KEY FEATURES:
1. **Dataset Processing**: WS, DE, LO, HU training sets + test-only datasets
2. **9 Split Methods**: rm, ac, cl, cs, en, pc, sa, sc, ti
3. **Standardized Output**: Consistent format for all downstream modules
4. **Applicability Domain**: Statistical analysis and outlier detection
5. **Data Validation**: SMILES validation, duplicate removal, sanitization
6. **Visualization**: Distribution plots, correlation matrices, split analysis

RECENT UPDATES (2024):
- Added 9 different splitting strategies for robust evaluation
- Standardized output format for all modules
- Added test-only datasets (FreeSolv, Lipophilicity, AqSolDB, BigSolDB)
- Improved SMILES validation and molecular standardization
- Enhanced applicability domain analysis

SPLIT METHODS:
- rm: Random splitting (80/20)
- ac: Activity cliff splitting
- cl: Cluster-based splitting
- cs: Chemical space splitting
- en: Ensemble splitting
- pc: Physicochemical property splitting
- sa: Solubility-aware splitting
- sc: Scaffold-based splitting
- ti: Time-based splitting

OUTPUT STRUCTURE:
data/
├── train/{split}/
│   └── {split}_{dataset}_train.csv
├── test/{split}/
│   └── {split}_{dataset}_test.csv
└── test_only/
    └── {dataset}.csv

result/1_preprocess/
├── split_statistics.json
├── applicability_domain/
│   └── ad_analysis.json
└── plots/
    ├── distribution_plots.png
    └── split_comparison.png

USAGE:
python 1_preprocess.py [options]
  --datasets: Datasets to process (ws/de/lo/hu/all)
  --splits: Split methods to use (rm/ac/cl/etc. or all)
  --test-size: Test set proportion (default: 0.2)
  --ad-mode: Applicability domain mode (strict/flexible/adaptive)
  --analyze-only: Only analyze existing data
  --split-only: Only create splits (skip analysis)

Performance monitoring mode:
  python 1_preprocess.py --monitor-performance --log-level debug

Supported Datasets:
------------------
- **WS (Water Solubility)**: ~9,000 compounds with experimental solubility values
- **DE (Density)**: ~1,100 compounds with density measurements
- **LO (LogS)**: ~1,300 compounds with LogS values
- **HU (Human)**: ~600+ compounds with human-relevant endpoints

Output Structure:
----------------
data/
├── train/           # Training datasets
│   ├── ws_train.csv
│   ├── de_train.csv
│   └── ...
├── test/            # Test datasets
│   ├── ws_test.csv
│   ├── de_test.csv
│   └── ...
└── analysis/        # Analysis results
    ├── statistics/
    ├── visualizations/
    └── reports/
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import time
import json
import psutil
import traceback
from datetime import datetime

# Performance Monitor class for 1_preprocess.py
class PerformanceMonitor:
    """
    Monitor and track performance metrics during data preprocessing operations.

    This class provides comprehensive monitoring of computational resources including
    memory usage, execution time, CPU utilization, and generates detailed performance
    reports for optimization and debugging purposes.

    Features:
    ---------
    - Real-time memory monitoring with peak usage tracking
    - Execution time measurement with checkpoint support
    - CPU utilization monitoring
    - Automatic log generation with timestamps
    - JSON performance reports for analysis
    - Memory optimization suggestions

    Example Usage:
    -------------
    # Basic performance monitoring
    monitor = PerformanceMonitor(1, "preprocess", "logs/performance")
    monitor.start()

    # Process data with checkpoints
    monitor.checkpoint("Loading datasets")
    data = load_datasets()

    monitor.checkpoint("Splitting data")
    train, test = split_data(data)

    monitor.checkpoint("Analysis complete")
    monitor.stop()

    # Generate performance report
    report = monitor.generate_report()
    print(f"Peak memory: {report['peak_memory_mb']:.1f} MB")
    print(f"Total time: {report['total_time']:.2f} seconds")

    Performance Metrics Tracked:
    ---------------------------
    - Initial memory baseline
    - Peak memory usage during processing
    - Memory delta (increase from baseline)
    - Execution time with sub-process timing
    - Checkpoint timestamps for bottleneck identification
    - CPU utilization patterns
    """

    def __init__(self, code_num: int, code_name: str, result_dir: str):
        """
        Initialize performance monitor with comprehensive tracking capabilities.

        Args:
            code_num (int): Script number identifier (e.g., 1 for 1_preprocess.py)
            code_name (str): Descriptive script name (e.g., "preprocess", "optimization")
            result_dir (str): Directory path to save performance logs and reports

        Example:
            monitor = PerformanceMonitor(1, "preprocess", "logs/preprocessing")
            # Creates: logs/preprocessing/1_preprocess_log.txt
            #          logs/preprocessing/1_preprocess_performance.json
        """
        self.code_num = code_num
        self.code_name = code_name
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.initial_memory = None
        self.max_memory = 0
        self.process = psutil.Process()

        # Logging
        self.log_file = self.result_dir / f"{code_num}_{code_name}_log.txt"
        self.performance_file = self.result_dir / f"{code_num}_{code_name}_performance.json"

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"ANO Framework - {code_name.title()} Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.max_memory = self.initial_memory
        self.log_progress(f"[MONITOR] Performance monitoring started")
        self.log_progress(f"[MONITOR] Initial memory: {self.initial_memory:.2f} MB")

    def log_progress(self, message: str, level: str = "info"):
        """
        Log a progress message.

        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Update max memory
        if current_memory > self.max_memory:
            self.max_memory = current_memory

        # Format message
        if level == "error":
            log_line = f"[{timestamp}] [ERROR] {message}"
        elif level == "warning":
            log_line = f"[{timestamp}] [WARN] {message}"
        else:
            log_line = f"[{timestamp}] [INFO] {message}"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
            f.flush()

        # Also print to console
        print(log_line)

    def stop(self) -> dict:
        """
        Stop performance monitoring and return statistics.

        Returns:
            Dictionary containing performance statistics
        """
        self.end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Calculate statistics
        duration = self.end_time - self.start_time if self.start_time else 0
        memory_delta = final_memory - self.initial_memory if self.initial_memory else 0

        stats = {
            "code_num": self.code_num,
            "code_name": self.code_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration,
            "duration_formatted": self._format_duration(duration),
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "max_memory_mb": self.max_memory,
            "memory_delta_mb": memory_delta,
            "timestamp": datetime.now().isoformat()
        }

        # Log final statistics
        self.log_progress(f"[MONITOR] Performance monitoring stopped")
        self.log_progress(f"[MONITOR] Duration: {stats['duration_formatted']}")
        self.log_progress(f"[MONITOR] Memory usage - Initial: {self.initial_memory:.2f} MB, "
                         f"Max: {self.max_memory:.2f} MB, Final: {final_memory:.2f} MB")
        self.log_progress(f"[MONITOR] Memory delta: {memory_delta:+.2f} MB")

        # Save performance data
        with open(self.performance_file, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in human readable format.

        Args:
            seconds (float): Duration in seconds

        Returns:
            str: Formatted duration (e.g., "1h 23m 45.67s")
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"

    def checkpoint(self, message: str):
        """
        Log a checkpoint with current performance metrics.

        Args:
            message (str): Checkpoint description message
        """
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        if self.start_time:
            elapsed = current_time - self.start_time
            self.log_progress(f"[CHECKPOINT] {message} - Elapsed: {self._format_duration(elapsed)}, "
                             f"Memory: {current_memory:.2f} MB")
        else:
            self.log_progress(f"[CHECKPOINT] {message} - Memory: {current_memory:.2f} MB")


# Import required modules
try:
    from extra_code.preprocess import load_data
    from extra_code.qsar_analysis import run_enhanced_analysis
    print("Successfully imported preprocessing modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure extra_code directory exists with preprocess.py and qsar_analysis modules")

# Import from config
try:
    from config import (
        DATASETS, DATASET_NAME_TO_KEY, TEST_ONLY_DATASETS,
        generate_dataset_abbreviation, get_dataset_display_name,
        AD_MODE_CONFIG, PREPROCESS_CONFIG, ACTIVE_SPLIT_TYPES,
        DATA_PATH, RESULT_PATH
    )
    print("Successfully imported config settings")
except ImportError as e:
    print(f"Error importing config: {e}")
    # Fallback if config not available
    DATASETS = {}
    DATASET_NAME_TO_KEY = {}
    TEST_ONLY_DATASETS = ['FreeSolv', 'Lipophilicity', 'AqSolDB', 'BigSolDB']
    ACTIVE_SPLIT_TYPES = ['rm']  # Default to random split only
    AD_MODE_CONFIG = {'default_mode': 'flexible'}
    PREPROCESS_CONFIG = {'test_size': 0.2, 'random_state': 42}

def get_dataset_abbreviation(dataset_name: str) -> str:
    """
    Get abbreviation for dataset name.

    Args:
        dataset_name (str): Full dataset name

    Returns:
        str: Dataset abbreviation (e.g., 'ws' for 'WaterSolubility')
    """
    # Use config's function
    return generate_dataset_abbreviation(dataset_name)

def get_descriptor_list(args):
    """
    Get descriptor list based on command-line arguments.

    Args:
        args: Parsed argparse Namespace object with descriptor_mode, custom_descriptors, num_descriptors

    Returns:
        List[str] or None: List of descriptor names based on mode (fast/full/custom)
    """
    from config import CHEMICAL_DESCRIPTORS
    from extra_code.qsar_analysis.config import DESCRIPTOR_NAMES_FAST

    # Handle both dict and list formats for CHEMICAL_DESCRIPTORS
    if isinstance(CHEMICAL_DESCRIPTORS, dict) and 'selection' in CHEMICAL_DESCRIPTORS:
        full_descriptors = CHEMICAL_DESCRIPTORS['selection']
    else:
        full_descriptors = CHEMICAL_DESCRIPTORS
    
    if args.descriptor_mode == 'fast':
        return DESCRIPTOR_NAMES_FAST
    elif args.descriptor_mode == 'full':
        return full_descriptors
    elif args.descriptor_mode == 'custom':
        if args.custom_descriptors:
            descriptors = []
            for item in args.custom_descriptors:
                # Check if it's a number (index)
                if item.isdigit():
                    idx = int(item)
                    if 0 <= idx < len(full_descriptors):
                        descriptors.append(full_descriptors[idx])
                    else:
                        print(f"[WARNING] Index {idx} out of range (0-{len(full_descriptors)-1})")
                # Otherwise treat as descriptor name
                elif item in full_descriptors:
                    descriptors.append(item)
                else:
                    print(f"[WARNING] Unknown descriptor: {item}")
            return descriptors if descriptors else DESCRIPTOR_NAMES_FAST
        elif args.num_descriptors:
            # Use first N descriptors
            n = min(args.num_descriptors, len(full_descriptors))
            return full_descriptors[:n]
        else:
            print("[WARNING] Custom mode but no descriptors specified, using fast mode")
            return DESCRIPTOR_NAMES_FAST
    
    return None

def print_dataset_info(df_dict: Dict) -> None:
    """
    Print dataset information with abbreviations in formatted table.

    Args:
        df_dict (Dict): Dictionary mapping dataset names to DataFrames
    """
    print("\nAvailable Datasets:")
    print("-" * 60)
    print(f"{'Full Name':<30} {'Abbreviation':<15} {'Samples':<10}")
    print("-" * 60)
    
    for name, df in df_dict.items():
        abbrev = get_dataset_abbreviation(name)
        n_samples = len(df) if hasattr(df, '__len__') else df.shape[0]
        print(f"{name:<30} {abbrev:<15} {n_samples:<10}")
    print("-" * 60)

def data_split_only(
    df_dict: Optional[Dict] = None,
    output_dir: str = None,
    test_size: float = 0.2,
    test_only_datasets: Optional[List[str]] = None,
    update_data_folder: bool = True,
    random_seed: int = 42,
    max_samples: int = 30000,
    descriptor_list: Optional[List[str]] = None
) -> Dict:
    """
    Perform rapid data splitting without comprehensive analysis (optimized for speed).

    This function provides fast dataset splitting using various strategies while maintaining
    reproducibility and data integrity. It's designed for quick experimentation and
    initial model development phases where detailed analysis can be deferred.

    Key Features:
    ------------
    - Multiple splitting strategies (random, scaffold, temporal)
    - Automatic dataset loading and validation
    - Stratified sampling for balanced splits
    - Memory-efficient processing for large datasets
    - Reproducible results with seed control

    Parameters:
    -----------
    df_dict : Dict, optional
        Pre-loaded dictionary of datasets {dataset_name: DataFrame}.
        If None, automatically loads from config.DATASET_PATHS.
        Example: {'ws': ws_df, 'de': de_df, 'lo': lo_df}

    output_dir : str, optional
        Directory path for saving split results. If None, uses 'data_splits/'.
        Creates subdirectories: train/, test/, metadata/

    test_size : float, default=0.2
        Fraction of data for test set (0.1 to 0.5).
        Example: 0.2 = 80% train, 20% test

    test_only_datasets : List[str], optional
        Datasets reserved exclusively for testing (no training splits).
        Example: ['external_validation', 'prospective_test']

    update_data_folder : bool, default=True
        Whether to update the standard data/train and data/test directories.
        Set False for custom output locations.

    random_seed : int, default=42
        Seed for reproducible random splits. Ensures consistent results
        across multiple runs for comparison studies.

    max_samples : int, default=30000
        Maximum samples per dataset to prevent memory issues.
        Large datasets are sampled randomly while maintaining distribution.

    descriptor_list : List[str], optional
        Specific molecular descriptors to include in splits.
        If None, includes all available descriptors.

    Returns:
    --------
    Dict : Split information and statistics
        {
            'datasets_processed': List[str],
            'split_method': str,
            'train_sizes': Dict[str, int],
            'test_sizes': Dict[str, int],
            'output_paths': Dict[str, str],
            'execution_time': float
        }

    Examples:
    ---------
    # Basic random split of all datasets
    result = data_split_only()
    print(f"Processed {len(result['datasets_processed'])} datasets")

    # Custom split with specific test size
    result = data_split_only(
        test_size=0.15,
        output_dir='experiments/split_001',
        random_seed=123
    )

    # Split specific datasets only
    ws_data = load_ws_dataset()
    de_data = load_de_dataset()
    result = data_split_only(
        df_dict={'ws': ws_data, 'de': de_data},
        test_size=0.25
    )

    # Large dataset handling
    result = data_split_only(
        max_samples=50000,
        test_size=0.2,
        descriptor_list=['morgan_fp', 'rdkit_desc']
    )

    Performance Notes:
    -----------------
    - Processing time: ~5-30 seconds depending on dataset size
    - Memory usage: Linear with dataset size, optimized for datasets up to 100K compounds
    - Disk space: ~2x original dataset size (train + test copies)
    """
    # Use default output directory if not specified
    if output_dir is None:
        output_dir = 'result/1_preprocess'
    
    print("\n[START] Starting SPLIT-ONLY mode (Fast)")
    print(f"[DATA] Test size: {test_size:.1%}")
    print(f"[DIR] Output directory: {output_dir}")
    
    # Load data if not provided
    if df_dict is None:
        print("\n[FOLDER] Loading datasets...")
        df_dict = load_data()
        print(f"[OK] Loaded {len(df_dict)} datasets")
    
    # Print dataset info
    print_dataset_info(df_dict)
    
    # Default test-only datasets from config
    if test_only_datasets is None:
        test_only_datasets = TEST_ONLY_DATASETS
    
    print(f"\n[TARGET] Test-only datasets: {test_only_datasets}")
    
    # Set descriptor configuration if provided
    if descriptor_list is not None:
        from extra_code.qsar_analysis import config as qsar_config
        qsar_config.ACTIVE_DESCRIPTORS = descriptor_list
        print(f"[CONFIG] Using {len(descriptor_list)} descriptors: {descriptor_list[:5]}..." if len(descriptor_list) > 5 else descriptor_list)
    
    # Run analysis in split-only mode
    # Note: run_enhanced_analysis uses ACTIVE_SPLIT_TYPES from config
    print(f"[CONFIG] Active split types: {ACTIVE_SPLIT_TYPES}")
    analyzer = run_enhanced_analysis(
        df_dict=df_dict,
        test_only_datasets=test_only_datasets,
        output_dir=output_dir,
        performance_mode=False,
        ad_mode='flexible',
        ad_analysis_mode='flexible',  # Minimal AD analysis
        max_samples=max_samples,
        show_recommendations=False,
        enable_reliability_scoring=False,
        test_size=test_size,
        split_only_mode=True,  # KEY: Enable split-only mode
        update_data_folder=update_data_folder
    )
    
    print("\n[OK] Data splitting completed!")
    print(f"[DIR] Splits saved to: {output_dir}")
    
    if update_data_folder:
        # The actual update is handled by analyzer._update_data_folder_with_splits()
        print("[DIR] data/train and data/test folders have been updated by the analyzer")
    
    return analyzer

def analysis_only(
    train_dir: str = "result/1_preprocess/train",
    test_dir: str = "result/1_preprocess/test",
    output_dir: str = "result/1_preprocess",
    ad_analysis_mode: str = 'all',
    use_advanced_ad: bool = False,
    max_samples: int = 30000,
    descriptor_list: Optional[List[str]] = None
) -> Dict:
    """
    Perform analysis only on existing splits
    
    Parameters:
    -----------
    train_dir : str
        Directory containing training splits
    test_dir : str
        Directory containing test splits
    output_dir : str
        Output directory for analysis results
    ad_analysis_mode : str
        AD analysis mode: 'all', 'strict', 'flexible', or 'adaptive'
    
    Returns:
    --------
    Dict : Analysis results
    """
    print("\n[ANALYSIS] Starting ANALYSIS-ONLY mode")
    print(f"[DIR] Train directory: {train_dir}")
    print(f"[DIR] Test directory: {test_dir}")
    print(f"[DIR] Output directory: {output_dir}")
    print(f"[TARGET] AD analysis mode: {ad_analysis_mode}")
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("Train/test directories not found. Run data_split_only() first.")
    
    # Load existing splits into df_dict format with splits preserved
    print("\n[FOLDER] Loading existing splits...")
    df_dict = {}
    splits_dict = {}  # Store split information separately
    
    # Get all split types (e.g., rm, sc, cs, etc.)
    split_types = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    # First pass: collect all unique dataset names and combine all data
    all_datasets = set()
    for split_type in split_types:
        split_train_dir = os.path.join(train_dir, split_type)
        for train_file in os.listdir(split_train_dir):
            if train_file.endswith('_train.csv'):
                dataset_name = train_file.replace(f'{split_type}_', '').replace('_train.csv', '')
                all_datasets.add(dataset_name)
    
    # Second pass: load data for each dataset
    import polars as pl
    for dataset_name in all_datasets:
        # Load the first split to get the full dataset
        first_split = split_types[0] if split_types else None
        if not first_split:
            continue
            
        split_train_dir = os.path.join(train_dir, first_split)
        split_test_dir = os.path.join(test_dir, first_split)
        
        train_file = f"{first_split}_{dataset_name}_train.csv"
        test_file = f"{first_split}_{dataset_name}_test.csv"
        
        train_path = os.path.join(split_train_dir, train_file)
        test_path = os.path.join(split_test_dir, test_file)
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            # Load and combine to get full dataset (NO DEDUPLICATION!)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Convert to polars
            df_dict[dataset_name] = pl.from_pandas(combined_df)
            
            # Now create splits with correct indices for each split type
            splits_dict[dataset_name] = {}
            for split_type in split_types:
                split_train_dir = os.path.join(train_dir, split_type)
                split_test_dir = os.path.join(test_dir, split_type)
                
                train_file = f"{split_type}_{dataset_name}_train.csv"
                test_file = f"{split_type}_{dataset_name}_test.csv"
                
                train_path = os.path.join(split_train_dir, train_file)
                test_path = os.path.join(split_test_dir, test_file)
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    train_df = pd.read_csv(train_path)
                    test_df = pd.read_csv(test_path)
                    
                    # Get indices based on original ordering
                    train_indices = list(range(len(train_df)))
                    test_indices = list(range(len(train_df), len(train_df) + len(test_df)))
                    
                    splits_dict[dataset_name][split_type] = {
                        'train_idx': train_indices,
                        'test_idx': test_indices
                    }
    
    print(f"[OK] Loaded {len(df_dict)} datasets from existing splits")
    print_dataset_info(df_dict)
    
    # Run enhanced analysis on loaded data
    print("\n[ANALYSIS] Starting enhanced analysis on existing splits...")
    print(f"[INFO] Loaded splits for: {list(splits_dict.keys())}")
    for dataset_name, dataset_splits in splits_dict.items():
        print(f"  {dataset_name}: {list(dataset_splits.keys())}")
    
    # Set descriptor configuration if provided
    if descriptor_list is not None:
        from extra_code.qsar_analysis import config as qsar_config
        qsar_config.ACTIVE_DESCRIPTORS = descriptor_list
        print(f"[CONFIG] Using {len(descriptor_list)} descriptors")
    
    # Create analyzer with existing splits - SKIP SPLIT CREATION
    analyzer = run_enhanced_analysis(
        df_dict=df_dict,
        output_dir=output_dir,
        test_only_datasets=[],  # Already handled in splits
        ad_analysis_mode=ad_analysis_mode,
        performance_mode=False,  # Full analysis
        enable_reliability_scoring=True,
        show_recommendations=True,
        skip_split_creation=True,  # Important: Don't create new splits!
        existing_splits=splits_dict,  # Pass the splits information
        max_samples=max_samples
    )
    
    print("\n[OK] Analysis completed successfully!")
    return analyzer

def full_preprocessing(
    df_dict: Optional[Dict] = None,
    output_dir: str = "result/1_preprocess",
    test_size: float = 0.2,
    test_only_datasets: Optional[List[str]] = None,
    ad_analysis_mode: str = 'all',
    update_data_folder: bool = True,
    max_samples: int = 30000,
    descriptor_list: Optional[List[str]] = None,
    use_advanced_ad: bool = False
) -> Dict:
    """
    Perform full preprocessing (split + analysis)
    
    Parameters:
    -----------
    df_dict : Dict, optional
        Dictionary of datasets. If None, will load automatically
    output_dir : str
        Output directory for results
    test_size : float
        Test set size (0.1 to 0.5)
    test_only_datasets : List[str], optional
        Datasets to use only for testing
    ad_analysis_mode : str
        AD analysis mode: 'all', 'strict', 'flexible', or 'adaptive'
    update_data_folder : bool
        Whether to update data/train and data/test folders
    
    Returns:
    --------
    Dict : Analysis results
    """
    print("\n[TOOL] Starting FULL PREPROCESSING (Split + Analysis)")
    print(f"[DATA] Test size: {test_size:.1%}")
    print(f"[DIR] Output directory: {output_dir}")
    print(f"[TARGET] AD analysis mode: {ad_analysis_mode}")

    # Create output directory if it doesn't exist, or use existing one for overwriting
    if os.path.exists(output_dir):
        print(f"[OVERWRITE] Using existing output directory: {output_dir}")
    else:
        print(f"[CREATE] Creating new output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data if not provided
    if df_dict is None:
        print("\n[FOLDER] Loading datasets...")
        df_dict = load_data()
        print(f"[OK] Loaded {len(df_dict)} datasets")
    
    # Print dataset info
    print_dataset_info(df_dict)
    
    # Default test-only datasets from config
    if test_only_datasets is None:
        test_only_datasets = TEST_ONLY_DATASETS
    
    print(f"\n[TARGET] Test-only datasets: {test_only_datasets}")
    
    # Set descriptor configuration if provided
    if descriptor_list is not None:
        from extra_code.qsar_analysis import config as qsar_config
        qsar_config.ACTIVE_DESCRIPTORS = descriptor_list
        print(f"[CONFIG] Using {len(descriptor_list)} descriptors")
    
    # Run full analysis
    analyzer = run_enhanced_analysis(
        df_dict=df_dict,
        test_only_datasets=test_only_datasets,
        output_dir=output_dir,
        performance_mode=False,
        ad_mode='flexible',  # Fallback mode
        ad_analysis_mode=ad_analysis_mode,  # Primary AD mode
        max_samples=max_samples,
        show_recommendations=False,
        enable_reliability_scoring=False,
        test_size=test_size,
        split_only_mode=False,  # Full analysis
        update_data_folder=update_data_folder
    )
    
    print("\n[OK] Full preprocessing completed!")
    print(f"[DIR] All results saved to: {output_dir}")
    
    if update_data_folder:
        # The actual update is handled by analyzer._update_data_folder_with_splits()
        print("[DIR] data/train and data/test folders have been updated by the analyzer")
    
    return analyzer

# Main execution
if __name__ == "__main__":
    # Set up logging to file
    from pathlib import Path
    from config import MODULE_NAMES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get module name from config
    module_name = MODULE_NAMES.get('1', '1_preprocess')

    # Create logs directory
    log_dir = Path(f"logs/{module_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{module_name}_{timestamp}.log"

    # Create a custom logger class
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
            self.log.write(f"{'='*60}\n")
            self.log.write(f"Module 1 (Preprocess) Execution Started: {datetime.now()}\n")
            self.log.write(f"{'='*60}\n\n")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    # Redirect stdout to both terminal and file
    logger = Logger(log_file)
    sys.stdout = logger

    # Initialize performance monitor
    monitor = PerformanceMonitor(code_num=1, code_name="preprocess", result_dir="result/1_preprocess")
    monitor.start()

    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="ANO Data Preprocessing - Flexible data preparation tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Default: Split-only mode
  python 1_preprocess_functions.py
  
  # Split with custom test size
  python 1_preprocess_functions.py --test-size 0.2
  
  # Analysis only with AD mode
  python 1_preprocess_functions.py --analyze-only --ad-mode strict
  
  # Full preprocessing
  python 1_preprocess_functions.py --full --ad-mode flexible

  # 2025 enhanced standards for industrial/military chemicals
  python 1_preprocess_functions.py --analyze-only --ad-mode all

  # Custom output directory
  python 1_preprocess_functions.py --output-dir my_results
        """
        )
        
        # Mode selection (mutually exclusive)
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument('--split-only', action='store_true', default=True,
                               help='Only perform data splitting (default)')
        mode_group.add_argument('--analyze-only', action='store_true',
                               help='Only analyze existing data')
        mode_group.add_argument('--full', action='store_true',
                               help='Full preprocessing (split + analysis)')
        
        # Parameters
        parser.add_argument('--test-size', type=float, default=PREPROCESS_CONFIG.get('test_size', 0.2),
                           help='Test set size (0.1-0.5, default: 0.2)')
        parser.add_argument('--ad-mode', choices=['strict', 'flexible', 'adaptive', 'all'],
                           default=None,  # Will be set based on mode
                           help='Applicability domain analysis mode: strict (ICH M7), flexible (consensus), adaptive (context-dependent), all (run all modes)')
        parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory (default: auto-generated based on mode)')
        parser.add_argument('--no-update-data', action='store_true',
                           help='Do NOT update data/train and data/test folders')
        parser.add_argument('--random-seed', type=int, default=PREPROCESS_CONFIG.get('random_state', 42),
                           help='Random seed for reproducibility')
        parser.add_argument('--test-only-datasets', nargs='+', default=TEST_ONLY_DATASETS,
                           help='Datasets to use only for testing')
        parser.add_argument('--advanced-ad', action='store_true', default=False,
                           help='Use advanced AD methods (Mahalanobis, Hotelling T², etc.)')
        
        # New parameters for samples and descriptors
        parser.add_argument('--max-samples', type=int, default=30000,
                           help='Maximum samples for analysis (default: 30000)')
        parser.add_argument('--descriptor-mode', choices=['fast', 'full', 'custom'], default='fast',
                           help='Descriptor mode: fast (7), full (49), or custom')
        parser.add_argument('--custom-descriptors', nargs='+', type=str,
                           help='Custom descriptor names or indices (e.g., 0 1 2 or MolWeight MolLogP)')
        parser.add_argument('--num-descriptors', type=int,
                           help='Use first N descriptors from full list (e.g., --num-descriptors 15)')
        
        args = parser.parse_args()
        
        monitor.log_progress("ANO Data Preprocessing Functions")
        monitor.log_progress("=" * 60)
        
        # Determine mode
        if args.analyze_only:
            mode = 'analyze'
            default_output = 'result/1_preprocess'
        elif args.full:
            mode = 'full'
            default_output = 'result/1_preprocess'
        else:
            mode = 'split'
            default_output = 'result/1_preprocess'

        output_dir = args.output_dir or default_output

        # Set AD mode based on execution mode and user preference
        if args.ad_mode is None:
            if mode == 'full':
                ad_mode = 'all'  # Full mode runs all AD methods by default
            else:
                ad_mode = AD_MODE_CONFIG.get('default_mode', 'flexible')  # Other modes use default
        else:
            ad_mode = args.ad_mode  # User explicitly specified
        
        monitor.log_progress(f"[START] Running in {mode.upper()} mode")
        monitor.log_progress(f"[DATA] Test size: {args.test_size:.1%}")
        monitor.log_progress(f"[SEARCH] AD mode: {ad_mode}")
        monitor.log_progress(f"[DIR] Output directory: {output_dir}")
        monitor.log_progress(f"[RANDOM] Random seed: {args.random_seed}")
        monitor.log_progress(f"[SAMPLES] Max samples: {args.max_samples}")
        monitor.log_progress(f"[DESCRIPTORS] Mode: {args.descriptor_mode}")
        
        # Get descriptor list based on arguments
        descriptor_list = get_descriptor_list(args)
        if descriptor_list:
            monitor.log_progress(f"[DESCRIPTORS] Using {len(descriptor_list)} descriptors")
        
        # Execute based on mode
        if mode == 'split':
            analyzer = data_split_only(
                test_size=args.test_size,
                output_dir=output_dir,
                test_only_datasets=args.test_only_datasets,
                update_data_folder=not args.no_update_data,  # Update by default
                random_seed=args.random_seed,
                max_samples=args.max_samples,
                descriptor_list=descriptor_list
            )
        elif mode == 'analyze':
            analyzer = analysis_only(
                output_dir=output_dir,
                ad_analysis_mode=ad_mode,
                use_advanced_ad=args.advanced_ad,
                max_samples=args.max_samples,
                descriptor_list=descriptor_list
            )
        elif mode == 'full':
            analyzer = full_preprocessing(
                test_size=args.test_size,
                output_dir=output_dir,
                ad_analysis_mode=ad_mode,
                test_only_datasets=args.test_only_datasets,
                update_data_folder=not args.no_update_data,  # Update by default
                max_samples=args.max_samples,
                descriptor_list=descriptor_list,
                use_advanced_ad=args.advanced_ad  # Enable advanced AD methods
            )
        
        monitor.log_progress("[OK] Processing completed successfully!")
    
    except Exception as e:
        monitor.log_progress(f"Error occurred: {e}", level="error")
        raise
    
    finally:
        # Stop monitoring and save results
        stats = monitor.stop()
        print(f"\n[PERFORMANCE] Execution time: {stats.get('duration_formatted', 'N/A')}")
        print(f"[PERFORMANCE] Memory change: {stats.get('memory_delta_mb', 0):+.2f} MB")

        # Close log file
        logger.log.write(f"\n{'='*60}\n")
        logger.log.write(f"Module 1 (Preprocess) Execution Finished: {datetime.now()}\n")
        logger.log.write(f"{'='*60}\n")
        sys.stdout = logger.terminal
        logger.close()