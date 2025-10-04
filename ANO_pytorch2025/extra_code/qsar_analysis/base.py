"""
QSAR Analyzer Base Module

This module contains the base class and common functionality for the QSAR analyzer.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use(\'Agg\')  # Non-interactive backend
import matplotlib.pyplot as plt

# Seaborn import with fallback
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None
    print("[WARNING] Seaborn not available, some visualizations may be limited")

from .config import PLOT_SETTINGS, DEFAULT_PARAMS, LOGGING_CONFIG

warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use(PLOT_SETTINGS['style'])
if SEABORN_AVAILABLE:
    sns.set_palette(PLOT_SETTINGS['palette'])

# Logging configuration
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    datefmt=LOGGING_CONFIG['date_format']
)

# RDKit import with error handling
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not available. Using simulated features.")


class BaseAnalyzer:
    """Base class for QSAR analysis with common functionality"""
    
    def __init__(self, output_dir: str = "result/1_preprocess", 
                 random_state: int = DEFAULT_PARAMS['random_state']):
        """Initialize base analyzer"""
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.performance_stats = {
            'start_time': time.time(),
            'dataset_times': {},
            'memory_usage': []
        }
        
        # Create output structure
        self.create_output_structure()
        
    def create_output_structure(self):
        """Create enhanced output directory structure"""
        # Since train/test folders are already created in splitters.py,
        # only create additional directories here
        subdirs = [
            "meta", 
            "summary", 
            "statistics", 
            "ad_analysis",
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    def track_memory_usage(self):
        """Track memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.performance_stats['memory_usage'].append(memory_mb)
        except:
            pass
    
    def cleanup_memory(self):
        """Clean up memory after analysis"""
        import gc
        gc.collect()
        print("\n[CLEAN] Memory cleanup completed")
    
    def validate_data(self, df: pl.DataFrame, name: str) -> bool:
        """Validate dataset structure"""
        # Check for either naming convention
        has_smiles_target = 'smiles' in df.columns and 'target' in df.columns
        has_target_xy = 'target_x' in df.columns and 'target_y' in df.columns
        
        if not (has_smiles_target or has_target_xy):
            print(f"[ERROR] Dataset '{name}' missing required columns. Found: {df.columns}")
            return False
        
        # Check for null values based on column names
        null_counts = df.null_count()
        if has_smiles_target:
            if null_counts['smiles'][0] > 0 or null_counts['target'][0] > 0:
                print(f"[WARNING] Dataset '{name}' contains null values")
        else:
            if null_counts['target_x'][0] > 0 or null_counts['target_y'][0] > 0:
                print(f"[WARNING] Dataset '{name}' contains null values")
        
        # Check minimum size
        if len(df) < 10:
            print(f"[ERROR] Dataset '{name}' has insufficient samples ({len(df)})")
            return False
        
        return True