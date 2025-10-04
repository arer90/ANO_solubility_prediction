"""
QSAR Visualization Utilities

This module contains utility functions for visualization management and analysis.
"""

import os
import gc
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use(\'Agg\')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import config
try:
    from ..config import PLOT_SETTINGS
except ImportError:
    from config import PLOT_SETTINGS


def safe_savefig(path: Path, dpi: int = 300, **kwargs):
    """Safely save figure"""
    try:
        # Create path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Debugging: print full path
        print(f"      Attempting to save to: {path}")

        # Default settings
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }

        # Override with kwargs (prevent duplicates)
        save_kwargs.update(kwargs)

        # Save file
        plt.savefig(path, **save_kwargs)
        print(f"      [OK] Saved: {path.name} at {path.parent}")
        return True
    except Exception as e:
        print(f"      [ERROR] Failed to save {path.name}: {str(e)}")
        print(f"      Full path was: {path}")
        import traceback
        traceback.print_exc()  # More detailed error information
        return False
    finally:
        plt.close()
        gc.collect()


def safe_to_excel(df: pd.DataFrame, path: Path, **kwargs):
    """Safely save Excel file"""
    try:
        # Create path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save file
        df.to_excel(path, **kwargs)
        print(f"      [OK] Saved: {path.name}")
        return True
    except Exception as e:
        print(f"      [ERROR] Failed to save {path.name}: {str(e)}")
        return False


def safe_excel_writer(path: Path, engine='openpyxl'):
    """Safe ExcelWriter context manager"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return pd.ExcelWriter(path, engine=engine)
    except Exception as e:
        print(f"      [ERROR] Failed to create ExcelWriter for {path.name}: {str(e)}")
        return None


def check_and_create_paths(output_dir: Path):
    """Check and create all required paths"""
    paths_to_create = [
        output_dir / 'meta',
        output_dir / 'statistics',
        output_dir / 'statistics' / 'individual_plots',
        output_dir / 'summary',
        output_dir / 'summary' / 'individual_plots',
        output_dir / 'ad_analysis',
        output_dir / 'ad_analysis' / 'ad_performance_analysis',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'random_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'scaffold_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'cluster_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'time_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'test_only'
    ]
    
    print("\nCreating/verifying output directories:")
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {path.relative_to(output_dir)}")
    
    return True


def list_output_files(output_dir: Path, extensions: List[str] = ['.png', '.xlsx', '.txt']):
    """List all files in output directory"""
    print(f"\nListing all files in {output_dir}:")

    all_files = []
    for ext in extensions:
        files = list(output_dir.rglob(f'*{ext}'))
        all_files.extend(files)

    # Group by directory
    files_by_dir = {}
    for file in all_files:
        try:
            rel_dir = file.parent.relative_to(output_dir)
        except ValueError:
            # If file is not relative to output_dir, use absolute path
            rel_dir = file.parent
        
        if rel_dir not in files_by_dir:
            files_by_dir[rel_dir] = []
        files_by_dir[rel_dir].append(file.name)
    
    # Output
    for dir_path, files in sorted(files_by_dir.items()):
        print(f"\n  {dir_path}/")
        for file in sorted(files):
            print(f"    - {file}")
    
    print(f"\nTotal files found: {len(all_files)}")
    return all_files


def verify_output_structure(output_dir: Path):
    """Verify output directory structure"""
    expected_dirs = {
        'meta': ['individual_plots'],
        'statistics': ['individual_plots'],
        'summary': ['individual_plots'],
        'ad_analysis': ['ad_performance_analysis'],
        'ad_analysis/ad_performance_analysis': [
            'random_split', 'scaffold_split', 'cluster_split', 
            'time_split', 'test_only'
        ]
    }
    
    print("\nVerifying output directory structure:")
    all_good = True
    
    for parent, subdirs in expected_dirs.items():
        parent_path = output_dir / parent
        if parent_path.exists():
            print(f"  [OK] {parent}/")
            for subdir in subdirs:
                subdir_path = parent_path / subdir
                if subdir_path.exists():
                    print(f"    [OK] {subdir}/")
                else:
                    print(f"    [ERROR] {subdir}/ (missing)")
                    all_good = False
        else:
            print(f"  [ERROR] {parent}/ (missing)")
            all_good = False
    
    return all_good


def create_visualizations_with_debugging(visualization_manager, datasets, splits,
                                       features, ad_analysis, statistical_results,
                                       performance_results=None):
    """Create visualizations with detailed progress information"""
    print("\n" + "="*60)
    print("STARTING VISUALIZATION CREATION")
    print("="*60)
    
    # Check paths before starting
    output_dir = visualization_manager.output_dir
    print(f"\nOutput directory: {output_dir}")
    print(f"Exists: {output_dir.exists()}")
    
    # Check input data
    print("\nInput data check:")
    print(f"  Datasets: {len(datasets) if datasets else 0}")
    print(f"  Splits: {len(splits) if splits else 0}")
    print(f"  Features: {len(features) if features else 0}")
    print(f"  AD Analysis: {len(ad_analysis) if ad_analysis else 0}")
    print(f"  Statistical Results: {len(statistical_results) if statistical_results else 0}")
    print(f"  Performance Results: {len(performance_results) if performance_results else 0}")
    
    # Create visualizations
    try:
        visualization_manager.create_all_visualizations(
            datasets, splits, features, ad_analysis, 
            statistical_results, performance_results
        )
        success = True
    except Exception as e:
        print(f"\n[ERROR] ERROR during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Check results
    print("\n" + "="*60)
    print("VISUALIZATION RESULTS")
    print("="*60)

    # List of generated files
    files = list_output_files(output_dir)
    
    # Check each main directory
    for subdir in ['meta', 'statistics', 'summary', 'ad_analysis']:
        subpath = output_dir / subdir
        if subpath.exists():
            file_count = len(list(subpath.rglob('*.*')))
            print(f"\n{subdir}: {file_count} files")

            # Count by file type
            png_count = len(list(subpath.rglob('*.png')))
            xlsx_count = len(list(subpath.rglob('*.xlsx')))
            txt_count = len(list(subpath.rglob('*.txt')))
            
            if png_count > 0:
                print(f"  - PNG files: {png_count}")
            if xlsx_count > 0:
                print(f"  - Excel files: {xlsx_count}")
            if txt_count > 0:
                print(f"  - Text files: {txt_count}")
        else:
            print(f"\n{subdir}: DIRECTORY NOT FOUND")
    
    # Verify structure
    print("\n" + "-"*60)
    structure_ok = verify_output_structure(output_dir)
    
    if success and structure_ok:
        print("\n[CHECK] Visualization completed successfully!")
    else:
        print("\n[WARNING] Visualization completed with issues.")
    
    return files, success


# Export functions
__all__ = [
    'safe_savefig',
    'safe_to_excel',
    'safe_excel_writer',
    'check_and_create_paths',
    'list_output_files',
    'verify_output_structure',
    'create_visualizations_with_debugging'
]