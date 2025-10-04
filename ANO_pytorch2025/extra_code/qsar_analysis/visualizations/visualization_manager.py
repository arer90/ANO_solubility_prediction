"""
QSAR Visualization Manager

This module contains the main visualization manager that coordinates all visualizers.
"""

from pathlib import Path
from typing import Dict, Optional

# Import visualizers
from .ad_plots import ADVisualizer
from .meta_plots import MetaVisualizer
from .stat_plots import StatisticalVisualizer
from .summary_plots import SummaryVisualizer

# Import utilities
from .visualization_utils import (
    check_and_create_paths,
    list_output_files,
    verify_output_structure
)

class VisualizationManager:
    """Main visualization manager that coordinates all visualizers"""
    
    def __init__(self, output_dir: Path, ad_mode: str = 'flexible', 
                 exclude_test_only: bool = True, n_jobs: int = 1):
        """
        Initialize visualization manager
        
        Parameters:
        -----------
        output_dir : Path
            Output directory for all visualizations
        ad_mode : str
            AD analysis mode ('strict', 'flexible', 'adaptive')
        exclude_test_only : bool
            Whether to exclude test-only datasets from certain analyses
        n_jobs : int
            Number of parallel jobs (currently sequential only)
        """
        self.output_dir = Path(output_dir)
        self.ad_mode = ad_mode
        self.exclude_test_only = exclude_test_only
        self.n_jobs = n_jobs
        
        print(f"\nInitializing Visualization Manager")
        print(f"  Output directory: {self.output_dir}")
        print(f"  AD mode: {self.ad_mode}")
        print(f"  Exclude test-only: {self.exclude_test_only}")
        
        # Verify all paths are created
        check_and_create_paths(self.output_dir)
        
        # Initialize all visualizers
        print("\nInitializing visualizers...")
        
        self.ad_visualizer = ADVisualizer(
            output_dir=self.output_dir,
            ad_mode=self.ad_mode,
            exclude_test_only=self.exclude_test_only,
            n_jobs=self.n_jobs
        )
        print("  [OK] AD Visualizer initialized")
        
        self.meta_visualizer = MetaVisualizer(output_dir=self.output_dir)
        print("  [OK] Meta Visualizer initialized")
        
        self.stat_visualizer = StatisticalVisualizer(output_dir=self.output_dir)
        print("  [OK] Statistical Visualizer initialized")
        
        self.summary_visualizer = SummaryVisualizer(output_dir=self.output_dir)
        print("  [OK] Summary Visualizer initialized")
        
        print("\nAll visualizers initialized successfully!")
    
    def create_all_visualizations(self, datasets: Dict, splits: Dict, features: Dict, 
                                ad_analysis: Dict, statistical_results: Dict, 
                                performance_results: Optional[Dict] = None):
        """
        Create all visualizations
        
        Parameters:
        -----------
        datasets : Dict
            Dataset information
        splits : Dict
            Train/test split information
        features : Dict
            Feature data
        ad_analysis : Dict
            AD analysis results
        statistical_results : Dict
            Statistical analysis results
        performance_results : Optional[Dict]
            AD performance analysis results (if available)
        """
        print("\n" + "="*60)
        print("Creating all visualizations...")
        print("="*60)
        
        # Validate inputs
        if not self._validate_inputs(datasets, splits, features, ad_analysis, statistical_results):
            print("\n[WARNING] Input validation failed. Some visualizations may be skipped.")
        
        # 1. AD visualizations
        print("\n1. Creating AD Visualizations:")
        print("-" * 40)
        try:
            self.ad_visualizer.create_all_ad_visualizations(
                ad_analysis, features, performance_results
            )
            print("  [OK] AD visualizations completed")
        except Exception as e:
            print(f"  [ERROR] AD visualizations failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 2. Meta visualizations
        print("\n2. Creating Meta Visualizations:")
        print("-" * 40)
        try:
            self.meta_visualizer.create_all_meta_visualizations(
                datasets, splits, features, ad_analysis
            )
            print("  [OK] Meta visualizations completed")
        except Exception as e:
            print(f"  [ERROR] Meta visualizations failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 3. Statistical visualizations
        print("\n3. Creating Statistical Visualizations:")
        print("-" * 40)
        try:
            self.stat_visualizer.create_all_statistical_visualizations(
                splits, features, statistical_results
            )
            print("  [OK] Statistical visualizations completed")
        except Exception as e:
            print(f"  [ERROR] Statistical visualizations failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 4. Summary visualizations
        print("\n4. Creating Summary Visualizations:")
        print("-" * 40)
        try:
            self.summary_visualizer.create_all_summary_visualizations(
                datasets, splits, ad_analysis, statistical_results
            )
            print("  [OK] Summary visualizations completed")
        except Exception as e:
            print(f"  [ERROR] Summary visualizations failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("[OK] All visualization tasks completed")
        print("="*60)
        
        # Verify path contents
        self._verify_output_contents()
    
    def analyze_ad_performance(self, features_dict: Dict, targets_dict: Dict, 
                             ad_analysis_dict: Dict, splits_dict: Dict) -> Optional[Dict]:
        """
        Analyze AD performance for all datasets
        
        This delegates to the AD visualizer's performance analyzer
        """
        print("\nAnalyzing AD Performance...")
        print("-" * 40)
        
        if hasattr(self.ad_visualizer, 'analyze_ad_performance_for_all_datasets'):
            return self.ad_visualizer.analyze_ad_performance_for_all_datasets(
                features_dict, targets_dict, ad_analysis_dict, splits_dict
            )
        else:
            print("  [ERROR] AD performance analyzer not available")
            return None
    
    def _validate_inputs(self, datasets: Dict, splits: Dict, features: Dict,
                        ad_analysis: Dict, statistical_results: Dict) -> bool:
        """Validate input data"""
        print("\nValidating input data...")
        
        valid = True
        
        # Check datasets
        if not datasets:
            print("  [WARNING] No datasets provided")
            valid = False
        else:
            print(f"  [OK] Datasets: {len(datasets)}")
            
        # Check splits
        if not splits:
            print("  [WARNING] No splits provided")
            valid = False
        else:
            print(f"  [OK] Splits: {len(splits)}")
            
        # Check features
        if not features:
            print("  [WARNING] No features provided")
            valid = False
        else:
            print(f"  [OK] Features: {len(features)}")
            
        # Check AD analysis
        if not ad_analysis:
            print("  [WARNING] No AD analysis results provided")
            valid = False
        else:
            print(f"  [OK] AD Analysis: {len(ad_analysis)}")
            
        # Check statistical results
        if not statistical_results:
            print("  [WARNING] No statistical results provided")
            valid = False
        else:
            print(f"  [OK] Statistical Results: {len(statistical_results)}")
        
        # Check consistency
        if datasets and splits:
            dataset_names = set(datasets.keys())
            split_names = set(splits.keys())
            if dataset_names != split_names:
                print(f"  [WARNING] Dataset names mismatch: {dataset_names - split_names}")
                valid = False
        
        return valid
    
    def _verify_output_contents(self):
        """Verify output directory contents"""
        print("\nVerifying output directory contents:")
        print("-" * 40)
        
        dirs_to_check = [
            (self.output_dir / 'meta', 'Meta visualizations'),
            (self.output_dir / 'statistics', 'Statistical visualizations'),
            (self.output_dir / 'summary', 'Summary visualizations'),
            (self.output_dir / 'ad_analysis', 'AD analysis')
        ]
        
        total_files = 0
        
        for dir_path, description in dirs_to_check:
            if dir_path.exists():
                # Count files by type
                png_files = list(dir_path.rglob('*.png'))
                xlsx_files = list(dir_path.rglob('*.xlsx'))
                txt_files = list(dir_path.rglob('*.txt'))
                
                dir_total = len(png_files) + len(xlsx_files) + len(txt_files)
                total_files += dir_total
                
                print(f"\n  {description} ({dir_path.name}):")
                print(f"    Total files: {dir_total}")
                
                if png_files:
                    print(f"    - PNG files: {len(png_files)}")
                if xlsx_files:
                    print(f"    - Excel files: {len(xlsx_files)}")
                if txt_files:
                    print(f"    - Text files: {len(txt_files)}")
                
                # List subdirectories
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"    - Subdirectories: {', '.join(d.name for d in subdirs)}")
            else:
                print(f"\n  {description} ({dir_path.name}): NOT FOUND")
        
        print(f"\nTotal files created: {total_files}")
        
        # Verify structure
        if verify_output_structure(self.output_dir):
            print("\n[CHECK] Output directory structure is complete")
        else:
            print("\n[WARNING] Some directories are missing")


# Convenience function for quick initialization
def create_visualization_manager(output_dir: Path, **kwargs) -> VisualizationManager:
    """
    Create and return a VisualizationManager instance
    
    Parameters:
    -----------
    output_dir : Path
        Output directory for visualizations
    **kwargs : dict
        Additional arguments for VisualizationManager
        
    Returns:
    --------
    VisualizationManager instance
    """
    return VisualizationManager(output_dir, **kwargs)