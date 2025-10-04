import time
from typing import Dict, List, Tuple
import gc
import numpy as np
import polars as pl

from .base import BaseAnalyzer
from .config import DEFAULT_PARAMS

class DataLoader(BaseAnalyzer):
    """Handles data loading and preprocessing for QSAR analysis"""
    
    def __init__(self, output_dir: str = "result/1_preprocess",
                 random_state: int = DEFAULT_PARAMS['random_state'],
                 performance_mode: bool = DEFAULT_PARAMS['performance_mode'],
                 max_samples_analysis: int = DEFAULT_PARAMS['max_samples_analysis']):
        """Initialize data loader"""
        super().__init__(output_dir, random_state)
        self.performance_mode = performance_mode
        self.max_samples_analysis = max_samples_analysis
        
        # Storage
        self.datasets = {}
        self.splits = {}
        
    def load_and_preprocess_data(self, df_dict: Dict[str, pl.DataFrame], 
                            test_only_datasets: List[str] = None) -> Dict:
        """Load and preprocess datasets with memory optimization"""
        if test_only_datasets is None:
            test_only_datasets = []
            
        print("\n[CHART] Loading and preprocessing data")
        
        for name, df in df_dict.items():
            try:
                print(f"  Processing {name}...")
                start_time = time.time()
                
                # Validate data
                if not self.validate_data(df, name):
                    continue
                
                # Clean and filter data
                clean_df = self._clean_dataset(df)
                
                if len(clean_df) < 10:
                    print(f"    ⚠ Skipping {name}: insufficient data after cleaning ({len(clean_df)} samples)")
                    continue
                
                # AGGRESSIVE SAMPLING
                original_size = len(clean_df)
                
                # Sampling strategy
                if self.performance_mode and DEFAULT_PARAMS.get('enable_sampling', True):
                    sampling_ratio = DEFAULT_PARAMS.get('sampling_ratio', 0.1)
                    max_samples = self.max_samples_analysis
                    
                    if original_size > max_samples:
                        # Ratio-based sampling for large datasets
                        if original_size > 10000:
                            sample_size = min(max_samples, int(original_size * sampling_ratio))
                        else:
                            sample_size = max_samples
                        
                        print(f"    [FAST] Sampling {sample_size}/{original_size} ({sample_size/original_size*100:.1f}%)")
                        clean_df = clean_df.sample(n=sample_size, seed=self.random_state)
                
                # Store the size before deleting
                analysis_size = len(clean_df)
                
                self.datasets[name] = {
                    'data': clean_df,
                    'is_test_only': name in test_only_datasets,
                    'original_size': original_size,
                    'analysis_size': analysis_size
                }
                
                # Extract targets and SMILES with memory optimization
                # Check column names and use appropriate ones
                if "smiles" in clean_df.columns and "target" in clean_df.columns:
                    smiles_col = "smiles"
                    target_col = "target"
                elif "target_x" in clean_df.columns and "target_y" in clean_df.columns:
                    smiles_col = "target_x"
                    target_col = "target_y"
                else:
                    print(f"[ERROR] Dataset '{name}' missing required columns. Found: {clean_df.columns}")
                    continue
                    
                self.splits[name] = {
                    'smiles': clean_df.select(smiles_col).to_series().to_list(),
                    'targets': clean_df.select(target_col).to_numpy().flatten().astype(np.float32),
                    'splits': {}
                }
                
                elapsed = time.time() - start_time
                self.performance_stats['dataset_times'][name] = elapsed
                
                print(f"    [OK] Loaded: {analysis_size:,} samples ({'test-only' if name in test_only_datasets else 'train/test'}) in {elapsed:.2f}s")
                
                # Immediate memory cleanup
                del clean_df
                gc.collect()
                
                # Memory tracking
                self.track_memory_usage()
                
            except Exception as e:
                print(f"    [ERROR] Failed to process {name}: {str(e)}")
                continue
        
        if not self.datasets:
            raise ValueError("No datasets loaded successfully")
        
        print(f"[OK] Successfully loaded {len(self.datasets)} datasets")
        return self.datasets
    
    def _clean_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and prepare dataset with enhanced error handling"""
        # Check column names and rename for consistency
        if "smiles" in df.columns and "target" in df.columns:
            smiles_col = "smiles"
            target_col = "target"
        elif "target_x" in df.columns and "target_y" in df.columns:
            # Rename for consistency: target_x -> smiles, target_y -> target
            df = df.rename({"target_x": "smiles", "target_y": "target"})
            smiles_col = "smiles"
            target_col = "target"
        else:
            # Try to find suitable columns
            print(f"[WARNING] Columns not standard. Found: {df.columns}")
            return df
            
        # Filter out null values
        clean_df = df.filter(
            (pl.col(smiles_col).is_not_null()) & 
            (pl.col(target_col).is_not_null())
        )
        
        # Ensure correct data types with better error handling
        try:
            from .config import POLARS_TYPE_MAPPING, DEFAULT_PARAMS
            
            float_type = POLARS_TYPE_MAPPING.get(
                DEFAULT_PARAMS.get('float_precision', 'float32'), 
                pl.Float32
            )
            
            clean_df = clean_df.with_columns([
                pl.col(smiles_col).cast(pl.Utf8).alias(smiles_col),
                pl.col(target_col).cast(float_type).alias(target_col)
            ])
        except ImportError:
            # Fallback to basic types
            clean_df = clean_df.with_columns([
                pl.col(smiles_col).cast(pl.Utf8).alias(smiles_col),
                pl.col(target_col).cast(pl.Float32).alias(target_col)
            ])
        except Exception as e:
            print(f"Warning: Type casting failed: {e}, using defaults")
            clean_df = clean_df.with_columns([
                pl.col(smiles_col).cast(pl.Utf8).alias(smiles_col),
                pl.col(target_col).cast(pl.Float32).alias(target_col)
            ])
            
        return clean_df
    
    def validate_data_structure(self) -> bool:
        """Validate data structure before analysis"""
        print("\n[SEARCH] Validating data structure...")
        
        issues = []
        
        # Check datasets
        if not self.datasets:
            issues.append("No datasets loaded")
        
        # Check each dataset
        for name, dataset_info in self.datasets.items():
            if 'data' not in dataset_info:
                issues.append(f"{name}: Missing 'data' field")
            if 'is_test_only' not in dataset_info:
                issues.append(f"{name}: Missing 'is_test_only' field")
        
        # Check splits
        if hasattr(self, 'splits') and self.splits:
            for name, split_info in self.splits.items():
                if 'smiles' not in split_info:
                    issues.append(f"{name}: Missing 'smiles' field")
                if 'targets' not in split_info:
                    issues.append(f"{name}: Missing 'targets' field")
        
        if issues:
            print("[ERROR] Validation issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("[CHECK] Data structure validation passed")
            return True