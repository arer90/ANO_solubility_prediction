import os
import platform
import polars as pl
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
import chardet
import warnings
import gc
import psutil
from functools import lru_cache, wraps
import tracemalloc
warnings.filterwarnings('ignore')

# 메모리 모니터링 데코레이터
def monitor_memory(func):
    """메모리 사용량을 모니터링하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if kwargs.get('verbose', False):
            print(f"  Memory - Current: {current / (1 << 20):.1f}MB, Peak: {peak / (1 << 20):.1f}MB")
        
        return result
    return wrapper

def load_data(data_folder: str = "data", verbose: bool = False, details: bool = False, 
              parallel: bool = True, chunk_size: int = 100000) -> Dict[str, pl.DataFrame]:
    """
    Optimized data loader - reads all files and returns clean DataFrames
    
    Parameters:
    -----------
    data_folder : str - target folder containing data files
    verbose : bool - show processing details
    details : bool - show detailed file information
    parallel : bool - use parallel processing
    chunk_size : int - chunk size for large file processing
    
    Returns:
    --------
    Dict[str, pl.DataFrame] - {filename_without_extension: DataFrame}
    Each DataFrame contains: [filename, target_x, target_y]
    """
    
    # Find data folder automatically
    data_path = _find_data_folder(data_folder)
    
    if verbose:
        print(f"Data folder: {data_path}")
        print(f"OS: {platform.system()}, CPU cores: {mp.cpu_count()}")
        print(f"Available memory: {psutil.virtual_memory().available / (1 << 30):.1f}GB")
    
    # Get supported files
    files = _get_files(data_path, verbose)
    
    if not files:
        raise ValueError(f"No supported files found in {data_path}")
    
    if verbose:
        print(f"Found {len(files)} files")
        if details:
            for f in files:
                size_mb = f.stat().st_size / (1 << 20)  # 정확한 MB 변환
                print(f"  {f.name}: {size_mb:.2f} MB")
    
    # Process files
    if parallel and len(files) > 1:
        results = _process_parallel_optimized(files, verbose, chunk_size)
    else:
        results = _process_sequential_optimized(files, verbose, chunk_size)
    
    # Create final dictionary
    df_dict = {}
    success_count = 0
    failed_count = 0
    
    for result in results:
        if result is not None:
            filename, df = result
            # Remove extension from filename
            clean_name = Path(filename).stem
            df_dict[clean_name] = df
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary
    _print_summary(df_dict, success_count, failed_count, verbose, details)
    
    return df_dict

def _find_data_folder(data_folder: str) -> Path:
    """Find data folder in current or parent directories"""
    current = Path.cwd()
    
    # Try current directory
    if (current / data_folder).exists():
        return current / data_folder
    
    # Try parent directories (up to 3 levels)
    for i in range(1, 4):
        parent = current
        for _ in range(i):
            parent = parent.parent
        if (parent / data_folder).exists():
            return parent / data_folder
    
    # Return original path (will cause error if not found)
    return Path(data_folder)

def _get_files(data_path: Path, verbose: bool) -> List[Path]:
    """Get supported files from data folder"""
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")
    
    supported_extensions = {'.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json', '.parquet'}
    files = []
    
    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    
    # Sort by size (small files first)
    return sorted(files, key=lambda x: x.stat().st_size)

@monitor_memory
def _process_single_file(file_path: Path, chunk_size: int = 100000) -> Optional[tuple]:
    """Process a single file and return (filename, DataFrame) or None"""
    try:
        # Detect encoding (cached)
        encoding = _detect_encoding_cached(str(file_path))
        
        # Debug print for problematic files
        if any(name in file_path.name.lower() for name in ['curated', 'huusk']):
            print(f"    Processing {file_path.name} with encoding: {encoding}")
        
        # Read file (with chunk support for large files)
        file_size_mb = file_path.stat().st_size / (1 << 20)
        
        if file_size_mb > 100:  # Large file threshold: 100MB
            df = _read_large_file(file_path, encoding, chunk_size)
        else:
            df = _read_file(file_path, encoding)
            
        if df is None:
            print(f"    Failed to read {file_path.name}")
            return None
        
        if any(name in file_path.name.lower() for name in ['curated', 'huusk']):
            print(f"    Successfully read {file_path.name}: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
        
        # Clean DataFrame using optimized method
        clean_df = _clean_dataframe_optimized(df, file_path.name)
        
        # Clean up original dataframe
        del df
        _dynamic_gc_collect()
        
        return (file_path.name, clean_df)
        
    except Exception as e:
        print(f"    Error processing {file_path.name}: {str(e)}")
        return None

@lru_cache(maxsize=128)
def _detect_encoding_cached(file_path_str: str) -> str:
    """Cached encoding detection"""
    return _detect_encoding(Path(file_path_str))

def _detect_encoding(file_path: Path) -> str:
    """Enhanced encoding detection with dynamic sample size"""
    # Dynamic sample size based on file size
    file_size = file_path.stat().st_size
    sample_size = min(file_size, 65536)  # Max 64KB sample
    
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16', 'utf-32']
    
    try:
        # First try chardet with dynamic sample
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
        
        result = chardet.detect(raw_data)
        detected_encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0.0)
        
        if confidence > 0.8 and detected_encoding:
            return detected_encoding
        
        # If confidence is low, try common encodings
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
                
    except Exception:
        pass
    
    return 'utf-8'

def _read_file(file_path: Path, encoding: str) -> Optional[pl.DataFrame]:
    """Enhanced file reading with better error handling"""
    ext = file_path.suffix.lower()
    
    encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for enc in encodings_to_try:
        try:
            if ext == '.csv':
                return pl.read_csv(file_path, encoding=enc, ignore_errors=True, 
                                 infer_schema_length=10000, low_memory=False)
            elif ext == '.tsv':
                return pl.read_csv(file_path, separator='\t', encoding=enc, 
                                 ignore_errors=True, infer_schema_length=10000)
            elif ext == '.txt':
                return pl.read_csv(file_path, encoding=enc, ignore_errors=True,
                                 infer_schema_length=10000)
            elif ext in ['.xlsx', '.xls']:
                pd_df = pd.read_excel(file_path, engine='openpyxl' if ext == '.xlsx' else 'xlrd')
                df = pl.from_pandas(pd_df)
                del pd_df
                _dynamic_gc_collect()
                return df
            elif ext == '.json':
                return pl.read_json(file_path)
            elif ext == '.parquet':
                return pl.read_parquet(file_path)
            else:
                return None
                
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # For non-encoding errors, try pandas as fallback
            if ext == '.csv':
                try:
                    pd_df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip', low_memory=False)
                    df = pl.from_pandas(pd_df)
                    del pd_df
                    _dynamic_gc_collect()
                    return df
                except:
                    continue
            else:
                continue
    
    return None

def _read_large_file(file_path: Path, encoding: str, chunk_size: int) -> Optional[pl.DataFrame]:
    """Read large CSV files in chunks"""
    try:
        chunks = []
        reader = pl.read_csv_batched(
            file_path, 
            encoding=encoding, 
            batch_size=chunk_size,
            ignore_errors=True,
            infer_schema_length=10000
        )
        
        for chunk in reader:
            chunks.append(chunk)
            # Check memory usage
            if psutil.virtual_memory().percent > 80:
                _dynamic_gc_collect()
        
        return pl.concat(chunks) if chunks else None
        
    except Exception as e:
        print(f"    Error reading large file {file_path.name}: {e}")
        return _read_file(file_path, encoding)  # Fallback to regular reading

def _clean_dataframe_optimized(df: pl.DataFrame, filename: str) -> pl.DataFrame:
    """Optimized DataFrame cleaning with better memory efficiency"""
    
    filename_lower = filename.lower()
    
    # File-specific column mapping
    column_mappings = {
        'ws496': ('SMILES', 'exp'),
        'delaney': ('smiles', 'measured log solubility in mols per litre'),
        'lovric': ('isomeric_smiles', 'logS0'),
        'huusk': ('SMILES', 'Solubility'),
        'curated-solubility': ('SMILES', 'Solubility'),
        'sampl': ('smiles', 'expt'),
        'lipophilicity': ('smiles', 'exp'),
    }
    
    # Find matching mapping
    target_x_col, target_y_col = None, None
    for key, (x_col, y_col) in column_mappings.items():
        if key in filename_lower:
            target_x_col, target_y_col = x_col, y_col
            break
    
    # Generic fallback if no specific mapping found
    if not target_x_col:
        target_x_col = _find_column(df, ['SMILES', 'smiles', 'smi', 'isomeric_smiles', 'canonical_smiles'])
        target_y_col = _find_column(df, ['Solubility', 'exp', 'expt', 'logS0', 
                                        'measured log solubility in mols per litre',
                                        'y', 'value', 'target', 'logS', 'logs', 'logP'])
    
    # Create clean DataFrame using Polars native operations (more efficient)
    expressions = [
        pl.lit(Path(filename).stem).alias("filename"),
    ]
    
    # Add target_x column
    if target_x_col and target_x_col in df.columns:
        expressions.append(
            df[target_x_col].cast(pl.String, strict=False).alias("target_x")
        )
    else:
        expressions.append(pl.lit(None).alias("target_x"))
    
    # Add target_y column
    if target_y_col and target_y_col in df.columns:
        expressions.append(
            df[target_y_col].cast(pl.Float64, strict=False).alias("target_y")
        )
    else:
        expressions.append(pl.lit(None).alias("target_y"))
    
    # Create new dataframe efficiently
    clean_df = df.select(expressions)
    
    return clean_df

def _find_column(df: pl.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive)"""
    df_columns_lower = [col.lower() for col in df.columns]
    
    for name in possible_names:
        if name.lower() in df_columns_lower:
            idx = df_columns_lower.index(name.lower())
            return df.columns[idx]
    
    return None

def _dynamic_gc_collect():
    """Collect garbage only when memory usage is high"""
    if psutil.virtual_memory().percent > 70:
        gc.collect()

def _process_sequential_optimized(files: List[Path], verbose: bool, chunk_size: int) -> List[Optional[tuple]]:
    """Optimized sequential processing with memory management"""
    if verbose:
        print("Processing files sequentially...")
    
    results = []
    for i, file_path in enumerate(files):
        if verbose:
            print(f"  Processing {i+1}/{len(files)}: {file_path.name}")
        
        result = _process_single_file(file_path, chunk_size)
        results.append(result)
        
        # Dynamic memory cleanup
        _dynamic_gc_collect()
    
    return results

def _process_parallel_optimized(files: List[Path], verbose: bool, chunk_size: int) -> List[Optional[tuple]]:
    """Optimized parallel processing based on file sizes"""
    # Leave one CPU for system
    max_workers = min(4, len(files), max(1, mp.cpu_count() - 1))
    
    # Categorize files by size
    small_threshold = 10 * (1 << 20)  # 10MB
    large_threshold = 100 * (1 << 20)  # 100MB
    
    small_files = [f for f in files if f.stat().st_size < small_threshold]
    medium_files = [f for f in files if small_threshold <= f.stat().st_size < large_threshold]
    large_files = [f for f in files if f.stat().st_size >= large_threshold]
    
    if verbose:
        print(f"Processing files in parallel (max workers: {max_workers})")
        print(f"  Small files (<10MB): {len(small_files)}")
        print(f"  Medium files (10-100MB): {len(medium_files)}")
        print(f"  Large files (>100MB): {len(large_files)}")
    
    results = []
    
    try:
        # Small files: ThreadPool (I/O bound)
        if small_files:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                small_results = list(executor.map(
                    lambda f: _process_single_file(f, chunk_size), 
                    small_files
                ))
                results.extend(small_results)
        
        # Medium files: ThreadPool with fewer workers
        if medium_files:
            with ThreadPoolExecutor(max_workers=max(2, max_workers // 2)) as executor:
                medium_results = list(executor.map(
                    lambda f: _process_single_file(f, chunk_size), 
                    medium_files
                ))
                results.extend(medium_results)
        
        # Large files: Process sequentially to avoid memory issues
        if large_files:
            if verbose:
                print("  Processing large files sequentially to manage memory...")
            for file in large_files:
                result = _process_single_file(file, chunk_size)
                results.append(result)
                _dynamic_gc_collect()
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Parallel processing failed, switching to sequential: {e}")
        return _process_sequential_optimized(files, verbose, chunk_size)

def _print_summary(df_dict: Dict[str, pl.DataFrame], success: int, failed: int, 
                  verbose: bool, details: bool):
    """Print loading summary with memory statistics"""
    
    total_rows = sum(len(df) for df in df_dict.values())
    total_memory = sum(df.estimated_size() for df in df_dict.values()) / (1 << 20)  # MB
    
    print("\n" + "="*50)
    print("Data Loading Complete!")
    print("="*50)
    print(f"Success: {success} files")
    print(f"Failed: {failed} files")
    print(f"Total data: {total_rows:,} rows")
    print(f"Memory usage: {total_memory:.1f} MB")
    print(f"System memory: {psutil.virtual_memory().percent:.1f}% used")
    
    if verbose or details:
        print(f"\nLoaded DataFrames:")
        for name, df in sorted(df_dict.items()):
            rows = len(df)
            memory = df.estimated_size() / (1 << 20)
            
            # Check data availability
            has_x = df["target_x"].drop_nulls().len() > 0
            has_y = df["target_y"].drop_nulls().len() > 0
            
            print(f"  {name}: {rows:,} rows ({memory:.1f} MB)")
            print(f"    target_x: {'✓' if has_x else '✗'} ({df['target_x'].drop_nulls().len()} values)")
            print(f"    target_y: {'✓' if has_y else '✗'} ({df['target_y'].drop_nulls().len()} values)")
            
            if details and has_y:
                try:
                    y_stats = df.select([
                        pl.col("target_y").drop_nulls().min().alias("min"),
                        pl.col("target_y").drop_nulls().max().alias("max"),
                        pl.col("target_y").drop_nulls().mean().alias("mean"),
                        pl.col("target_y").drop_nulls().std().alias("std")
                    ])
                    
                    if len(y_stats) > 0:
                        min_val = y_stats["min"][0]
                        max_val = y_stats["max"][0]
                        mean_val = y_stats["mean"][0]
                        std_val = y_stats["std"][0]
                        
                        if all(x is not None for x in [min_val, max_val, mean_val]):
                            print(f"    target_y stats: min={min_val:.3f}, max={max_val:.3f}, "
                                  f"mean={mean_val:.3f}, std={std_val:.3f if std_val else 0:.3f}")
                except:
                    pass
    
    print(f"\nUsage example:")
    print(f"  df_dict = load_data()")
    if df_dict:
        first_name = list(df_dict.keys())[0]
        print(f"  {first_name}_df = df_dict['{first_name}']")
        print(f"  print({first_name}_df.head())")


# Performance testing utilities
def benchmark_loading(data_folder: str = "data", runs: int = 3):
    """Benchmark data loading performance"""
    import time
    
    print("Running performance benchmark...")
    
    # Test different configurations
    configs = [
        ("Sequential", {"parallel": False}),
        ("Parallel", {"parallel": True}),
        ("Parallel + Large chunks", {"parallel": True, "chunk_size": 200000}),
    ]
    
    for config_name, config_params in configs:
        times = []
        
        for i in range(runs):
            start = time.time()
            df_dict = load_data(data_folder, verbose=False, **config_params)
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Clean up
            del df_dict
            gc.collect()
        
        avg_time = sum(times) / len(times)
        print(f"\n{config_name}:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Times: {[f'{t:.2f}s' for t in times]}")


# Usage examples
if __name__ == "__main__":
    # Basic usage
    df_dict = load_data()
    
    # With all options
    # df_dict = load_data(verbose=True, details=True, parallel=True, chunk_size=150000)
    
    # Benchmark performance
    # benchmark_loading()
    
    # Access individual DataFrames
    for name, df in df_dict.items():
        print(f"\n{name} dataset:")
        print(f"Shape: {df.shape}")
        print(df.head(3))
        
    print(f"\nTotal datasets loaded: {len(df_dict)}")
    print(f"Available datasets: {list(df_dict.keys())}")