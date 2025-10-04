from .config import SYSTEM_INFO

"""Memory optimization configuration - Complete version"""

MEMORY_CONFIG = {
    'max_memory_mb': int(SYSTEM_INFO['memory_gb'] * 1024 * 0.7),  # Use 70% of memory
    
    'max_batch_size': 1000 if SYSTEM_INFO['memory_gb'] < 8 else 5000,
    'max_chunk_size': 10000 if SYSTEM_INFO['memory_gb'] < 8 else 50000,
    
    'large_dataset_threshold': 50000 if SYSTEM_INFO['memory_gb'] >= 16 else 10000,
    'very_large_dataset_threshold': 100000 if SYSTEM_INFO['memory_gb'] >= 32 else 50000,
    
    # Add missing auto_sampling_threshold
    'auto_sampling_threshold': {
        'tanimoto': 1e6,  # 1M comparisons
        'fingerprint': 1e7,  # 10M comparisons
        'distance': 1e6,  # 1M comparisons
        'similarity': 5e5,  # 500K comparisons
        'default': 1e6
    }
}

def should_sample(n_comparisons: int, comparison_type: str = 'default') -> bool:
    """Determine whether to use automatic sampling"""
    threshold = MEMORY_CONFIG['auto_sampling_threshold'].get(comparison_type, 1e6)
    return n_comparisons > threshold

def get_optimal_batch_size(total_size: int, n_workers: int) -> int:
    """Calculate optimal batch size considering memory constraints"""
    import psutil
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    max_batch = MEMORY_CONFIG['max_batch_size']
    
    # Reduce batch size if memory is insufficient
    if available_memory < 1000:  # Less than 1GB
        max_batch = max_batch // 2
    
    batch_size = max(10, min(max_batch, total_size // (n_workers * 4)))
    return batch_size

def get_optimal_chunk_size(data_size: int, feature_size: int) -> int:
    """Calculate optimal chunk size based on data and memory constraints"""
    bytes_per_element = 4  # float32
    memory_per_chunk = data_size * feature_size * bytes_per_element / (1024**2)  # MB
    
    available_memory = SYSTEM_INFO['memory_gb'] * 1024 * 0.5  # Use 50% of memory
    optimal_chunks = max(1, int(memory_per_chunk / available_memory))
    
    return max(100, data_size // optimal_chunks)

def check_memory_usage():
    """Check current memory usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'warning': memory.percent > 80
        }
    except:
        return {
            'total_gb': 0,
            'available_gb': 0,
            'used_percent': 0,
            'warning': False
        }

def get_safe_sample_size(total_size: int, max_memory_mb: int = None) -> int:
    """Calculate safe sample size based on memory constraints"""
    if max_memory_mb is None:
        max_memory_mb = MEMORY_CONFIG['max_memory_mb']
    
    # Maximum sample size based on memory situation
    memory_status = check_memory_usage()
    
    if memory_status['warning']:
        # Smaller sample when memory is insufficient
        safe_size = min(total_size, max_memory_mb // 10)
    else:
        safe_size = min(total_size, max_memory_mb // 5)
    
    return max(100, safe_size)  # Minimum 100 samples