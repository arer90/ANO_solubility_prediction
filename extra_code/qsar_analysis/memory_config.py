from .config import SYSTEM_INFO

"""Memory optimization configuration - 완전한 버전"""

MEMORY_CONFIG = {
    'max_memory_mb': int(SYSTEM_INFO['memory_gb'] * 1024 * 0.7),  # 70% 사용
    
    'max_batch_size': 1000 if SYSTEM_INFO['memory_gb'] < 8 else 5000,
    'max_chunk_size': 10000 if SYSTEM_INFO['memory_gb'] < 8 else 50000,
    
    'large_dataset_threshold': 50000 if SYSTEM_INFO['memory_gb'] >= 16 else 10000,
    'very_large_dataset_threshold': 100000 if SYSTEM_INFO['memory_gb'] >= 32 else 50000,
    
    # 누락된 auto_sampling_threshold 추가
    'auto_sampling_threshold': {
        'tanimoto': 1e6,  # 1M comparisons
        'fingerprint': 1e7,  # 10M comparisons
        'distance': 1e6,  # 1M comparisons
        'similarity': 5e5,  # 500K comparisons
        'default': 1e6
    }
}

def should_sample(n_comparisons: int, comparison_type: str = 'default') -> bool:
    """자동 샘플링 여부 결정"""
    threshold = MEMORY_CONFIG['auto_sampling_threshold'].get(comparison_type, 1e6)
    return n_comparisons > threshold

def get_optimal_batch_size(total_size: int, n_workers: int) -> int:
    """메모리를 고려한 최적 배치 크기 계산"""
    import psutil
    
    # 사용 가능한 메모리 확인
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    max_batch = MEMORY_CONFIG['max_batch_size']
    
    # 메모리가 부족하면 배치 크기 줄이기
    if available_memory < 1000:  # 1GB 미만
        max_batch = max_batch // 2
    
    batch_size = max(10, min(max_batch, total_size // (n_workers * 4)))
    return batch_size

def get_optimal_chunk_size(data_size: int, feature_size: int) -> int:
    """데이터와 메모리에 따른 최적 청크 크기 계산"""
    bytes_per_element = 4  # float32
    memory_per_chunk = data_size * feature_size * bytes_per_element / (1024**2)  # MB
    
    available_memory = SYSTEM_INFO['memory_gb'] * 1024 * 0.5  # 50% 사용
    optimal_chunks = max(1, int(memory_per_chunk / available_memory))
    
    return max(100, data_size // optimal_chunks)

def check_memory_usage():
    """현재 메모리 사용량 체크"""
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
    """안전한 샘플 크기 계산"""
    if max_memory_mb is None:
        max_memory_mb = MEMORY_CONFIG['max_memory_mb']
    
    # 메모리 상황에 따른 최대 샘플 크기
    memory_status = check_memory_usage()
    
    if memory_status['warning']:
        # 메모리 부족시 더 작은 샘플
        safe_size = min(total_size, max_memory_mb // 10)
    else:
        safe_size = min(total_size, max_memory_mb // 5)
    
    return max(100, safe_size)  # 최소 100개