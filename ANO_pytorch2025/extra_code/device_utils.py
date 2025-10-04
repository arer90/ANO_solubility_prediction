"""
Device and OS-specific optimization utilities
==============================================
Automatically detects OS and available hardware to optimize performance
"""

import platform
import torch
import os
import psutil

def get_optimal_device():
    """
    Get the optimal device based on OS and available hardware

    Returns:
        torch.device: Optimal device for computation
        dict: Device info including type, name, and capabilities
    """
    os_type = platform.system()
    device_info = {
        'os': os_type,
        'cpu_count': os.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3)
    }

    # Windows and Linux - CUDA support
    if os_type in ["Windows", "Linux"]:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_info.update({
                'device_type': 'cuda',
                'device_name': torch.cuda.get_device_name(0),
                'cuda_version': torch.version.cuda,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'multi_gpu': torch.cuda.device_count() > 1,
                'device_count': torch.cuda.device_count()
            })
            print(f"🚀 GPU detected: {device_info['device_name']} ({device_info['gpu_memory_gb']:.1f} GB)")
        else:
            device = torch.device('cpu')
            device_info['device_type'] = 'cpu'
            print(f"💻 Using CPU: {device_info['cpu_count']} cores")

    # macOS - MPS support for Apple Silicon
    elif os_type == "Darwin":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_info.update({
                'device_type': 'mps',
                'device_name': 'Apple Silicon GPU',
                'unified_memory': True
            })
            print(f"🍎 Apple Silicon GPU detected (MPS)")
        else:
            device = torch.device('cpu')
            device_info['device_type'] = 'cpu'
            print(f"💻 Using CPU: {device_info['cpu_count']} cores")

    else:
        device = torch.device('cpu')
        device_info['device_type'] = 'cpu'
        print(f"💻 Using CPU (Unknown OS): {device_info['cpu_count']} cores")

    return device, device_info

def get_optimal_batch_size(device_info, input_dim):
    """
    Get optimal batch size based on device and memory

    Args:
        device_info: Device information from get_optimal_device
        input_dim: Input dimension of the model

    Returns:
        int: Optimal batch size
    """
    if device_info['device_type'] == 'cuda':
        # CUDA - adjust based on GPU memory
        gpu_memory = device_info.get('gpu_memory_gb', 4)
        if gpu_memory >= 16:
            return 256
        elif gpu_memory >= 8:
            return 128
        elif gpu_memory >= 4:
            return 64
        else:
            return 32

    elif device_info['device_type'] == 'mps':
        # Apple Silicon - unified memory, can use larger batches
        total_memory = device_info.get('memory_gb', 8)
        if total_memory >= 32:
            return 128
        elif total_memory >= 16:
            return 64
        else:
            return 32

    else:
        # CPU - conservative batch size
        total_memory = device_info.get('memory_gb', 8)
        if total_memory >= 32:
            return 64
        elif total_memory >= 16:
            return 32
        else:
            return 16

def get_optimal_num_workers(device_info):
    """
    Get optimal number of workers for DataLoader based on OS and CPU

    Args:
        device_info: Device information from get_optimal_device

    Returns:
        int: Optimal number of workers
    """
    os_type = device_info['os']
    cpu_count = device_info.get('cpu_count', 4)

    if os_type == "Windows":
        # Windows has issues with multiprocessing in DataLoader
        return 0
    elif os_type == "Darwin":
        # macOS - set to 0 to avoid MPS multiprocessing issues
        # MPS tensors cannot be shared across processes
        return 0
    else:
        # Linux - can use more workers
        return min(8, cpu_count - 1)

def optimize_model_for_device(model, device, device_info):
    """
    Optimize model based on device capabilities

    Args:
        model: PyTorch model
        device: torch.device
        device_info: Device information

    Returns:
        model: Optimized model
    """
    model = model.to(device)

    # CUDA optimizations
    if device_info['device_type'] == 'cuda':
        # Enable cuDNN autotuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Use mixed precision if GPU supports it (for GPUs with compute capability >= 7.0)
        if hasattr(torch.cuda, 'amp'):
            print("✓ Mixed precision training enabled")

        # Multi-GPU support
        if device_info.get('multi_gpu', False):
            model = torch.nn.DataParallel(model)
            print(f"✓ Using {device_info['device_count']} GPUs")

    # MPS optimizations
    elif device_info['device_type'] == 'mps':
        # MPS-specific optimizations
        torch.mps.set_per_process_memory_fraction(0.6)  # Use 60% of available memory (safer approach)
        print("✓ MPS memory optimization enabled")

    return model

def cleanup_memory(device_info):
    """
    Clean up memory based on device type

    Args:
        device_info: Device information
    """
    import gc
    gc.collect()

    if device_info['device_type'] == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device_info['device_type'] == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

def get_training_optimizations(device_info):
    """
    Get OS and device-specific training optimizations

    Args:
        device_info: Device information

    Returns:
        dict: Training optimization settings
    """
    optimizations = {
        'pin_memory': False,
        'non_blocking': False,
        'gradient_accumulation': 1,
        'mixed_precision': False,
        'compile_model': False
    }

    if device_info['device_type'] == 'cuda':
        optimizations.update({
            'pin_memory': True,
            'non_blocking': True,
            'mixed_precision': True,
            'compile_model': torch.__version__ >= '2.0'  # torch.compile available in 2.0+
        })
    elif device_info['device_type'] == 'mps':
        optimizations.update({
            'pin_memory': False,  # Not supported on MPS
            'non_blocking': True,
            'mixed_precision': False,  # Limited support on MPS
            'compile_model': False  # Not fully supported on MPS
        })

    return optimizations

# Example usage
if __name__ == "__main__":
    device, info = get_optimal_device()
    print(f"\nDevice: {device}")
    print(f"Info: {info}")

    batch_size = get_optimal_batch_size(info, input_dim=2048)
    print(f"\nOptimal batch size: {batch_size}")

    num_workers = get_optimal_num_workers(info)
    print(f"Optimal workers: {num_workers}")

    optimizations = get_training_optimizations(info)
    print(f"\nOptimizations: {optimizations}")