"""
Performance Monitoring Utilities
=================================
Tracks execution time, memory usage, and GPU utilization
"""

import time
import psutil
import platform
import torch
import os
from datetime import datetime
import json
from pathlib import Path

class PerformanceMonitor:
    """Monitor performance metrics for modules"""

    def __init__(self, module_name="Unknown", output_dir="logs/performance"):
        self.module_name = module_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
        self.metrics = {}

        # System info
        self.system_info = self._get_system_info()

    def _get_system_info(self):
        """Collect system information"""
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
        }

        # GPU info
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_type'] = 'CUDA'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['gpu_available'] = True
            info['gpu_type'] = 'MPS'
            info['gpu_name'] = 'Apple Silicon GPU'
        else:
            info['gpu_available'] = False
            info['gpu_type'] = 'None'

        return info

    def start(self, task_name=""):
        """Start monitoring"""
        self.task_name = task_name
        self.start_time = time.time()

        # Memory usage
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / (1024**3)  # GB
        self.start_cpu_percent = psutil.cpu_percent(interval=0.1)

        # GPU memory
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't provide detailed memory stats
            self.start_gpu_memory = 0

        print(f"\n{'='*60}")
        print(f"🚀 Starting: {self.module_name} - {task_name}")
        print(f"{'='*60}")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💻 System: {self.system_info['platform']} ({self.system_info['cpu_count']} CPUs)")

        if self.system_info['gpu_available']:
            print(f"🎮 GPU: {self.system_info['gpu_name']} ({self.system_info['gpu_type']})")
        else:
            print(f"🎮 GPU: Not available (CPU only)")

        print(f"🧠 Initial Memory: {self.start_memory:.2f} GB")
        print(f"{'='*60}\n")

    def end(self):
        """End monitoring and report results"""
        if self.start_time is None:
            return

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Memory usage
        process = psutil.Process()
        end_memory = process.memory_info().rss / (1024**3)  # GB
        memory_used = end_memory - self.start_memory
        peak_memory = process.memory_info().rss / (1024**3)

        # CPU usage
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        avg_cpu_percent = (self.start_cpu_percent + end_cpu_percent) / 2

        # GPU memory
        gpu_memory_used = 0
        gpu_peak_memory = 0
        if torch.cuda.is_available():
            gpu_memory_used = (torch.cuda.memory_allocated() / (1024**3)) - self.start_gpu_memory
            gpu_peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        # Store metrics
        self.metrics = {
            'module': self.module_name,
            'task': self.task_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': self._format_time(elapsed_time),
            'memory_used_gb': memory_used,
            'peak_memory_gb': peak_memory,
            'avg_cpu_percent': avg_cpu_percent,
            'gpu_memory_used_gb': gpu_memory_used,
            'gpu_peak_memory_gb': gpu_peak_memory,
            'system_info': self.system_info
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Completed: {self.module_name} - {self.task_name}")
        print(f"{'='*60}")
        print(f"⏱️  Time Elapsed: {self._format_time(elapsed_time)}")
        print(f"💾 Memory Used: {memory_used:+.2f} GB (Peak: {peak_memory:.2f} GB)")
        print(f"🔥 Avg CPU Usage: {avg_cpu_percent:.1f}%")

        if self.system_info['gpu_available']:
            if self.system_info['gpu_type'] == 'CUDA':
                print(f"🎮 GPU Memory: {gpu_memory_used:+.2f} GB (Peak: {gpu_peak_memory:.2f} GB)")
                utilization = self._get_gpu_utilization()
                if utilization:
                    print(f"📊 GPU Utilization: {utilization}%")
            elif self.system_info['gpu_type'] == 'MPS':
                print(f"🎮 GPU: MPS (Apple Silicon) - Active")

        print(f"{'='*60}\n")

        # Save to log file
        self._save_log()

        return self.metrics

    def _format_time(self, seconds):
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage (NVIDIA only)"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return None

    def _save_log(self):
        """Save performance log to file"""
        log_file = self.output_dir / f"{self.module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"📝 Log saved to: {log_file}")

    def checkpoint(self, message=""):
        """Log intermediate checkpoint"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**3)
            print(f"⏸️  [{self._format_time(elapsed)}] {message} | Memory: {current_memory:.2f} GB")

# Convenience functions for module usage
def start_monitoring(module_name, task_name=""):
    """Start performance monitoring for a module"""
    monitor = PerformanceMonitor(module_name)
    monitor.start(task_name)
    return monitor

def get_device_with_monitoring():
    """Get optimal device and print device info"""
    try:
        from device_utils import get_optimal_device
        device, device_info = get_optimal_device()

        print(f"\n🖥️  Device Configuration:")
        print(f"   • Device: {device}")
        print(f"   • Type: {device_info.get('device_type', 'cpu')}")

        if device_info.get('device_type') == 'cuda':
            print(f"   • GPU: {device_info.get('device_name', 'Unknown')}")
            print(f"   • Memory: {device_info.get('gpu_memory_gb', 0):.1f} GB")
        elif device_info.get('device_type') == 'mps':
            print(f"   • GPU: Apple Silicon (MPS)")
            print(f"   • Unified Memory: {device_info.get('memory_gb', 0):.1f} GB")
        else:
            print(f"   • CPU Cores: {device_info.get('cpu_count', 0)}")
            print(f"   • RAM: {device_info.get('memory_gb', 0):.1f} GB")

        return device, device_info
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Using device: {device}")
        return device, {}

# Example usage
if __name__ == "__main__":
    # Test monitoring
    monitor = start_monitoring("Module_Test", "Sample Task")

    # Simulate some work
    import time
    for i in range(3):
        time.sleep(1)
        monitor.checkpoint(f"Step {i+1} completed")

    # End monitoring
    monitor.end()