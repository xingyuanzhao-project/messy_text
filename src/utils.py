import sys
import logging
import psutil
import os
from typing import Any

def setup_logger(log_file: str = "processing.log") -> logging.Logger:
    """
    Sets up a file-based logger for long-running server jobs.
    """
    logger = logging.getLogger("MessyTextProcessor")
    logger.setLevel(logging.INFO)
    
    # File handler (persists logs)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Stream handler (for stdout/slurm output)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def log_memory_usage(logger: logging.Logger, context: str = ""):
    """
    Logs current memory usage of the process.
    Replaces the notebook's 'show_resource' with system-level tracking.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[Memory] {context}: {mem_mb:.2f} MB")

def get_deep_size(obj: Any, seen=None) -> int:
    """
    Recursively calculates object size (from notebook logic).
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i, seen) for i in obj])
    return size


def check_gpu_info(logger: logging.Logger):
    """
    Get actual GPU hardware info via nvidia-smi.
    Returns list of GPU info dicts or None if no GPU.
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr.strip()}")
            return None
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            info = {
                'id': int(parts[0]),
                'name': parts[1],
                'total_memory_mb': float(parts[2]),
                'used_memory_mb': float(parts[3]),
                'free_memory_mb': float(parts[4]),
                'utilization_pct': float(parts[5]) if parts[5] != '[N/A]' else 0
            }
            gpu_info.append(info)
            logger.info(f"GPU {info['id']}: {info['name']} | "
                       f"Memory: {info['used_memory_mb']:.0f}/{info['total_memory_mb']:.0f} MB used | "
                       f"Utilization: {info['utilization_pct']:.0f}%")
        
        return gpu_info if gpu_info else None
    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
        return None
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return None


def check_vllm_server(client, expected_model: str, logger: logging.Logger):
    """
    Check vLLM server connectivity and available models.
    Matches notebook pattern: list models + test request.
    
    Returns:
        tuple: (success: bool, available_models: list, test_result: str or None)
    """
    available_models = []
    
    # 1. List available models (from notebook: client.models.list())
    try:
        models = client.models.list()
        for model in models.data:
            available_models.append(model.id)
            logger.info(f"vLLM available model: {model.id}")
    except Exception as e:
        logger.error(f"Cannot connect to vLLM server: {e}")
        return False, [], None
    
    # 2. Check if expected model is available
    if expected_model not in available_models:
        logger.error(f"Expected model '{expected_model}' not in available models: {available_models}")
        return False, available_models, None
    
    # 3. Test request (from notebook: 1+1 test)
    try:
        response = client.chat.completions.create(
            model=expected_model,
            messages=[{'role': 'user', 'content': 'What is 1+1? Reply with just the number.'}],
            max_tokens=10,
            temperature=0.0
        )
        test_result = response.choices[0].message.content.strip()
        logger.info(f"vLLM test request successful: 1+1 = {test_result}")
        return True, available_models, test_result
    except Exception as e:
        logger.error(f"vLLM test request failed: {e}")
        return False, available_models, None