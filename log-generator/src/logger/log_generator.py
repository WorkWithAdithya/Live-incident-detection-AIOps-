from datetime import datetime
from src.metrics.cpu import get_cpu_usage
from src.metrics.memory import get_memory_usage
from src.metrics.disk import get_disk_usage
def generate_log():
    return {
        "TimeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CPU-Usage-Percentage": get_cpu_usage(),
        "Memory-Usage-Percentage": get_memory_usage(),
        "Disk-Usage-Percentage": get_disk_usage()
    }
