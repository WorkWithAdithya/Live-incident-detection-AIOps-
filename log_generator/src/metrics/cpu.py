import psutil

# Prime the CPU counter once at import time.
# The first call always returns 0.0 and is discarded.
psutil.cpu_percent(interval=None)

def get_cpu_usage():
    # interval=None → non-blocking, returns usage since last call instantly
    return psutil.cpu_percent(interval=None)