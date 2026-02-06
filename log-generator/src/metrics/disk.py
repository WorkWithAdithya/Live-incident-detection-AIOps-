import psutil

def get_disk_usage():
    return psutil.disk_usage('/').percent
