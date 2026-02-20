import time
import threading
from src.config import LOG_INTERVAL_SECONDS, CLEANUP_INTERVAL_SECONDS
from src.logger.log_generator import generate_log
from src.logger.db_writer import write_to_db
from src.logger.db_init import init_db
from src.logger.db_cleanup import cleanup_old_logs


def cleanup_worker():
    """
    Background thread that periodically cleans up old logs.
    Runs independently from the main logging loop.
    """
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        cleanup_old_logs()


def main():
    print("🚀 Log Generator Started...")

    # Initialize the database table if it doesn't exist
    init_db()

    # Start cleanup worker thread in background
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print(f"🧹 Cleanup worker started (runs every {CLEANUP_INTERVAL_SECONDS}s)")

    # Main logging loop
    while True:
        log = generate_log()
        write_to_db(log)
        print("✅ Log written to NeonDB:", log)
        time.sleep(LOG_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()