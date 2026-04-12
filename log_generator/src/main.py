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

    init_db()

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print(f"🧹 Cleanup worker started (runs every {CLEANUP_INTERVAL_SECONDS}s)")

    print(f"⏱️  Logging every {LOG_INTERVAL_SECONDS}s")

    while True:
        loop_start = time.time()           # ← record when loop began

        log = generate_log()
        write_to_db(log)
        print("✅ Log written to NeonDB:", log)

        elapsed = time.time() - loop_start
        sleep_for = LOG_INTERVAL_SECONDS - elapsed

        if sleep_for > 0:
            time.sleep(sleep_for)          # sleep only remaining time
        # if elapsed > interval, skip sleep entirely and log immediately


if __name__ == "__main__":
    main()