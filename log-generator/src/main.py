import time
from src.config import LOG_INTERVAL_SECONDS
from src.logger.log_generator import generate_log
from src.logger.db_writer import write_to_db
from src.logger.db_init import init_db


def main():
    print("🚀 Log Generator Started...")

    # Initialize the database table if it doesn't exist
    init_db()

    while True:
        log = generate_log()
        write_to_db(log)
        print("✅ Log written to NeonDB:", log)
        time.sleep(LOG_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
