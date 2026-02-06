import time
from src.config import LOG_INTERVAL_SECONDS, OUTPUT_FILE
from src.logger.log_generator import generate_log
from src.logger.excel_writer import write_to_excel


def main():
    print("🚀 Log Generator Started...")
    while True:
        log = generate_log()
        write_to_excel(log, OUTPUT_FILE)
        print("Log written:", log)
        time.sleep(LOG_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
