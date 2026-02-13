import psycopg2
from src.config import DATABASE_URL

def write_to_db(log_data):
    """
    Inserts a single log entry into the system_logs table in NeonPostgreSQL.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO system_logs (timestamp, cpu_usage, memory_usage, disk_usage)
            VALUES (%s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            log_data["TimeStamp"],
            log_data["CPU-Usage-Percentage"],
            log_data["Memory-Usage-Percentage"],
            log_data["Disk-Usage-Percentage"]
        ))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Database write error: {e}")