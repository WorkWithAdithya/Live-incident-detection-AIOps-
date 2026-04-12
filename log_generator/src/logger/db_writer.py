import psycopg2
from psycopg2 import pool
from src.config import DATABASE_URL

# Create a persistent connection pool once at startup.
# min=1, max=3 is plenty for a single-threaded logger.
_pool = psycopg2.pool.SimpleConnectionPool(1, 3, DATABASE_URL)


def write_to_db(log_data):
    """
    Inserts a single log entry into the system_logs table in NeonPostgreSQL.
    Uses a persistent connection pool — no reconnection overhead per write.
    """
    conn = None
    try:
        conn = _pool.getconn()
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

    except Exception as e:
        print(f"❌ Database write error: {e}")
        if conn:
            conn.rollback()

    finally:
        if conn:
            _pool.putconn(conn)  # Return connection to pool, don't close it