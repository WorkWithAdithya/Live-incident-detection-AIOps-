import psycopg2
from src.config import DATABASE_URL

def init_db():
    """
    Creates the system_logs table in NeonPostgreSQL if it doesn't already exist.
    Run this once before starting the logger.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        create_table_query = """
            CREATE TABLE IF NOT EXISTS system_logs (
                id              SERIAL PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                cpu_usage       FLOAT NOT NULL,
                memory_usage    FLOAT NOT NULL,
                disk_usage      FLOAT NOT NULL
            );
        """

        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialized. Table 'system_logs' is ready.")

    except Exception as e:
        print(f"❌ Database initialization error: {e}")


if __name__ == "__main__":
    init_db()