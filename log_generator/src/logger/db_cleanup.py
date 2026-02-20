import psycopg2
from src.config import DATABASE_URL, LOG_RETENTION_HOURS
from datetime import datetime, timedelta


def cleanup_old_logs():
    """
    Deletes logs older than the configured retention period.
    Called periodically to keep database size manageable.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Calculate cutoff time
        cutoff_time = datetime.now() - timedelta(hours=LOG_RETENTION_HOURS)

        # Delete old records
        delete_query = """
            DELETE FROM system_logs
            WHERE timestamp < %s
        """
        
        cursor.execute(delete_query, (cutoff_time,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()

        if deleted_count > 0:
            print(f"🗑️  Cleaned up {deleted_count} old log entries (older than {LOG_RETENTION_HOURS}h)")
        
        return deleted_count

    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return 0


def cleanup_by_count(keep_last_n=10000):
    """
    Alternative: Keep only the last N records, delete older ones.
    Useful if you want to maintain a fixed dataset size.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        delete_query = """
            DELETE FROM system_logs
            WHERE id NOT IN (
                SELECT id FROM system_logs
                ORDER BY timestamp DESC
                LIMIT %s
            )
        """
        
        cursor.execute(delete_query, (keep_last_n,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()

        if deleted_count > 0:
            print(f"🗑️  Cleaned up {deleted_count} old entries (kept last {keep_last_n})")
        
        return deleted_count

    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return 0


def get_log_count():
    """
    Returns the total number of logs in the database.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM system_logs")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return count
        
    except Exception as e:
        print(f"❌ Error getting log count: {e}")
        return 0


if __name__ == "__main__":
    # Manual cleanup script
    print("Current log count:", get_log_count())
    cleanup_old_logs()
    print("Log count after cleanup:", get_log_count())