#!/usr/bin/env python3
"""
Manual cleanup script for system logs.
Run this anytime to clean up old data from the database.

Usage:
    python3 cleanup_manual.py              # Use default retention from .env
    python3 cleanup_manual.py --hours 12   # Keep only last 12 hours
    python3 cleanup_manual.py --count 5000 # Keep only last 5000 records
"""

import argparse
from src.logger.db_cleanup import cleanup_old_logs, cleanup_by_count, get_log_count
from src.config import LOG_RETENTION_HOURS


def main():
    parser = argparse.ArgumentParser(description='Cleanup old system logs')
    parser.add_argument('--hours', type=int, help='Keep logs from last N hours')
    parser.add_argument('--count', type=int, help='Keep only last N records')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("System Logs Cleanup Tool")
    print("=" * 50)
    print(f"Current log count: {get_log_count()}")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No data will be deleted")
        print()
    
    if args.count:
        print(f"Keeping only last {args.count} records...")
        if not args.dry_run:
            deleted = cleanup_by_count(args.count)
            print(f"✅ Deleted {deleted} records")
    elif args.hours:
        print(f"Keeping logs from last {args.hours} hours...")
        if not args.dry_run:
            # Temporarily override retention hours
            import src.config as config
            original_retention = config.LOG_RETENTION_HOURS
            config.LOG_RETENTION_HOURS = args.hours
            deleted = cleanup_old_logs()
            config.LOG_RETENTION_HOURS = original_retention
            print(f"✅ Deleted {deleted} records")
    else:
        print(f"Using default retention: {LOG_RETENTION_HOURS} hours...")
        if not args.dry_run:
            deleted = cleanup_old_logs()
            print(f"✅ Deleted {deleted} records")
    
    print()
    print(f"Final log count: {get_log_count()}")
    print("=" * 50)


if __name__ == "__main__":
    main()