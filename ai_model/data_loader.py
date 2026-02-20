"""
Data Loader - Load data from CSV or NeonPostgreSQL database
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, database_url=None):
        self.database_url = database_url
    
    def load_from_csv(self, csv_path="data/synthetic_data.csv"):
        """Load data from CSV file"""
        print(f"📁 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
        return df
    
    def load_from_database(self, limit=None, hours_ago=None):
        """Load data from NeonPostgreSQL database"""
        if not self.database_url:
            raise ValueError("Database URL not provided")
        
        print("🗄️  Loading data from database...")
        
        try:
            conn = psycopg2.connect(self.database_url)
            
            query = "SELECT timestamp, cpu_usage, memory_usage, disk_usage FROM system_logs"
            
            # Add time filter if specified
            if hours_ago:
                cutoff_time = datetime.now() - timedelta(hours=hours_ago)
                query += f" WHERE timestamp >= '{cutoff_time}'"
            
            query += " ORDER BY timestamp ASC"
            
            # Add limit if specified
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(df)} rows from database")
            return df
            
        except Exception as e:
            print(f"❌ Database loading error: {e}")
            return None
    
    def get_statistics(self, df):
        """Print dataset statistics"""
        print("\n📊 Dataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\n  CPU Usage:")
        print(f"    Mean: {df['cpu_usage'].mean():.2f}%")
        print(f"    Min:  {df['cpu_usage'].min():.2f}%")
        print(f"    Max:  {df['cpu_usage'].max():.2f}%")
        print(f"\n  Memory Usage:")
        print(f"    Mean: {df['memory_usage'].mean():.2f}%")
        print(f"    Min:  {df['memory_usage'].min():.2f}%")
        print(f"    Max:  {df['memory_usage'].max():.2f}%")
        print(f"\n  Disk Usage:")
        print(f"    Mean: {df['disk_usage'].mean():.2f}%")
        print(f"    Min:  {df['disk_usage'].min():.2f}%")
        print(f"    Max:  {df['disk_usage'].max():.2f}%")
        
        if 'label' in df.columns:
            print(f"\n  Label Distribution:")
            label_counts = df['label'].value_counts().sort_index()
            label_names = {0: 'Normal', 1: 'Warning', 2: 'Critical', 3: 'Anomaly'}
            for label, count in label_counts.items():
                print(f"    {label} ({label_names.get(label, 'Unknown')}): {count}")


if __name__ == "__main__":
    # Test loading from CSV
    loader = DataLoader()
    df = loader.load_from_csv()
    loader.get_statistics(df)