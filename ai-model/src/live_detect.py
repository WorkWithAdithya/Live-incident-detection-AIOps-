import pandas as pd
import time
import joblib
from src.preprocess import load_and_transform


# Paths
LIVE_DATA_PATH = "/home/msis/Desktop/AIOps/log-generator/output/system_logs.xlsx"
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "CPU-Usage-Percentage",
    "Memory-Usage-Percentage",
    "Disk-Usage-Percentage"
]

CHECK_INTERVAL = 5  # seconds

def trigger_alert(row, anomaly):
    timestamp = row.get("TimeStamp", "UNKNOWN_TIME")

    if anomaly == 1:
        print(f"[{timestamp}] 🚨 ML ANOMALY DETECTED")

    if row["CPU-Usage-Percentage"] > 90:
        print(f"[{timestamp}] 🔥 CRITICAL CPU ALERT")

    if row["Memory-Usage-Percentage"] > 90:
        print(f"[{timestamp}] 🔥 CRITICAL MEMORY ALERT")

    if row["Disk-Usage-Percentage"] > 95:
        print(f"[{timestamp}] 🔥 CRITICAL DISK ALERT")


def main():
    print("🚀 Live detection started...")
    model = joblib.load(MODEL_PATH)

    last_processed_row = 0

    while True:
        df = pd.read_excel(LIVE_DATA_PATH, engine="openpyxl")

        if len(df) <= last_processed_row:
            time.sleep(CHECK_INTERVAL)
            continue

        new_data = df.iloc[last_processed_row:]
        X = new_data[FEATURES]

        X_scaled = load_and_transform(X, SCALER_PATH)
        predictions = model.predict(X_scaled)

        for i, row in new_data.iterrows():
            anomaly = 1 if predictions[i - last_processed_row] == -1 else 0
            trigger_alert(row, anomaly)

        last_processed_row = len(df)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
