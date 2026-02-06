import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from src.preprocess import fit_scaler


# Paths
TRAIN_DATA_PATH = "/home/msis/Desktop/AIOps/ai-model/data/system_performance_metrics.csv"
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "CPU-Usage-Percentage",
    "Memory-Usage-Percentage",
    "Disk-Usage-Percentage"
]

def main():
    print("📥 Loading training data...")
    df = pd.read_csv(TRAIN_DATA_PATH)

    df = df[FEATURES].dropna()

    print("🔄 Scaling training data...")
    X_scaled = fit_scaler(df, SCALER_PATH)

    print("🤖 Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    joblib.dump(model, MODEL_PATH)
    print("✅ Model training completed and saved.")

if __name__ == "__main__":
    main()
