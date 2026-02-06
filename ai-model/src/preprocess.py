from sklearn.preprocessing import StandardScaler
import joblib

def fit_scaler(df, scaler_path):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    joblib.dump(scaler, scaler_path)
    return scaled_data

def load_and_transform(df, scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler.transform(df)
