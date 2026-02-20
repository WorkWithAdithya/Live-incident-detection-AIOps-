"""
Data Preprocessing for Time-Series Model
Creates sliding windows and normalizes data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class TimeSeriesPreprocessor:
    def __init__(self, window_size=12, prediction_horizon=5):
        """
        Args:
            window_size: Number of past timestamps to use (default: 12 = 60 seconds at 5s intervals)
            prediction_horizon: Number of future timestamps to predict (default: 5 = 25 seconds)
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def create_sequences(self, data, labels=None):
        """
        Create sliding window sequences from time-series data
        
        Args:
            data: numpy array of shape (n_samples, n_features)
            labels: numpy array of labels (optional)
        
        Returns:
            X: Input sequences (n_sequences, window_size, n_features)
            y_forecast: Target values for forecasting (n_sequences, prediction_horizon, n_features)
            y_class: Classification labels (n_sequences,)
        """
        X, y_forecast, y_class = [], [], []
        
        total_window = self.window_size + self.prediction_horizon
        
        for i in range(len(data) - total_window + 1):
            # Input sequence
            X.append(data[i:i + self.window_size])
            
            # Forecasting target (next prediction_horizon values)
            y_forecast.append(data[i + self.window_size:i + total_window])
            
            # Classification label (label of the last point in sequence)
            if labels is not None:
                y_class.append(labels[i + self.window_size - 1])
        
        X = np.array(X)
        y_forecast = np.array(y_forecast)
        
        if labels is not None:
            y_class = np.array(y_class)
            return X, y_forecast, y_class
        else:
            return X, y_forecast, None
    
    def fit_transform(self, df):
        """
        Fit scaler and transform data
        
        Args:
            df: DataFrame with columns ['cpu_usage', 'memory_usage', 'disk_usage', 'label']
        
        Returns:
            X_train, y_forecast_train, y_class_train
        """
        # Extract features
        features = df[['cpu_usage', 'memory_usage', 'disk_usage']].values
        
        # Fit and transform
        features_scaled = self.scaler.fit_transform(features)
        self.is_fitted = True
        
        # Extract labels if available
        labels = df['label'].values if 'label' in df.columns else None
        
        # Create sequences
        X, y_forecast, y_class = self.create_sequences(features_scaled, labels)
        
        print(f"✅ Preprocessing complete:")
        print(f"   Input shape: {X.shape}")
        print(f"   Forecast target shape: {y_forecast.shape}")
        if y_class is not None:
            print(f"   Classification target shape: {y_class.shape}")
        
        return X, y_forecast, y_class
    
    def transform(self, df):
        """
        Transform data using fitted scaler (for test data or real-time prediction)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        # Extract features
        features = df[['cpu_usage', 'memory_usage', 'disk_usage']].values
        
        # Transform
        features_scaled = self.scaler.transform(features)
        
        # Extract labels if available
        labels = df['label'].values if 'label' in df.columns else None
        
        # Create sequences
        X, y_forecast, y_class = self.create_sequences(features_scaled, labels)
        
        return X, y_forecast, y_class
    
    def inverse_transform(self, data):
        """
        Convert normalized data back to original scale
        """
        # Reshape if needed
        original_shape = data.shape
        if len(original_shape) == 3:
            # (batch, time, features) -> (batch*time, features)
            data_reshaped = data.reshape(-1, original_shape[-1])
            inversed = self.scaler.inverse_transform(data_reshaped)
            # Reshape back
            inversed = inversed.reshape(original_shape)
        else:
            inversed = self.scaler.inverse_transform(data)
        
        return inversed
    
    def save_scaler(self, path="models/scaler.pkl"):
        """Save fitted scaler to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved to {path}")
    
    def load_scaler(self, path="models/scaler.pkl"):
        """Load fitted scaler from disk"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"✅ Scaler loaded from {path}")


def split_train_test(X, y_forecast, y_class, test_size=0.2):
    """
    Split data into train and test sets
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    
    y_forecast_train = y_forecast[:split_idx]
    y_forecast_test = y_forecast[split_idx:]
    
    if y_class is not None:
        y_class_train = y_class[:split_idx]
        y_class_test = y_class[split_idx:]
    else:
        y_class_train = None
        y_class_test = None
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Test:  {len(X_test)} sequences")
    
    return X_train, X_test, y_forecast_train, y_forecast_test, y_class_train, y_class_test


if __name__ == "__main__":
    # Test preprocessing
    from ai_model.data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_from_csv()
    
    preprocessor = TimeSeriesPreprocessor(window_size=12, prediction_horizon=5)
    X, y_forecast, y_class = preprocessor.fit_transform(df)
    
    print(f"\nSample input sequence shape: {X[0].shape}")
    print(f"Sample forecast target shape: {y_forecast[0].shape}")
    print(f"Sample classification label: {y_class[0]}")