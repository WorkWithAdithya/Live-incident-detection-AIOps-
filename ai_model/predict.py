"""
Real-time Prediction Engine
Loads trained model and makes predictions on live data
"""

import sys
import os
import numpy as np
import pandas as pd
from tensorflow import keras

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.preprocessing import TimeSeriesPreprocessor
from ai_model.data_loader import DataLoader


class RealtimePredictor:
    def __init__(self, model_path='models/hybrid_model.h5', scaler_path='models/scaler.pkl'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to fitted scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.preprocessor = None
        self.label_names = {
            0: 'Normal',
            1: 'Warning',
            2: 'Critical',
            3: 'Anomaly'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        print(f"📦 Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        
        print(f"📦 Loading scaler from {self.scaler_path}...")
        self.preprocessor = TimeSeriesPreprocessor(window_size=12, prediction_horizon=5)
        self.preprocessor.load_scaler(self.scaler_path)
        
        print("✅ Model and scaler loaded successfully!\n")
    
    def predict_from_dataframe(self, df):
        """
        Make prediction from DataFrame
        
        Args:
            df: DataFrame with at least 12 rows and columns ['cpu_usage', 'memory_usage', 'disk_usage']
        
        Returns:
            dict with predictions
        """
        if len(df) < 12:
            raise ValueError(f"Need at least 12 data points, got {len(df)}")
        
        # Take last 12 rows
        df_recent = df.tail(12).copy()
        
        # Ensure we have the required columns
        if not all(col in df_recent.columns for col in ['cpu_usage', 'memory_usage', 'disk_usage']):
            raise ValueError("DataFrame must have columns: cpu_usage, memory_usage, disk_usage")
        
        # Normalize
        features = df_recent[['cpu_usage', 'memory_usage', 'disk_usage']].values
        features_scaled = self.preprocessor.scaler.transform(features)
        
        # Reshape for model input: (1, 12, 3)
        X = features_scaled.reshape(1, 12, 3)
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Parse predictions based on model type
        if isinstance(predictions, list):
            # Hybrid model: [forecast, classification]
            forecast_scaled = predictions[0][0]  # Shape: (5, 3)
            class_probs = predictions[1][0]  # Shape: (4,)
            
            # Denormalize forecast
            forecast = self.preprocessor.inverse_transform(forecast_scaled.reshape(1, 5, 3))[0]
            
            # Get predicted class
            predicted_class = int(np.argmax(class_probs))
            confidence = float(class_probs[predicted_class])
            
            result = {
                'forecast': {
                    'cpu': forecast[:, 0].tolist(),
                    'memory': forecast[:, 1].tolist(),
                    'disk': forecast[:, 2].tolist()
                },
                'classification': {
                    'predicted_class': predicted_class,
                    'class_name': self.label_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': {
                        'Normal': float(class_probs[0]),
                        'Warning': float(class_probs[1]),
                        'Critical': float(class_probs[2]),
                        'Anomaly': float(class_probs[3])
                    }
                },
                'current_values': {
                    'cpu': float(df_recent['cpu_usage'].iloc[-1]),
                    'memory': float(df_recent['memory_usage'].iloc[-1]),
                    'disk': float(df_recent['disk_usage'].iloc[-1])
                }
            }
        else:
            # Single output model
            if predictions.shape[-1] == 3:
                # Forecaster only
                forecast_scaled = predictions[0]
                forecast = self.preprocessor.inverse_transform(forecast_scaled.reshape(1, 5, 3))[0]
                
                result = {
                    'forecast': {
                        'cpu': forecast[:, 0].tolist(),
                        'memory': forecast[:, 1].tolist(),
                        'disk': forecast[:, 2].tolist()
                    },
                    'current_values': {
                        'cpu': float(df_recent['cpu_usage'].iloc[-1]),
                        'memory': float(df_recent['memory_usage'].iloc[-1]),
                        'disk': float(df_recent['disk_usage'].iloc[-1])
                    }
                }
            else:
                # Classifier only
                class_probs = predictions[0]
                predicted_class = int(np.argmax(class_probs))
                confidence = float(class_probs[predicted_class])
                
                result = {
                    'classification': {
                        'predicted_class': predicted_class,
                        'class_name': self.label_names[predicted_class],
                        'confidence': confidence,
                        'probabilities': {
                            'Normal': float(class_probs[0]),
                            'Warning': float(class_probs[1]),
                            'Critical': float(class_probs[2]),
                            'Anomaly': float(class_probs[3])
                        }
                    },
                    'current_values': {
                        'cpu': float(df_recent['cpu_usage'].iloc[-1]),
                        'memory': float(df_recent['memory_usage'].iloc[-1]),
                        'disk': float(df_recent['disk_usage'].iloc[-1])
                    }
                }
        
        return result
    
    def predict_from_database(self, database_url, last_n=12):
        """
        Load recent data from database and make prediction
        
        Args:
            database_url: PostgreSQL connection string
            last_n: Number of recent records to load (default: 12)
        """
        loader = DataLoader(database_url=database_url)
        df = loader.load_from_database(limit=last_n)
        
        if df is None or len(df) < 12:
            raise ValueError(f"Not enough data in database. Need at least 12 records, got {len(df) if df is not None else 0}")
        
        return self.predict_from_dataframe(df)
    
    def print_prediction(self, result):
        """Pretty print prediction results"""
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        # Current values
        print("\n📊 Current System Status:")
        print(f"   CPU:    {result['current_values']['cpu']:.2f}%")
        print(f"   Memory: {result['current_values']['memory']:.2f}%")
        print(f"   Disk:   {result['current_values']['disk']:.2f}%")
        
        # Forecast
        if 'forecast' in result:
            print("\n🔮 Predicted Next 25 Seconds (5 steps @ 5s intervals):")
            forecast = result['forecast']
            for i in range(5):
                print(f"   Step {i+1} (+{(i+1)*5}s): CPU={forecast['cpu'][i]:.1f}%  "
                      f"Mem={forecast['memory'][i]:.1f}%  Disk={forecast['disk'][i]:.1f}%")
        
        # Classification
        if 'classification' in result:
            class_info = result['classification']
            class_name = class_info['class_name']
            confidence = class_info['confidence']
            
            # Color coding based on severity
            icons = {
                'Normal': '✅',
                'Warning': '⚠️ ',
                'Critical': '🔴',
                'Anomaly': '☠️ '
            }
            
            print(f"\n{icons[class_name]} Classification: {class_name}")
            print(f"   Confidence: {confidence*100:.1f}%")
            
            print("\n   Probability Distribution:")
            for label, prob in class_info['probabilities'].items():
                bar = '█' * int(prob * 20)
                print(f"   {label:10} {prob*100:5.1f}% {bar}")
        
        print("\n" + "="*60 + "\n")


def test_prediction():
    """Test prediction with sample data"""
    print("🧪 Testing Real-time Predictor...\n")
    
    # Create sample data (last 12 readings)
    sample_data = {
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=12, freq='5S'),
        'cpu_usage': [35, 37, 40, 42, 45, 48, 52, 56, 62, 68, 75, 82],
        'memory_usage': [40, 42, 44, 46, 49, 52, 55, 59, 64, 69, 75, 80],
        'disk_usage': [50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55]
    }
    df = pd.DataFrame(sample_data)
    
    # Initialize predictor
    predictor = RealtimePredictor(
        model_path='models/hybrid_model.h5',
        scaler_path='models/scaler.pkl'
    )
    
    # Make prediction
    result = predictor.predict_from_dataframe(df)
    predictor.print_prediction(result)
    
    return result


if __name__ == "__main__":
    test_prediction()