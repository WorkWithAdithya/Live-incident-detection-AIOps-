"""
CNN-LSTM Hybrid Model for Anomaly Detection
- CNN extracts features from time-series patterns
- LSTM learns temporal dependencies
- Dual output: Forecasting + Classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os


class AnomalyDetectionModel:
    def __init__(self, window_size=12, n_features=3, prediction_horizon=5, n_classes=4):
        """
        Args:
            window_size: Number of input time steps (default: 12)
            n_features: Number of features (cpu, memory, disk = 3)
            prediction_horizon: Number of future time steps to predict (default: 5)
            n_classes: Number of classification classes (Normal, Warning, Critical, Anomaly = 4)
        """
        self.window_size = window_size
        self.n_features = n_features
        self.prediction_horizon = prediction_horizon
        self.n_classes = n_classes
        self.model = None
    
    def build_cnn_lstm_forecaster(self):
        """
        Build CNN-LSTM model for time-series forecasting
        Predicts next prediction_horizon time steps
        """
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # CNN layers for feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # LSTM layers for temporal dependencies
        x = layers.LSTM(50, return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(25)(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers for forecasting
        x = layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer: Predict next prediction_horizon values for all features
        outputs = layers.Dense(self.prediction_horizon * self.n_features)(x)
        outputs = layers.Reshape((self.prediction_horizon, self.n_features))(outputs)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Forecaster')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_cnn_classifier(self):
        """
        Build CNN model for anomaly classification
        Classifies: Normal, Warning, Critical, Anomaly
        """
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # CNN layers
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer: Multi-class classification
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Classifier')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_hybrid_model(self):
        """
        Build hybrid model with both forecasting and classification
        """
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # Shared CNN feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # LSTM for temporal learning
        lstm_out = layers.LSTM(50, return_sequences=True)(x)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        lstm_out = layers.LSTM(25)(lstm_out)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        # Branch 1: Forecasting
        forecast_branch = layers.Dense(50, activation='relu')(lstm_out)
        forecast_branch = layers.Dropout(0.2)(forecast_branch)
        forecast_output = layers.Dense(self.prediction_horizon * self.n_features)(forecast_branch)
        forecast_output = layers.Reshape((self.prediction_horizon, self.n_features), name='forecast')(forecast_output)
        
        # Branch 2: Classification
        class_branch = layers.Dense(64, activation='relu')(lstm_out)
        class_branch = layers.Dropout(0.3)(class_branch)
        class_output = layers.Dense(self.n_classes, activation='softmax', name='classification')(class_branch)
        
        model = models.Model(
            inputs=inputs,
            outputs=[forecast_output, class_output],
            name='Hybrid_CNN_LSTM'
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'forecast': 'mse',
                'classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'forecast': 1.0,
                'classification': 0.5
            },
            metrics={
                'forecast': ['mae'],
                'classification': ['accuracy']
            }
        )
        
        return model
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        """Get training callbacks"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        return callbacks
    
    def print_model_summary(self, model):
        """Print model architecture"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        model.summary()
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test model building
    print("🏗️  Building models...\n")
    
    model_builder = AnomalyDetectionModel(
        window_size=12,
        n_features=3,
        prediction_horizon=5,
        n_classes=4
    )
    
    # Test forecaster
    print("1️⃣  CNN-LSTM Forecaster:")
    forecaster = model_builder.build_cnn_lstm_forecaster()
    model_builder.print_model_summary(forecaster)
    
    # Test classifier
    print("\n2️⃣  CNN Classifier:")
    classifier = model_builder.build_cnn_classifier()
    model_builder.print_model_summary(classifier)
    
    # Test hybrid
    print("\n3️⃣  Hybrid Model:")
    hybrid = model_builder.build_hybrid_model()
    model_builder.print_model_summary(hybrid)