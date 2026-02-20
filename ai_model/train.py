"""
Training Script for CNN-LSTM Anomaly Detection Model
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.data_generator import SyntheticDataGenerator
from ai_model.data_loader import DataLoader
from ai_model.preprocessing import TimeSeriesPreprocessor, split_train_test
from ai_model.cnn_lstm_model import AnomalyDetectionModel


class ModelTrainer:
    def __init__(self, use_synthetic=True, model_type='hybrid'):
        """
        Args:
            use_synthetic: If True, generate synthetic data. If False, load from database.
            model_type: 'forecaster', 'classifier', or 'hybrid'
        """
        self.use_synthetic = use_synthetic
        self.model_type = model_type
        self.model = None
        self.history = None
    
    def prepare_data(self):
        """Load and preprocess data"""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPARATION")
        print("="*60)
        
        if self.use_synthetic:
            # Generate synthetic data
            print("\n🎲 Generating synthetic dataset...")
            generator = SyntheticDataGenerator(num_samples=5000)
            data = generator.generate_dataset()
            df = generator.save_to_csv(data, output_path="data/synthetic_data.csv")
        else:
            # Load from database
            print("\n🗄️  Loading data from database...")
            from config import DATABASE_URL
            loader = DataLoader(database_url=DATABASE_URL)
            df = loader.load_from_database()
            
            if df is None or len(df) < 1000:
                print("⚠️  Not enough data in database. Using synthetic data instead.")
                self.use_synthetic = True
                return self.prepare_data()
        
        # Preprocessing
        print("\n🔧 Preprocessing data...")
        self.preprocessor = TimeSeriesPreprocessor(window_size=12, prediction_horizon=5)
        X, y_forecast, y_class = self.preprocessor.fit_transform(df)
        
        # Save scaler
        self.preprocessor.save_scaler("models/scaler.pkl")
        
        # Split train/test
        X_train, X_test, y_f_train, y_f_test, y_c_train, y_c_test = split_train_test(
            X, y_forecast, y_class, test_size=0.2
        )
        
        return X_train, X_test, y_f_train, y_f_test, y_c_train, y_c_test
    
    def build_model(self):
        """Build the model"""
        print("\n" + "="*60)
        print("STEP 2: MODEL BUILDING")
        print("="*60)
        
        model_builder = AnomalyDetectionModel(
            window_size=12,
            n_features=3,
            prediction_horizon=5,
            n_classes=4
        )
        
        if self.model_type == 'forecaster':
            print("\n🏗️  Building CNN-LSTM Forecaster...")
            self.model = model_builder.build_cnn_lstm_forecaster()
        elif self.model_type == 'classifier':
            print("\n🏗️  Building CNN Classifier...")
            self.model = model_builder.build_cnn_classifier()
        else:  # hybrid
            print("\n🏗️  Building Hybrid Model...")
            self.model = model_builder.build_hybrid_model()
        
        model_builder.print_model_summary(self.model)
        
        return model_builder
    
    def train_model(self, X_train, X_test, y_f_train, y_f_test, y_c_train, y_c_test, epochs=50, batch_size=32):
        """Train the model"""
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        # Get callbacks
        model_builder = AnomalyDetectionModel()
        callbacks = model_builder.get_callbacks(f'models/{self.model_type}_model.h5')
        
        # Train based on model type
        if self.model_type == 'forecaster':
            print(f"\n🚀 Training Forecaster for {epochs} epochs...")
            self.history = self.model.fit(
                X_train, y_f_train,
                validation_data=(X_test, y_f_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        elif self.model_type == 'classifier':
            print(f"\n🚀 Training Classifier for {epochs} epochs...")
            self.history = self.model.fit(
                X_train, y_c_train,
                validation_data=(X_test, y_c_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        else:  # hybrid
            print(f"\n🚀 Training Hybrid Model for {epochs} epochs...")
            self.history = self.model.fit(
                X_train,
                {'forecast': y_f_train, 'classification': y_c_train},
                validation_data=(X_test, {'forecast': y_f_test, 'classification': y_c_test}),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        print("\n✅ Training complete!")
    
    def evaluate_model(self, X_test, y_f_test, y_c_test):
        """Evaluate the trained model"""
        print("\n" + "="*60)
        print("STEP 4: MODEL EVALUATION")
        print("="*60)
        
        if self.model_type == 'forecaster':
            results = self.model.evaluate(X_test, y_f_test, verbose=0)
            print(f"\n📊 Test Loss: {results[0]:.4f}")
            print(f"📊 Test MAE: {results[1]:.4f}")
        
        elif self.model_type == 'classifier':
            results = self.model.evaluate(X_test, y_c_test, verbose=0)
            print(f"\n📊 Test Loss: {results[0]:.4f}")
            print(f"📊 Test Accuracy: {results[1]*100:.2f}%")
        
        else:  # hybrid
            results = self.model.evaluate(
                X_test,
                {'forecast': y_f_test, 'classification': y_c_test},
                verbose=0
            )
            print(f"\n📊 Total Test Loss: {results[0]:.4f}")
            print(f"📊 Forecast Loss: {results[1]:.4f}")
            print(f"📊 Classification Loss: {results[2]:.4f}")
            print(f"📊 Forecast MAE: {results[3]:.4f}")
            print(f"📊 Classification Accuracy: {results[4]*100:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        print("\n📈 Generating training plots...")
        
        os.makedirs("logs", exist_ok=True)
        
        if self.model_type == 'forecaster':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # MAE
            ax2.plot(self.history.history['mae'], label='Train MAE')
            ax2.plot(self.history.history['val_mae'], label='Val MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
        
        elif self.model_type == 'classifier':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss
            ax1.plot(self.history.history['loss'], label='Train Loss')
            ax1.plot(self.history.history['val_loss'], label='Val Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            ax2.plot(self.history.history['accuracy'], label='Train Accuracy')
            ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        else:  # hybrid
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Total Loss
            axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Forecast MAE
            axes[0, 1].plot(self.history.history['forecast_mae'], label='Train MAE')
            axes[0, 1].plot(self.history.history['val_forecast_mae'], label='Val MAE')
            axes[0, 1].set_title('Forecast MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Classification Accuracy
            axes[1, 0].plot(self.history.history['classification_accuracy'], label='Train Accuracy')
            axes[1, 0].plot(self.history.history['val_classification_accuracy'], label='Val Accuracy')
            axes[1, 0].set_title('Classification Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Classification Loss
            axes[1, 1].plot(self.history.history['classification_loss'], label='Train Loss')
            axes[1, 1].plot(self.history.history['val_classification_loss'], label='Val Loss')
            axes[1, 1].set_title('Classification Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"logs/training_history_{self.model_type}_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Training plot saved to {plot_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("CNN-LSTM ANOMALY DETECTION MODEL TRAINING")
    print("="*60)
    
    # Configuration
    USE_SYNTHETIC = True  # Set to False to use real database data
    MODEL_TYPE = 'hybrid'  # 'forecaster', 'classifier', or 'hybrid'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print(f"\n⚙️  Configuration:")
    print(f"   Data Source: {'Synthetic' if USE_SYNTHETIC else 'Database'}")
    print(f"   Model Type: {MODEL_TYPE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    # Initialize trainer
    trainer = ModelTrainer(use_synthetic=USE_SYNTHETIC, model_type=MODEL_TYPE)
    
    # Prepare data
    X_train, X_test, y_f_train, y_f_test, y_c_train, y_c_test = trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train_model(X_train, X_test, y_f_train, y_f_test, y_c_train, y_c_test, 
                       epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Evaluate model
    trainer.evaluate_model(X_test, y_f_test, y_c_test)
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Model saved: models/{MODEL_TYPE}_model.h5")
    print(f"📁 Scaler saved: models/scaler.pkl")
    print(f"📁 Training plots: logs/")


if __name__ == "__main__":
    main()