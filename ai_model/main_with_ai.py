"""
Integrated Main Script with AI-Powered Anomaly Detection
Combines logging + real-time prediction + multi-level alerting
"""

import time
import sys
import os
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# Add Project Root (AIOPS) to Python Path
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

# --------------------------------------------------
# Imports from Log Generator Module
# (Make sure folder is renamed to: log_generator)
# --------------------------------------------------

from log_generator.src.config import LOG_INTERVAL_SECONDS, DATABASE_URL
from log_generator.src.logger.log_generator import generate_log
from log_generator.src.logger.db_writer import write_to_db
from log_generator.src.logger.db_init import init_db

# --------------------------------------------------
# Imports from AI Model Module
# --------------------------------------------------

try:
    from ai_model.predict import RealtimePredictor
    from ai_model.alert_engine import AlertEngine, AlertLevel

    AI_AVAILABLE = True

except ImportError as e:

    print(f"⚠️  AI model not available: {e}")
    print("   Running in logging-only mode\n")

    AI_AVAILABLE = False


# ==================================================
# AI Monitoring System
# ==================================================

class AIMonitoringSystem:

    def __init__(self, use_ai=True, prediction_interval=60):
        """
        Initialize AI-powered monitoring system

        Args:
            use_ai: Enable AI prediction and alerting
            prediction_interval: Make predictions every N seconds
        """

        self.use_ai = use_ai and AI_AVAILABLE
        self.prediction_interval = prediction_interval

        self.last_prediction_time = 0

        self.log_buffer = []
        self.max_buffer_size = 12

        if self.use_ai:

            try:

                print("🤖 Initializing AI prediction engine...")

                self.predictor = RealtimePredictor(
                    model_path="models/hybrid_model.h5",
                    scaler_path="models/scaler.pkl"
                )

                self.alert_engine = AlertEngine(
                    warning_threshold=75,
                    critical_threshold=85,
                    anomaly_threshold=95,
                    forecast_warning_threshold=80,
                    forecast_critical_threshold=90
                )

                print("✅ AI engine initialized successfully!\n")

            except Exception as e:

                print(f"⚠️  Warning: Could not initialize AI engine: {e}")
                print("   Continuing without AI\n")

                self.use_ai = False


    # --------------------------------------------------
    # Buffer Handling
    # --------------------------------------------------

    def add_to_buffer(self, log_data):

        self.log_buffer.append(log_data)

        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer.pop(0)


    # --------------------------------------------------
    # AI Prediction
    # --------------------------------------------------

    def make_prediction(self):

        if len(self.log_buffer) < self.max_buffer_size:

            print(f"⏳ Collecting data... ({len(self.log_buffer)}/{self.max_buffer_size})")
            return None

        try:

            # Convert to DataFrame
            df = pd.DataFrame(self.log_buffer)

            # Rename columns
            df = df.rename(columns={
                "CPU-Usage-Percentage": "cpu_usage",
                "Memory-Usage-Percentage": "memory_usage",
                "Disk-Usage-Percentage": "disk_usage",
                "TimeStamp": "timestamp"
            })

            # Predict
            prediction = self.predictor.predict_from_dataframe(df)

            # Analyze
            alert = self.alert_engine.analyze_prediction(prediction)

            # Print alert
            self.alert_engine.print_alert(alert)

            # Critical condition
            if self.alert_engine.should_crash_system(alert):

                print("\n" + "=" * 60)
                print("☠️  ANOMALY DETECTED - SYSTEM CRASH")
                print("=" * 60)

                print("\n🛑 Simulating system shutdown...")
                print("💡 In production: trigger recovery & alerts")

                sys.exit(1)

            return alert


        except Exception as e:

            print(f"❌ Prediction Error: {e}")
            return None


    # --------------------------------------------------
    # Main Loop
    # --------------------------------------------------

    def run(self):

        print("🚀 AI-Powered Log Generator Started")
        print(f"   Logging interval: {LOG_INTERVAL_SECONDS}s")

        if self.use_ai:
            print(f"   AI interval: {self.prediction_interval}s")
        else:
            print("   AI: DISABLED")

        print()

        # Initialize DB
        init_db()

        iteration = 0

        while True:

            iteration += 1

            # Generate log
            log = generate_log()

            # Store in DB
            write_to_db(log)

            # Buffer
            if self.use_ai:
                self.add_to_buffer(log)

            # Display
            print(
                f"[{iteration}] "
                f"CPU={log['CPU-Usage-Percentage']:.1f}% | "
                f"MEM={log['Memory-Usage-Percentage']:.1f}% | "
                f"DISK={log['Disk-Usage-Percentage']:.1f}%"
            )

            # Prediction timing
            if self.use_ai:

                now = time.time()

                if now - self.last_prediction_time >= self.prediction_interval:

                    print("\n🔮 Running AI prediction...\n")

                    self.make_prediction()

                    self.last_prediction_time = now


            # Wait
            time.sleep(LOG_INTERVAL_SECONDS)


# ==================================================
# Entry Point
# ==================================================

def main():

    USE_AI = True
    PREDICTION_INTERVAL = 60

    try:

        system = AIMonitoringSystem(
            use_ai=USE_AI,
            prediction_interval=PREDICTION_INTERVAL
        )

        system.run()

    except KeyboardInterrupt:

        print("\n\n⏹️ Monitoring stopped by user")

    except Exception as e:

        print(f"\n\n❌ Fatal Error: {e}")
        raise


if __name__ == "__main__":
    main()