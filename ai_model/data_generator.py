"""
Synthetic Data Generator for System Logs
Generates realistic CPU, Memory, and Disk usage patterns
(No labels, No sequence IDs)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class SyntheticDataGenerator:

    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        np.random.seed(42)


    # --------------------------------------------------
    # Pattern Generators
    # --------------------------------------------------

    def generate_normal_pattern(self, base=30, variance=5, length=100):

        return base + np.random.normal(0, variance, length)


    def generate_gradual_increase(self, start=30, end=85, length=100):

        trend = np.linspace(start, end, length)
        noise = np.random.normal(0, 2, length)

        return trend + noise


    def generate_spike_pattern(self, base=40, spike_height=95, length=100):

        pattern = np.ones(length) * base

        spike_start = int(length * 0.7)
        spike_duration = int(length * 0.2)

        pattern[spike_start:spike_start + spike_duration] = spike_height

        pattern[spike_start-5:spike_start] = np.linspace(base, spike_height, 5)
        pattern[spike_start+spike_duration:spike_start+spike_duration+5] = np.linspace(
            spike_height, base, 5
        )

        noise = np.random.normal(0, 3, length)

        return pattern + noise


    def generate_anomaly_pattern(self, length=100):

        pattern = np.random.uniform(80, 99, length)

        spikes = np.random.choice([0, 1], length, p=[0.7, 0.3])

        pattern = pattern + spikes * np.random.uniform(0, 10, length)

        return np.clip(pattern, 0, 100)


    def generate_periodic_pattern(self, base=35, amplitude=20, length=100):

        x = np.linspace(0, 4*np.pi, length)

        pattern = base + amplitude * np.sin(x)

        noise = np.random.normal(0, 3, length)

        return pattern + noise


    def generate_memory_leak(self, start=30, length=100):

        growth_rate = np.random.uniform(0.3, 0.8)

        pattern = start + growth_rate * np.arange(length)

        noise = np.random.normal(0, 1, length)

        return np.clip(pattern + noise, 0, 100)


    # --------------------------------------------------
    # Dataset Generator (NO LABELS)
    # --------------------------------------------------

    def generate_dataset(self):

        data = []

        patterns_per_type = self.num_samples // 6

        print("🎲 Generating synthetic dataset...")


        # 1. Normal
        print("  → Normal patterns...")
        for _ in range(patterns_per_type * 2):

            cpu = self.generate_normal_pattern(base=np.random.uniform(20, 40))
            memory = self.generate_normal_pattern(base=np.random.uniform(30, 50))
            disk = self.generate_normal_pattern(base=np.random.uniform(40, 60))

            data.append((cpu, memory, disk))


        # 2. Gradual Increase
        print("  → Gradual increase patterns...")
        for _ in range(patterns_per_type):

            cpu = self.generate_gradual_increase()
            memory = self.generate_gradual_increase()
            disk = self.generate_normal_pattern(base=np.random.uniform(45, 60))

            data.append((cpu, memory, disk))


        # 3. Spikes
        print("  → Spike patterns...")
        for _ in range(patterns_per_type):

            cpu = self.generate_spike_pattern()
            memory = self.generate_spike_pattern()
            disk = self.generate_normal_pattern(base=np.random.uniform(45, 60))

            data.append((cpu, memory, disk))


        # 4. Anomalies
        print("  → Anomaly patterns...")
        for _ in range(patterns_per_type // 2):

            cpu = self.generate_anomaly_pattern()
            memory = self.generate_anomaly_pattern()
            disk = self.generate_anomaly_pattern()

            data.append((cpu, memory, disk))


        # 5. Periodic
        print("  → Periodic patterns...")
        for _ in range(patterns_per_type // 2):

            cpu = self.generate_periodic_pattern()
            memory = self.generate_periodic_pattern()
            disk = self.generate_normal_pattern(base=np.random.uniform(45, 60))

            data.append((cpu, memory, disk))


        # 6. Memory Leak
        print("  → Memory leak patterns...")
        for _ in range(patterns_per_type // 2):

            cpu = self.generate_normal_pattern()
            memory = self.generate_memory_leak()
            disk = self.generate_normal_pattern(base=np.random.uniform(45, 60))

            data.append((cpu, memory, disk))


        print(f"✅ Generated {len(data)} sequences")

        return data


    # --------------------------------------------------
    # Save to CSV (NO LABEL, NO SEQUENCE ID)
    # --------------------------------------------------

    def save_to_csv(self, data, output_path="data/synthetic_data.csv"):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []

        base_time = datetime.now() - timedelta(days=30)


        for cpu, memory, disk in data:

            for t in range(len(cpu)):

                timestamp = base_time + timedelta(seconds=t * 5)

                rows.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "cpu_usage": round(float(np.clip(cpu[t], 0, 100)), 2),
                    "memory_usage": round(float(np.clip(memory[t], 0, 100)), 2),
                    "disk_usage": round(float(np.clip(disk[t], 0, 100)), 2)
                })


        df = pd.DataFrame(rows)

        df.to_csv(output_path, index=False)

        print(f"✅ Saved {len(rows)} rows to {output_path}")

        return df


# ==================================================
# Main
# ==================================================

def main():

    generator = SyntheticDataGenerator(num_samples=10000)

    data = generator.generate_dataset()

    df = generator.save_to_csv(data)

    print("\n📊 Sample data:")
    print(df.head(10))


if __name__ == "__main__":
    main()