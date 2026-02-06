import pandas as pd
from src.config import DATA_PATH

def load_data():
    # Load dataset (CSV from Kaggle or Excel from log generator)
    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_excel(DATA_PATH, engine="openpyxl")

    # Select only the columns you generate everywhere
    df = df[
        [
            "CPU-Usage-Percentage",
            "Memory-Usage-Percentage",
            "Disk-Usage-Percentage"
        ]
    ]

    # Drop missing values
    df = df.dropna()

    return df
