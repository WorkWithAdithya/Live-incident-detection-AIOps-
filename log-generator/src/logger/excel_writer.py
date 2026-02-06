import pandas as pd
import os

def write_to_excel(log_data, file_path):
    df_new = pd.DataFrame([log_data])

    # If file does not exist, create it
    if not os.path.exists(file_path):
        df_new.to_excel(file_path, index=False, engine="openpyxl")
        return

    # If file exists, safely append
    try:
        df_existing = pd.read_excel(file_path, engine="openpyxl")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file_path, index=False, engine="openpyxl")
    except Exception:
        # If file exists but is corrupted, recreate it
        df_new.to_excel(file_path, index=False, engine="openpyxl")
