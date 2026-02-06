import pandas as pd

def predict_anomalies(model, X):
    predictions = model.predict(X)
    # -1 = anomaly, 1 = normal
    return pd.Series(predictions).map({1: 0, -1: 1})
