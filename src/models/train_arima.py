"""
Train an ARIMA model on the Lifetouch Heart Rate.
Saves the model to disk.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

DATA_PATH = "data/processed/feature_data.csv"
MODEL_PATH = "models/arima_model.pkl"

def train_arima_model():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    series = df["Lifetouch Heart Rate"].dropna()

    # Define ARIMA model order (from analysis or auto_arima)
    model = ARIMA(series, order=(1, 1, 2))
    model_fit = model.fit()

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_fit, MODEL_PATH)
    print(f"✅ ARIMA model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_arima_model()
