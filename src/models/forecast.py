"""
Load trained models and make predictions on test set.
Used by backend or dashboard for visualization.
"""

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

DATA_PATH = "data/processed/feature_data.csv"
ARIMA_MODEL_PATH = "models/arima_model.pkl"
RF_MODEL_PATH = "models/rf_model.pkl"

def evaluate(y_true, y_pred, model_name):
    print(f"🔍 {model_name} Evaluation")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")

def forecast_arima():
    model = joblib.load(ARIMA_MODEL_PATH)
    forecast = model.forecast(steps=24)
    print("✅ ARIMA forecast:", forecast.head())
    return forecast

def forecast_rf():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    model = joblib.load(RF_MODEL_PATH)

    features = [col for col in df.columns if col != "Lifetouch Heart Rate" and not col.endswith("_log")]
    X = df[features].dropna()
    y = df.loc[X.index, "Lifetouch Heart Rate"]

    y_pred = model.predict(X)
    evaluate(y, y_pred, "Random Forest")

    return pd.Series(y_pred, index=X.index)

if __name__ == "__main__":
    forecast_arima()
    forecast_rf()
