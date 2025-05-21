"""
Tests for model forecast outputs and dimensions.
"""

import joblib
import pandas as pd

def test_arima_forecast_shape():
    model = joblib.load("models/arima_model.pkl")
    forecast = model.forecast(steps=5)
    assert len(forecast) == 5

def test_rf_forecast_shape():
    model = joblib.load("models/rf_model.pkl")
    df = pd.read_csv("data/processed/feature_data.csv", index_col=0)
    df = df.dropna()
    features = [col for col in df.columns if col != "Lifetouch Heart Rate" and not col.endswith("_log")]
    preds = model.predict(df[features])
    assert len(preds) == len(df)
