"""
Streamlit dashboard for visualizing heart rate time series and model forecasts.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from src.utils.metrics import rmse, mape

# Load data and models
DATA_PATH = "data/processed/feature_data.csv"
ARIMA_MODEL_PATH = "models/arima_model.pkl"
RF_MODEL_PATH = "models/rf_model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df

@st.cache_resource
def load_models():
    arima = joblib.load(ARIMA_MODEL_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    return arima, rf

def predict_rf(model, df):
    features = [col for col in df.columns if col != "Lifetouch Heart Rate" and not col.endswith("_log")]
    df_clean = df.dropna(subset=features)
    X = df_clean[features]
    y_true = df_clean["Lifetouch Heart Rate"]
    y_pred = model.predict(X)
    return y_true, y_pred

def main():
    st.title("📊 Heart Rate Forecasting Dashboard")

    df = load_data()
    arima_model, rf_model = load_models()

    st.subheader("Raw Heart Rate Time Series")
    st.line_chart(df["Lifetouch Heart Rate"])

    st.subheader("Select Forecasting Model")
    model_choice = st.selectbox("Choose a model", ["ARIMA", "Random Forest"])

    if model_choice == "ARIMA":
        st.write("🔮 Forecasting using ARIMA model...")
        forecast = arima_model.forecast(steps=24)
        st.line_chart(forecast)
        st.success("Forecast complete.")
    else:
        st.write("🔮 Forecasting using Random Forest model...")
        y_true, y_pred = predict_rf(rf_model, df)
        results = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}, index=y_true.index)
        st.line_chart(results)
        
        st.subheader("📈 Model Evaluation")
        st.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
        st.metric("RMSE", f"{rmse(y_true, y_pred):.2f}")
        st.metric("MAPE", f"{mape(y_true, y_pred):.2f}%")

    st.caption("Developed by Daniel Jude • Time Series Forecasting Project")

if __name__ == "__main__":
    main()
