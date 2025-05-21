"""
Handles model loading and feature preprocessing for API use.
"""

import joblib
import pandas as pd
import numpy as np
import os

ARIMA_MODEL_PATH = "models/arima_model.pkl"
RF_MODEL_PATH = "models/rf_model.pkl"

def load_arima_model():
    return joblib.load(ARIMA_MODEL_PATH)

def load_rf_model():
    return joblib.load(RF_MODEL_PATH)

def prepare_rf_input(input_values: list) -> pd.DataFrame:
    """
    Simulate lag/rolling features for Random Forest input.
    """
    df = pd.DataFrame({"Lifetouch Heart Rate": input_values})
    df["HR_lag_1"] = df["Lifetouch Heart Rate"].shift(1)
    df["HR_lag_2"] = df["Lifetouch Heart Rate"].shift(2)
    df["HR_lag_3"] = df["Lifetouch Heart Rate"].shift(3)
    df["HR_roll_3"] = df["Lifetouch Heart Rate"].rolling(window=3).mean()
    df["HR_roll_6"] = df["Lifetouch Heart Rate"].rolling(window=6).mean()
    df["HR_roll_12"] = df["Lifetouch Heart Rate"].rolling(window=12).mean()
    df.dropna(inplace=True)
    return df
