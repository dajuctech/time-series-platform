"""
Feature engineering for time series forecasting.
Includes log transformation, differencing, rolling means, and lag features.
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/feature_data.csv"

def add_rolling_features(df: pd.DataFrame, target_col: str, windows=[3, 6, 12]) -> pd.DataFrame:
    """Adds rolling mean features."""
    for window in windows:
        df[f"{target_col}_rollmean_{window}"] = df[target_col].rolling(window=window).mean()
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, lags=[1, 2, 3]) -> pd.DataFrame:
    """Adds lag features to capture time dependencies."""
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

def log_transform(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Applies log transform to reduce variance."""
    df[f"{target_col}_log"] = np.log1p(df[target_col])
    return df

def prepare_features():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = add_rolling_features(df, "Lifetouch Heart Rate")
    df = add_lag_features(df, "Lifetouch Heart Rate")
    df = log_transform(df, "Lifetouch Heart Rate")
    
    df.dropna(inplace=True)  # Drop rows with NaNs due to rolling and lag
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"✅ Feature-enhanced data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_features()
