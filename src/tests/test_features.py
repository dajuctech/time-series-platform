"""
Unit tests for feature engineering: rolling means, lags, etc.
"""

import pandas as pd
import numpy as np
from src.preprocessing.features import add_rolling_features, add_lag_features, log_transform

def test_add_rolling_features():
    df = pd.DataFrame({"Lifetouch Heart Rate": [1, 2, 3, 4, 5, 6]})
    result = add_rolling_features(df.copy(), "Lifetouch Heart Rate", windows=[2])
    assert "Lifetouch Heart Rate_rollmean_2" in result.columns

def test_add_lag_features():
    df = pd.DataFrame({"Lifetouch Heart Rate": [10, 20, 30, 40]})
    result = add_lag_features(df.copy(), "Lifetouch Heart Rate", lags=[1])
    assert "Lifetouch Heart Rate_lag_1" in result.columns

def test_log_transform():
    df = pd.DataFrame({"Lifetouch Heart Rate": [1, 10, 100]})
    result = log_transform(df.copy(), "Lifetouch Heart Rate")
    assert "Lifetouch Heart Rate_log" in result.columns
