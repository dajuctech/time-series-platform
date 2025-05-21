"""
Custom metrics for evaluating regression models.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (ignores zero targets).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        return np.nan

def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Prints common regression metrics with labels.
    """
    print(f"\n📊 {model_name} Evaluation")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
