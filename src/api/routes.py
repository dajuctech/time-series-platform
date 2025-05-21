"""
Defines API routes for model predictions.
"""

from fastapi import APIRouter
from src.api.model_loader import load_arima_model, load_rf_model, prepare_rf_input
import pandas as pd
from pydantic import BaseModel
from typing import List

router = APIRouter()

# Request model
class TimeSeriesRequest(BaseModel):
    values: List[float]  # Time series values to be used for forecasting

@router.post("/predict/arima")
def predict_arima(request: TimeSeriesRequest):
    model = load_arima_model()
    forecast = model.forecast(steps=len(request.values))
    return {"model": "ARIMA", "forecast": forecast.tolist()}

@router.post("/predict/rf")
def predict_rf(request: TimeSeriesRequest):
    model = load_rf_model()
    df = prepare_rf_input(request.values)
    prediction = model.predict(df)
    return {"model": "Random Forest", "forecast": prediction.tolist()}
