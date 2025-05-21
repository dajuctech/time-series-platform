"""
FastAPI entry point for serving time series forecasts.
"""

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Time Series Forecasting API",
    description="Predicts heart rate using ARIMA and Random Forest",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Time Series Forecasting API"}
