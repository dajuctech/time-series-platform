"""
Integration test for FastAPI endpoints using TestClient.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_arima_prediction():
    response = client.post("/predict/arima", json={"values": [60, 61, 62, 63, 64]})
    assert response.status_code == 200
    assert "forecast" in response.json()

def test_rf_prediction():
    response = client.post("/predict/rf", json={"values": list(range(20, 35))})
    assert response.status_code == 200
    assert "forecast" in response.json()
