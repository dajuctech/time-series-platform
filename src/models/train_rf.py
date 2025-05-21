"""
Train a Random Forest model using lag/rolling features.
Saves the model to disk.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "data/processed/feature_data.csv"
MODEL_PATH = "models/rf_model.pkl"

def train_rf_model():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    target = "Lifetouch Heart Rate"
    features = [col for col in df.columns if col != target and not col.endswith("_log")]

    X = df[features].dropna()
    y = df.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Random Forest model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_rf_model()
