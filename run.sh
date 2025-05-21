#!/bin/bash

echo "🚀 Starting End-to-End Pipeline Execution"

echo "🔹 Step 1: Ingesting Data"
python src/data_ingestion/ingest.py || { echo "❌ Failed at ingest.py"; exit 1; }

echo "🔹 Step 2: Cleaning Outliers"
python src/preprocessing/outliers.py || { echo "❌ Failed at outliers.py"; exit 1; }

echo "🔹 Step 3: Feature Engineering"
python src/preprocessing/features.py || { echo "❌ Failed at features.py"; exit 1; }

echo "🔹 Step 4: Training ARIMA Model"
python src/models/train_arima.py || { echo "❌ Failed at train_arima.py"; exit 1; }

echo "🔹 Step 5: Training Random Forest Model"
python src/models/train_rf.py || { echo "❌ Failed at train_rf.py"; exit 1; }

echo "🔹 Step 6: Launching FastAPI Server"
uvicorn src.api.main:app --reload

echo "✅ Pipeline Complete"
