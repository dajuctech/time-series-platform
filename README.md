# 🧠 Time Series Intelligence Platform

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/github/license/dajuctech/time-series-platform)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A full-stack, end-to-end platform for **heart rate time series forecasting** using ARIMA and Random Forest. Built with a modern tech stack including **FastAPI**, **Streamlit**, **Docker**, and **Kubernetes** for deployment — and powered by **clean architecture**, **automated testing**, and **interactive visualization**.

---

## 🚀 Features

- 📥 Data ingestion from local or cloud
- 🧼 Schema validation & IQR-based outlier cleaning
- 🔍 Exploratory Data Analysis (EDA)
- 🛠 Feature Engineering (lags, rolling means, log transform)
- 🤖 ARIMA and Random Forest training
- 🌐 RESTful API with FastAPI
- 📊 Streamlit dashboard for predictions
- ☁️ Docker & Kubernetes deployment-ready
- ✅ Fully tested with `pytest` for CI/CD

---

## 📂 Project Structure

```
time-series-platform/
├── data/                # Raw and processed CSVs
├── notebooks/           # Jupyter notebook for manual EDA
├── src/
│   ├── api/             # FastAPI routes and model loading
│   ├── dashboard/       # Streamlit UI app
│   ├── data_ingestion/  # Data loading and schema checks
│   ├── models/          # ARIMA & RF training + forecast
│   ├── preprocessing/   # Outlier removal, rolling features
│   ├── utils/           # Logger and metrics utils
│   └── cloud/           # Dockerfile and K8s configs
├── tests/               # Unit tests with pytest
├── run.sh               # One-click execution script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── README.md            # You're reading it!
```

---

## ⚙️ Installation

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ (Optional) Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate        # Windows
```

---

## ▶️ Usage

### 🧩 Run the full pipeline
```bash
chmod +x run.sh
./run.sh
```

### 🌐 Launch the API
```bash
uvicorn src.api.main:app --reload
# Open http://127.0.0.1:8000/docs
```

### 📊 Launch the dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📤 Cloud Deployment

### 🐳 Docker
```bash
docker build -t time-series-api -f src/cloud/docker/Dockerfile .
docker run -p 8000:8000 time-series-api
```

### ☸️ Kubernetes
```bash
kubectl apply -f src/cloud/k8s/deployment.yaml
```

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 📈 Model Evaluation

| Model   | MSE   | MAE  | RMSE |
|---------|-------|------|------|
| ARIMA   | 11.6  | 2.39 | 3.41 |
| SARIMA  | 12.21 | 2.70 | 3.49 |
| SARIMAX | 12.28 | 2.71 | 3.50 |

✅ **SARIMA is the best choice** for seasonal and trend stability.

---

## 👨‍💻 Author

**Daniel Jude**  
🔗 GitHub: [@dajuctech](https://github.com/dajuctech)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full text.
