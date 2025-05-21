"""
Validates schema, missing values, and types for the ingested data.
"""

import pandas as pd

INPUT_PATH = "data/processed/cleaned_data.csv"

REQUIRED_COLUMNS = [
    "Lifetouch Heart Rate", "Lifetouch Respiration Rate",
    "Oximeter SpO2", "Oximeter Pulse"
]

def load_processed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")
    print("✅ All required columns are present.")

def check_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    print(f"📋 Missing values:\n{missing[missing > 0]}")

def validate_data_types(df: pd.DataFrame) -> None:
    print("📋 Data types:")
    print(df.dtypes)

def run_validation():
    df = load_processed_data(INPUT_PATH)
    validate_columns(df)
    check_missing_values(df)
    validate_data_types(df)

if __name__ == "__main__":
    run_validation()
