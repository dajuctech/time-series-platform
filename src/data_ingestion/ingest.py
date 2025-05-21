"""
Ingest data from a remote CSV file or local source.
Saves cleaned version to /data/processed/.
"""

import pandas as pd
import os

# Constants
DATA_URL = "https://drive.google.com/uc?id=1DcEZemAvmlBxwmc4IEIkDRkTpHLAqTBH"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

def load_data(url: str) -> pd.DataFrame:
    """Loads dataset from a given URL."""
    df = pd.read_csv(url)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and transformation."""
    df['Timestamp (GMT)'] = pd.to_datetime(df['Timestamp (GMT)'])
    df.set_index('Timestamp (GMT)', inplace=True)
    return df

def save_data(df: pd.DataFrame, path: str) -> None:
    """Saves the processed dataframe to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"✅ Data saved to {path}")

def run_ingestion():
    df = load_data(DATA_URL)
    df = preprocess_data(df)
    save_data(df, OUTPUT_PATH)

if __name__ == "__main__":
    run_ingestion()
