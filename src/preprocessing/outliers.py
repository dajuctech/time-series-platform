"""
Detects and handles outliers using the IQR method.
"""

import pandas as pd
import os

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/cleaned_no_outliers.csv"

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Removes outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"✅ Removed outliers in {column}: {df.shape[0] - cleaned_df.shape[0]} rows dropped.")
    return cleaned_df

def clean_outliers():
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = remove_outliers_iqr(df, "Lifetouch Heart Rate")
    df = remove_outliers_iqr(df, "Lifetouch Respiration Rate")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"✅ Outlier-free data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_outliers()
