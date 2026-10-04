import os
import pandas as pd
from validation import validate_data

DATA_PATH = "data/raw/CIB.csv"
OUTPUT_PATH = "data/processed/CIB_clean.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicates:", df.duplicated().sum())

    return df


if __name__ == "__main__":
    df = load_data()

    df = validate_data(df)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nValidated data saved to:", OUTPUT_PATH)