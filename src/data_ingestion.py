import pandas as pd
from validation import validate_data



DATA_PATH = "data/raw/CIB.csv"

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

