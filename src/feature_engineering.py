import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/CIB_clean.csv"
OUTPUT_PATH = "data/processed/CIB_features.csv"


def create_features(df):

    df = df.copy()

    # Make sure data is sorted by date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Rename columns
    df = df.rename(columns={
        "Price": "Close",
        "Vol.": "Volume"
    })

    # =========================================================
    # 1. Returns
    # =========================================================

    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # =========================================================
    # 2. Moving Averages
    # =========================================================

    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    df["EMA_10"] = df["Close"].ewm(
        span=10,
        adjust=False
    ).mean()

    df["EMA_20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # =========================================================
    # 3. RSI
    # =========================================================

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["RSI_14"] = 100 - (100 / (1 + rs))

    # =========================================================
    # 4. MACD
    # =========================================================

    ema_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Hist"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # =========================================================
    # 5. Bollinger Bands
    # =========================================================

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()

    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)

    df["BB_Width"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        / rolling_mean
    )

    # =========================================================
    # 6. Volatility
    # =========================================================

    df["Volatility_10"] = (
        df["Return_1d"].rolling(10).std()
    )

    df["Volatility_20"] = (
        df["Return_1d"].rolling(20).std()
    )

    # =========================================================
    # 7. Price Range
    # =========================================================

    df["High_Low_Range"] = (
        (df["High"] - df["Low"])
        / df["Close"]
    )

    df["Open_Close_Range"] = (
        (df["Close"] - df["Open"])
        / df["Open"]
    )

    # =========================================================
    # 8. Volume Features
    # =========================================================

    df["Volume_Change"] = df["Volume"].pct_change()

    df["Volume_SMA_10"] = (
        df["Volume"].rolling(10).mean()
    )

    # =========================================================
    # 9. Lag Features
    # =========================================================

    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    # =========================================================
    # 10. Target
    # Tomorrow's closing price
    # =========================================================

    df["Target"] = df["Close"].shift(-1)

    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT_PATH)

    df = create_features(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Feature engineering completed.")
    print("Shape:", df.shape)
    print("Saved to:", OUTPUT_PATH)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())