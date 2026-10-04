import pandas as pd


def validate_data(df):

    # Date
    df["Date"] = pd.to_datetime(df["Date"])

    # Volume
    df["Vol."] = (
        df["Vol."]
        .str.replace("K", "e3")
        .str.replace("M", "e6")
        .str.replace("B", "e9")
        .astype(float)
    )

    # Change %
    df["Change %"] = (
        df["Change %"]
        .str.replace("%", "")
        .astype(float)
    )

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # Validation
    assert df["Date"].is_unique, "Duplicate dates found"

    assert (
        df["High"] >= df[["Open", "Price"]].max(axis=1)
    ).all(), "Invalid High values"

    assert (
        df["Low"] <= df[["Open", "Price"]].min(axis=1)
    ).all(), "Invalid Low values"

    assert (df["Vol."] >= 0).all(), "Negative volume found"

    return df