from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cib_stock_best_regression.pkl"
)


# =========================
# Load model
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="CIB Stock Prediction API",
    description="API for predicting the next trading day's CIB closing price",
    version="1.0.0"
)


# =========================
# Input validation
# =========================

class StockInput(BaseModel):

    Open: float = Field(gt=0)
    High: float = Field(gt=0)
    Low: float = Field(gt=0)
    Close: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_ohlc(self):

        if self.High < self.Low:
            raise ValueError(
                "High cannot be lower than Low"
            )

        if self.High < max(self.Open, self.Close):
            raise ValueError(
                "High must be greater than or equal to Open and Close"
            )

        if self.Low > min(self.Open, self.Close):
            raise ValueError(
                "Low must be less than or equal to Open and Close"
            )

        return self


# =========================
# Home endpoint
# =========================

@app.get("/")
def home():

    return {
        "message": "CIB Stock Prediction API is running"
    }


# =========================
# Prediction endpoint
# =========================

@app.post("/predict")
def predict(data: StockInput):

    input_data = pd.DataFrame([{
        "Open": data.Open,
        "High": data.High,
        "Low": data.Low,
        "Close": data.Close
    }])

    prediction = model.predict(input_data)[0]

    return {
        "input": {
            "Open": data.Open,
            "High": data.High,
            "Low": data.Low,
            "Close": data.Close
        },
        "predicted_next_day_close": float(prediction)
    }