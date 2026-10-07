from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "CIB_features.csv"

TARGET = "Target"

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close"
]


# =========================
# Load Data
# =========================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# =========================
# Time-Based Split
# =========================

train = df[df["Date"] < "2025-01-01"].copy()

valid = df[
    (df["Date"] >= "2025-01-01") &
    (df["Date"] < "2026-01-01")
].copy()

test = df[
    (df["Date"] >= "2026-01-01") &
    (df["Date"] < "2026-07-01")
].copy()


# =========================
# Remove NaN Rows
# =========================

train = train.dropna(subset=FEATURES + [TARGET])
valid = valid.dropna(subset=FEATURES + [TARGET])
test = test.dropna(subset=FEATURES + [TARGET])


# =========================
# Features / Target
# =========================

X_train = train[FEATURES]
y_train = train[TARGET]

X_valid = valid[FEATURES]
y_valid = valid[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


print("Data shapes:")
print("Train:", X_train.shape)
print("Validation:", X_valid.shape)
print("Test:", X_test.shape)


# =========================
# Models
# =========================

models = {

    "Linear_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random_Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False
    )
}


# =========================
# MLflow
# =========================

mlflow.set_experiment("CIB-Stock-Prediction")


# =========================
# Training
# =========================

results = []

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred_valid = model.predict(X_valid)
        y_pred_test = model.predict(X_test)

        # =====================
        # Validation Metrics
        # =====================

        val_mae = mean_absolute_error(
            y_valid,
            y_pred_valid
        )

        val_rmse = mean_squared_error(
            y_valid,
            y_pred_valid
        ) ** 0.5

        val_r2 = r2_score(
            y_valid,
            y_pred_valid
        )

        # =====================
        # Test Metrics
        # =====================

        test_mae = mean_absolute_error(
            y_test,
            y_pred_test
        )

        test_rmse = mean_squared_error(
            y_test,
            y_pred_test
        ) ** 0.5

        test_r2 = r2_score(
            y_test,
            y_pred_test
        )

        # =====================
        # Log Parameters
        # =====================

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "target",
            TARGET
        )

        mlflow.log_param(
            "features",
            FEATURES
        )

        mlflow.log_param(
            "features_count",
            len(FEATURES)
        )

        # =====================
        # Log Metrics
        # =====================

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # =====================
        # Log Model
        # =====================

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # =====================
        # Save Results
        # =====================

        results.append({
            "Model": model_name,

            "Val_MAE": val_mae,
            "Val_RMSE": val_rmse,
            "Val_R2": val_r2,

            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2
        })


# =========================
# Model Comparison
# =========================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Test_RMSE"
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)