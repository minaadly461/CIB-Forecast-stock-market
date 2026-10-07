import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Configuration
# =========================

DATA_PATH = "data/processed/CIB_features.csv"

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
# Prepare Data
# =========================

train = train.dropna(subset=FEATURES + [TARGET])
valid = valid.dropna(subset=FEATURES + [TARGET])
test = test.dropna(subset=FEATURES + [TARGET])

X_train = train[FEATURES]
y_train = train[TARGET]

X_valid = valid[FEATURES]
y_valid = valid[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# =========================
# Model
# =========================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])


# =========================
# Training + MLflow
# =========================

mlflow.set_experiment("CIB-Stock-Prediction")

with mlflow.start_run(run_name="Linear_Regression_OHLC"):

    model.fit(X_train, y_train)

    # Predictions
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    # Validation metrics
    val_mae = mean_absolute_error(y_valid, y_pred_valid)
    val_rmse = mean_squared_error(
        y_valid,
        y_pred_valid
    ) ** 0.5
    val_r2 = r2_score(y_valid, y_pred_valid)

    # Test metrics
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(
        y_test,
        y_pred_test
    ) ** 0.5
    test_r2 = r2_score(y_test, y_pred_test)

    # Log parameters
    mlflow.log_param("model", "Linear Regression")
    mlflow.log_param("target", TARGET)
    mlflow.log_param("features", FEATURES)
    mlflow.log_param("features_count", len(FEATURES))

    # Log validation metrics
    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_r2", val_r2)

    # Log test metrics
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model"
    )


# =========================
# Results
# =========================

print("\n===== Linear Regression =====")

print("\nValidation:")
print(f"MAE  : {val_mae:.6f}")
print(f"RMSE : {val_rmse:.6f}")
print(f"R²   : {val_r2:.6f}")

print("\nTest:")
print(f"MAE  : {test_mae:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"R²   : {test_r2:.6f}")