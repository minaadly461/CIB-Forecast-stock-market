import mlflow
import joblib

model = mlflow.sklearn.load_model(
    "models:/CIB_Stock_Best_Regression/1"
)

joblib.dump(model, "models/cib_stock_best_regression.pkl")

print("Model saved successfully.")