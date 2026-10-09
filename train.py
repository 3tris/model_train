import os

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "California_Housing_Optimization"
RANDOM_SEED = 42

RUN_CONFIGS = [
    {"name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]


def prepare_data():
    """Load California Housing and split 80% train / 20% validation."""
    print("Fetching and splitting California Housing data...")
    data = fetch_california_housing()
    X_train, X_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_train, X_val, y_train, y_val


def execute_run(config, X_train, X_val, y_train, y_val):
    """Train one model, log params / metrics / model to MLflow, return results."""
    with mlflow.start_run(run_name=config["name"]) as run:
        params = {
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": RANDOM_SEED,
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
        )

        preds = model.predict(dval)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        print(f"{config['name']} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        return {
            "name": config["name"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "run_id": run.info.run_id,
        }


def print_summary(results):
    """Print a markdown comparison table and the best run (lowest RMSE)."""
    print("\n| Run | Max Depth | Learning Rate | RMSE | MAE | R2 |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['name']} | {r['max_depth']} | {r['learning_rate']} | "
            f"{r['rmse']:.4f} | {r['mae']:.4f} | {r['r2']:.4f} |"
        )
    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest model (lowest RMSE): {best['name']} (run_id={best['run_id']})")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val = prepare_data()

    results = [execute_run(c, X_train, X_val, y_train, y_val) for c in RUN_CONFIGS]
    print_summary(results)


if __name__ == "__main__":
    main()
