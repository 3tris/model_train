# MLflow Experiment Tracking - California Housing

## 1. Task Description

This project uses MLflow to track and compare three XGBoost regression experiments that predict California house prices. Each run uses different hyperparameters (`max_depth`, `learning_rate`). For every run, MLflow records the parameters, the validation metrics (RMSE, MAE, R²), and the trained model as an artifact. The best model is selected using **RMSE** (lower is better).

- Dataset: California Housing (`sklearn.datasets.fetch_california_housing`)
- Split: 80% train / 20% validation, `random_state=42`
- Model: XGBoost (`xgb.train`, 100 boosting rounds)

## 2. How to Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv mlops-env
source mlops-env/bin/activate        # Windows: mlops-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the MLflow server (in a separate terminal, same environment)
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000

# 4. Run the experiments
python train.py

# 5. Open the MLflow UI
# http://127.0.0.1:5000  ->  experiment: California_Housing_Optimization
```

## 3. Comparison Table

| Run | Max Depth | Learning Rate | RMSE | MAE | R² |
|---|---|---|---|---|---|
| Run 1 | 3 | 0.1 | 0.539 | 0.368 | 0.779 |
| **Run 2** | **5** | **0.05** | **0.522** | **0.355** | **0.792** |
| Run 3 | 7 | 0.01 | 0.700 | 0.533 | 0.626 |

All metrics are computed on the 20% validation set.

## 4. Selected Best Model

**Run 2** (`max_depth = 5`, `learning_rate = 0.05`) with a validation RMSE of **0.522**.

## 5. Why This Model?

I selected **Run 2** because it achieved the lowest validation RMSE (0.522), which is the primary selection metric for this task. It also had the best MAE (0.355) and the best R² (0.792), so it was the strongest model on all three metrics.

Run 1 (depth 3, learning rate 0.1) came close (RMSE 0.539), but its shallower trees likely limited how much of the data's structure it could capture. Run 3 (depth 7, learning rate 0.01) performed worst (RMSE 0.700): with a very small learning rate and only 100 boosting rounds, the model likely did not have enough iterations to converge, so it underfit despite having the deepest trees. Run 2 gave the best balance between model complexity and learning speed.
