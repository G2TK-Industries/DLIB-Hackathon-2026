import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================
# Load Prediction Files
# =========================

sarimax = pd.read_csv(
    "sarimax_predictions.csv"
)

lstm = pd.read_csv(
    "lstm_predictions.csv"
)



# =========================
# Evaluation Function
# =========================

def evaluate_model(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) /
            np.where(actual == 0, 1, actual)
        )
    ) * 100


    return rmse, mae, r2, mape



# =========================
# SARIMAX Evaluation
# =========================

print("\n===== SARIMAX Evaluation =====")


sarimax_rmse, sarimax_mae, sarimax_r2, sarimax_mape = evaluate_model(
    sarimax["actual"],
    sarimax["forecast"]
)


print(f"RMSE : {sarimax_rmse:.4f}")
print(f"MAE  : {sarimax_mae:.4f}")
print(f"R2   : {sarimax_r2:.4f}")
print(f"MAPE : {sarimax_mape:.2f}%")



# =========================
# LSTM Evaluation
# =========================

print("\n===== LSTM Evaluation =====")


lstm_rmse, lstm_mae, lstm_r2, lstm_mape = evaluate_model(
    lstm["Actual_BDI"],
    lstm["Predicted_BDI"]
)


print(f"RMSE : {lstm_rmse:.4f}")
print(f"MAE  : {lstm_mae:.4f}")
print(f"R2   : {lstm_r2:.4f}")
print(f"MAPE : {lstm_mape:.2f}%")



# =========================
# Compare Models
# =========================

results = pd.DataFrame({

    "Model": [
        "SARIMAX",
        "LSTM"
    ],

    "RMSE": [
        sarimax_rmse,
        lstm_rmse
    ],

    "MAE": [
        sarimax_mae,
        lstm_mae
    ],

    "R2 Score": [
        sarimax_r2,
        lstm_r2
    ],

    "MAPE (%)": [
        sarimax_mape,
        lstm_mape
    ]

})


print("\n===== Model Comparison =====")

print(results)



# Save results

results.to_csv(
    "model_evaluation_results.csv",
    index=False
)

print("\nSaved: model_evaluation_results.csv")