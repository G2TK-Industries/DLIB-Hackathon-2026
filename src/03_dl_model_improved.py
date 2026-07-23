"""
Improved deep-learning forecast for IndabaX Botswana 2026.

Key change from the original LSTM:
- Forecast the next 12 MONTHLY LOG CHANGES in Botswana Food CPI.
- Reconstruct the 12 future Food CPI levels.
- Convert those levels to the official target, YoY food-price inflation.

This target is much more stationary than raw YoY inflation and preserves the
exact identity: food_inflation_t = 100 * (food_cpi_t / food_cpi_{t-12} - 1).
No 2024 values are used. The 2024 denominator is known 2023 Food CPI.
"""
from __future__ import annotations
import json, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

SEED_LIST = [7, 19, 42, 73, 101]
WINDOW = 24
HORIZON = 12
HIDDEN = 16
MAX_EPOCHS = 300
PATIENCE = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4

BASE = Path(__file__).resolve().parent
PANEL_PATH = BASE / "merged_monthly_panel.csv"
OUT_FORECAST = BASE / "lstm_cpi_2024_forecast.csv"
OUT_BEST = BASE / "best_model_predictions.csv"
OUT_METRICS = BASE / "dl_metrics_improved.json"
OUT_PLOT = BASE / "dl_diagnostics_improved.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CPIGrowthLSTM(nn.Module):
    def __init__(self, hidden: int = HIDDEN, horizon: int = HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(self.dropout(h_n[-1]))


def build_samples(log_growth: pd.Series):
    values = log_growth.dropna().values.astype(np.float32)
    dates = log_growth.dropna().index
    samples = []
    for t in range(WINDOW - 1, len(values) - HORIZON):
        x = values[t - WINDOW + 1:t + 1].reshape(WINDOW, 1)
        y = values[t + 1:t + 1 + HORIZON]
        samples.append((dates[t], x, y))
    return samples


def arrays(samples, x_scaler, y_mean, y_std):
    X = np.stack([x_scaler.transform(s[1]) for s in samples]).astype(np.float32)
    Y = np.stack([s[2] for s in samples]).astype(np.float32)
    Y = ((Y - y_mean) / y_std).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)


def train_with_early_stopping(train_samples, val_samples, seed: int):
    set_seed(seed)
    x_scaler = StandardScaler().fit(np.vstack([s[1] for s in train_samples]))
    y_train_raw = np.stack([s[2] for s in train_samples])
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std() + 1e-8)
    X_train, y_train = arrays(train_samples, x_scaler, y_mean, y_std)
    X_val, y_val = arrays(val_samples, x_scaler, y_mean, y_std)

    model = CPIGrowthLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_loss = np.inf
    best_state = None
    best_epoch = 1
    patience = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = loss_fn(train_pred, y_train)
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history.append((epoch, float(train_loss.item()), float(val_loss)))

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, x_scaler, y_mean, y_std, best_epoch, best_loss, history


def refit_all(samples, seed: int, epochs: int):
    set_seed(seed)
    x_scaler = StandardScaler().fit(np.vstack([s[1] for s in samples]))
    y_raw = np.stack([s[2] for s in samples])
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std() + 1e-8)
    X, y = arrays(samples, x_scaler, y_mean, y_std)
    model = CPIGrowthLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, x_scaler, y_mean, y_std


def predict_growth(model, scaler, y_mean, y_std, window_values):
    X = scaler.transform(window_values.reshape(-1, 1)).reshape(1, WINDOW, 1)
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
    return z * y_std + y_mean


def growth_to_inflation(growth, last_cpi, denominator_cpi):
    future_cpi = last_cpi * np.exp(np.cumsum(growth))
    inflation = 100.0 * (future_cpi / denominator_cpi - 1.0)
    return future_cpi, inflation


def main():
    torch.set_num_threads(2)
    panel = pd.read_csv(PANEL_PATH, parse_dates=["Date"]).set_index("Date").asfreq("MS")
    panel["food_cpi_log_growth"] = np.log(panel["food_cpi"]).diff()
    samples = build_samples(panel["food_cpi_log_growth"])

    # Strict 2023 test: model origin Dec-2022 predicts Jan-Dec 2023.
    test_origin = pd.Timestamp("2022-12-01")
    test_sample = [s for s in samples if s[0] == test_origin]
    if len(test_sample) != 1:
        raise RuntimeError("Expected exactly one Dec-2022 test sample")
    trainval = [s for s in samples if s[0] < test_origin]
    split = int(len(trainval) * 0.85)
    train_samples, val_samples = trainval[:split], trainval[split:]

    test_predictions = []
    selected_epochs = []
    histories = []
    actual_2023 = panel.loc["2023-01-01":"2023-12-01", "food_inflation"].values
    denominator_2023 = panel.loc["2022-01-01":"2022-12-01", "food_cpi"].values

    for seed in SEED_LIST:
        model, scaler, y_mean, y_std, best_epoch, best_loss, history = train_with_early_stopping(
            train_samples, val_samples, seed
        )
        growth = predict_growth(model, scaler, y_mean, y_std, test_sample[0][1].reshape(-1))
        _, pred = growth_to_inflation(growth, panel.loc[test_origin, "food_cpi"], denominator_2023)
        test_predictions.append(pred)
        selected_epochs.append(best_epoch)
        histories.append(history)

    test_ensemble = np.mean(test_predictions, axis=0)
    test_rmse = float(np.sqrt(mean_squared_error(actual_2023, test_ensemble)))
    test_mae = float(mean_absolute_error(actual_2023, test_ensemble))
    test_smape = float(100 * np.mean(2 * np.abs(test_ensemble - actual_2023) /
                                     (np.abs(actual_2023) + np.abs(test_ensemble) + 1e-8)))

    # Production refit includes every supervised sample whose target is known by Dec-2023,
    # including the Dec-2022 -> 2023 sample. Epoch count comes from validation only.
    production_samples = [s for s in samples if s[0] <= test_origin]
    production_window = panel.loc["2022-01-01":"2023-12-01", "food_cpi_log_growth"].values
    denominator_2024 = panel.loc["2023-01-01":"2023-12-01", "food_cpi"].values
    last_cpi = float(panel.loc["2023-12-01", "food_cpi"])
    production_predictions = []
    production_cpi = []

    for seed, epochs in zip(SEED_LIST, selected_epochs):
        model, scaler, y_mean, y_std = refit_all(production_samples, seed, epochs)
        growth = predict_growth(model, scaler, y_mean, y_std, production_window)
        cpi_pred, inflation_pred = growth_to_inflation(growth, last_cpi, denominator_2024)
        production_predictions.append(inflation_pred)
        production_cpi.append(cpi_pred)

    forecast = np.mean(production_predictions, axis=0)
    future_cpi = np.mean(production_cpi, axis=0)
    future_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    forecast_df = pd.DataFrame({
        "year_month": future_dates.strftime("%Y-%m"),
        "forecast": forecast,
    })
    forecast_df.to_csv(OUT_FORECAST, index=False)
    forecast_df.to_csv(OUT_BEST, index=False)

    metrics = {
        "model": "LSTM on monthly log Food-CPI growth; direct 12-step output; 5-seed ensemble",
        "window": WINDOW,
        "horizon": HORIZON,
        "hidden_units": HIDDEN,
        "seeds": SEED_LIST,
        "selected_epochs": selected_epochs,
        "n_train_samples": len(train_samples),
        "n_early_stop_validation_samples": len(val_samples),
        "n_production_samples": len(production_samples),
        "test_period": "2023-01 to 2023-12",
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_smape_pct": test_smape,
        "no_2024_feature_values_used": True,
        "target_identity": "100 * (Food_CPI_t / Food_CPI_t-12 - 1)",
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))

    # Diagnostics: holdout forecast and production forecast.
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    d2023 = pd.date_range("2023-01-01", periods=12, freq="MS")
    axes[0].plot(d2023, actual_2023, marker="o", label="Actual 2023")
    axes[0].plot(d2023, test_ensemble, marker="o", label="LSTM holdout forecast")
    axes[0].set_title(f"Strict 12-month holdout: RMSE={test_rmse:.3f}")
    axes[0].set_ylabel("Food inflation (% YoY)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(future_dates, forecast, marker="o", label="2024 forecast")
    axes[1].set_title("Production forecast (no 2024 inputs)")
    axes[1].set_ylabel("Food inflation (% YoY)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)

    print("2023 strict holdout metrics")
    print(f"RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | sMAPE={test_smape:.2f}%")
    print("Selected epochs by seed:", selected_epochs)
    print("\n2024 forecast")
    print(forecast_df.to_string(index=False))
    print(f"\nSaved: {OUT_BEST.name}, {OUT_METRICS.name}, {OUT_PLOT.name}")


if __name__ == "__main__":
    main()
