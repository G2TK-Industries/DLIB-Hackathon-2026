"""
Improved classical model for IndabaX Botswana 2026.

Builds two legitimate SARIMAX formulations and selects between them using
rolling 12-month-origin validation rather than AIC alone:
A) Direct Food Inflation SARIMAX with lag-12 exogenous drivers.
B) SARIMA on monthly log Food-CPI growth, converted exactly to YoY inflation.

No 2024 macro values are used.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

BASE = Path(__file__).resolve().parent
PANEL_PATH = BASE / "merged_monthly_panel.csv"
OUT_FORECAST = BASE / "sarimax_improved_2024_forecast.csv"
OUT_METRICS = BASE / "classical_metrics_improved.json"
OUT_PLOT = BASE / "classical_diagnostics_improved.png"

EXOG_COLS = [
    "Brent_USD_per_barrel_lag12",
    "bdi_mean_lag12",
    "policy_rate_lag12",
    "zaf_food_inflation_lag12",
    "bdi_momentum_3m_lag12",
]

DIRECT_CANDIDATES = [
    ((1, 1, 1), (0, 0, 1, 12)),
    ((1, 1, 1), (1, 0, 1, 12)),
    ((2, 1, 1), (0, 0, 1, 12)),
    ((1, 1, 2), (1, 0, 1, 12)),
    ((2, 1, 2), (1, 0, 1, 12)),
]
CPI_GROWTH_CANDIDATES = [
    ((0, 0, 1), (0, 0, 0, 12)),
    ((1, 0, 0), (0, 0, 0, 12)),
    ((1, 0, 1), (0, 0, 0, 12)),
    ((2, 0, 1), (0, 0, 0, 12)),
    ((1, 0, 1), (0, 0, 1, 12)),
]
ORIGINS = pd.to_datetime(["2019-12-01", "2020-12-01", "2021-12-01", "2022-12-01"])


def metrics(actual, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "smape_pct": float(100 * np.mean(2 * np.abs(pred - actual) /
                                         (np.abs(actual) + np.abs(pred) + 1e-8))),
    }


def direct_forecast(panel, origin, order, seasonal_order):
    df = panel[["food_inflation"] + EXOG_COLS].dropna()
    train = df.loc[:origin]
    future_idx = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future_x = df.loc[future_idx, EXOG_COLS]
    fit = SARIMAX(
        train["food_inflation"], exog=train[EXOG_COLS],
        order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=250)
    return fit.forecast(12, exog=future_x).values


def cpi_growth_forecast(panel, origin, order, seasonal_order):
    growth = np.log(panel["food_cpi"]).diff().dropna().loc[:origin]
    fit = SARIMAX(
        growth, order=order, seasonal_order=seasonal_order, trend="c",
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=250)
    g = fit.forecast(12).values
    future_idx = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future_cpi = float(panel.loc[origin, "food_cpi"]) * np.exp(np.cumsum(g))
    denominator = panel.loc[future_idx - pd.DateOffset(years=1), "food_cpi"].values
    return 100 * (future_cpi / denominator - 1)


def rolling_select(panel):
    rows = []
    for family, candidates in [("direct_exog", DIRECT_CANDIDATES), ("cpi_growth", CPI_GROWTH_CANDIDATES)]:
        for order, seasonal in candidates:
            fold_rmse = []
            fold_mae = []
            valid = True
            for origin in ORIGINS:
                future_idx = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=12, freq="MS")
                actual = panel.loc[future_idx, "food_inflation"].values
                try:
                    if family == "direct_exog":
                        pred = direct_forecast(panel, origin, order, seasonal)
                    else:
                        pred = cpi_growth_forecast(panel, origin, order, seasonal)
                    m = metrics(actual, pred)
                    fold_rmse.append(m["rmse"])
                    fold_mae.append(m["mae"])
                except Exception:
                    valid = False
                    break
            if valid:
                weights = np.arange(1, len(fold_rmse) + 1, dtype=float)
                rows.append({
                    "family": family,
                    "order": order,
                    "seasonal_order": seasonal,
                    "fold_rmse": fold_rmse,
                    "fold_mae": fold_mae,
                    "mean_rmse": float(np.mean(fold_rmse)),
                    "recent_weighted_rmse": float(np.average(fold_rmse, weights=weights)),
                })
    if not rows:
        raise RuntimeError("No SARIMAX candidate fitted successfully")
    rows.sort(key=lambda r: r["recent_weighted_rmse"])
    return rows[0], rows


def production_fit(panel, selected):
    family = selected["family"]
    order = tuple(selected["order"])
    seasonal = tuple(selected["seasonal_order"])
    future_idx = pd.date_range("2024-01-01", periods=12, freq="MS")

    if family == "direct_exog":
        df = panel[["food_inflation"] + EXOG_COLS].dropna()
        y, X = df["food_inflation"], df[EXOG_COLS]
        fit = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
        raw_2023 = panel.loc["2023-01-01":"2023-12-01"]
        future_x = pd.DataFrame({
            "Brent_USD_per_barrel_lag12": raw_2023["Brent_USD_per_barrel"].values,
            "bdi_mean_lag12": raw_2023["bdi_mean"].values,
            "policy_rate_lag12": raw_2023["policy_rate"].values,
            "zaf_food_inflation_lag12": raw_2023["zaf_food_inflation"].values,
            "bdi_momentum_3m_lag12": raw_2023["bdi_momentum_3m"].values,
        }, index=future_idx)
        result = fit.get_forecast(12, exog=future_x)
        pred = result.predicted_mean.values
        ci = result.conf_int(alpha=0.05).values
        residuals = fit.resid
        aic = float(fit.aic)
    else:
        growth = np.log(panel["food_cpi"]).diff().dropna()
        fit = SARIMAX(growth, order=order, seasonal_order=seasonal, trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
        result = fit.get_forecast(12)
        g = result.predicted_mean.values
        g_ci = result.conf_int(alpha=0.05).values
        last_cpi = float(panel.loc["2023-12-01", "food_cpi"])
        denom = panel.loc["2023-01-01":"2023-12-01", "food_cpi"].values
        cpi_mid = last_cpi * np.exp(np.cumsum(g))
        cpi_low = last_cpi * np.exp(np.cumsum(g_ci[:, 0]))
        cpi_high = last_cpi * np.exp(np.cumsum(g_ci[:, 1]))
        pred = 100 * (cpi_mid / denom - 1)
        ci = np.column_stack([100 * (cpi_low / denom - 1), 100 * (cpi_high / denom - 1)])
        residuals = fit.resid
        aic = float(fit.aic)
    return pred, ci, residuals, aic


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["Date"]).set_index("Date").asfreq("MS")
    adf_stat, adf_p, *_ = adfuller(panel["food_inflation"].dropna())
    selected, leaderboard = rolling_select(panel)
    pred, ci, residuals, aic = production_fit(panel, selected)
    future_idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    out = pd.DataFrame({"year_month": future_idx.strftime("%Y-%m"), "forecast": pred})
    out.to_csv(OUT_FORECAST, index=False)

    lb = acorr_ljungbox(pd.Series(residuals).dropna(), lags=[12], return_df=True)
    report = {
        "model": "Rolling-origin-selected SARIMAX",
        "selected_family": selected["family"],
        "selected_order": list(selected["order"]),
        "selected_seasonal_order": list(selected["seasonal_order"]),
        "rolling_origins": [d.strftime("%Y-%m") for d in ORIGINS],
        "fold_rmse": selected["fold_rmse"],
        "mean_rmse": selected["mean_rmse"],
        "recent_weighted_rmse": selected["recent_weighted_rmse"],
        "adf_stat_food_inflation": float(adf_stat),
        "adf_p_food_inflation": float(adf_p),
        "full_fit_aic": aic,
        "ljung_box_lag12_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "no_2024_macro_values_used": True,
        "top_candidates": leaderboard[:5],
    }
    OUT_METRICS.write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_idx, pred, marker="o", label="SARIMAX forecast")
    ax.fill_between(future_idx, ci[:, 0], ci[:, 1], alpha=0.2, label="95% interval")
    ax.set_title(f"2024 forecast - selected family: {selected['family']}")
    ax.set_ylabel("Food inflation (% YoY)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=160)

    print("Selected:", selected)
    print(out.to_string(index=False))
    print("Saved:", OUT_FORECAST.name, OUT_METRICS.name, OUT_PLOT.name)


if __name__ == "__main__":
    main()
