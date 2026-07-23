"""Merge the five IndabaX Botswana datasets and engineer leakage-safe features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def monthly_bdi(group: pd.DataFrame) -> pd.Series:
    close = group["BDI_Close"].dropna()
    daily_ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    first = group.loc[group["Date"].dt.day <= 15, "BDI_Close"].mean()
    second = group.loc[group["Date"].dt.day > 15, "BDI_Close"].mean()
    mean = close.mean()
    std = close.std(ddof=1)
    return pd.Series({
        "bdi_mean": mean,
        "bdi_std": std,
        "bdi_min": close.min(),
        "bdi_max": close.max(),
        "bdi_range": close.max() - close.min(),
        "bdi_cv": std / mean if mean else np.nan,
        "bdi_month_return": close.iloc[-1] / close.iloc[0] - 1 if len(close) > 1 else 0.0,
        "bdi_first_second_change": second / first - 1 if pd.notna(first) and pd.notna(second) and first else np.nan,
        "bdi_pct_extreme_days": float((daily_ret.abs() > 0.03).mean()) if len(daily_ret) else 0.0,
        "bdi_positive_day_share": float((daily_ret > 0).mean()) if len(daily_ret) else np.nan,
        "bdi_last": close.iloc[-1],
    })


def main() -> None:
    bdi = pd.read_csv(BASE / "01_baltic_dry_index_daily.csv", parse_dates=["Date"]).sort_values("Date")
    brent = pd.read_csv(BASE / "02_brent_crude_monthly.csv", parse_dates=["Date"]).sort_values("Date")
    policy = pd.read_csv(BASE / "03_botswana_policy_rate.csv", parse_dates=["Date"]).sort_values("Date")
    fao = pd.read_csv(BASE / "04_fao_botswana_prices.csv", parse_dates=["Date"]).sort_values("Date")
    hcp = pd.read_csv(BASE / "05_human_capital_project.csv", parse_dates=["Date"]).sort_values("Date")

    bdi["Month"] = bdi["Date"].dt.to_period("M").dt.to_timestamp()
    bdi_monthly = bdi.groupby("Month").apply(monthly_bdi, include_groups=False).reset_index().rename(columns={"Month": "Date"})
    bdi_monthly["bdi_momentum_3m"] = bdi_monthly["bdi_mean"].pct_change(3)
    bdi_monthly["bdi_roll3_mean"] = bdi_monthly["bdi_mean"].rolling(3).mean()
    bdi_monthly["bdi_roll6_mean"] = bdi_monthly["bdi_mean"].rolling(6).mean()
    bdi_monthly["bdi_roll3_std"] = bdi_monthly["bdi_mean"].rolling(3).std()

    brent["Date"] = brent["Date"].dt.to_period("M").dt.to_timestamp()
    brent["brent_mom_pct"] = brent["Brent_USD_per_barrel"].pct_change(3)
    brent["brent_yoy_pct"] = brent["Brent_USD_per_barrel"].pct_change(12)
    brent["brent_roll3_mean"] = brent["Brent_USD_per_barrel"].rolling(3).mean()

    policy["Date"] = policy["Date"].dt.to_period("M").dt.to_timestamp()
    policy["policy_change_1m"] = policy["policy_rate"].diff()
    policy["policy_change_12m"] = policy["policy_rate"].diff(12)

    fao_wide = fao.pivot(index="Date", columns="Item Code", values="Value").reset_index().rename(columns={
        23012: "general_cpi", 23013: "food_cpi", 23014: "food_inflation"
    })

    regional = hcp.loc[hcp["INDICATOR"] == "FAO_CP_23014"].pivot(
        index="Date", columns="REF_AREA", values="Value"
    ).reset_index().rename(columns={
        "BWA": "bwa_food_inflation_hcp",
        "ZAF": "zaf_food_inflation",
        "NAM": "nam_food_inflation",
        "KEN": "ken_food_inflation",
        "ZWE": "zwe_food_inflation",
    })

    panel = (fao_wide.merge(bdi_monthly, on="Date", how="left")
             .merge(brent, on="Date", how="left")
             .merge(policy, on="Date", how="left")
             .merge(regional, on="Date", how="left")
             .sort_values("Date").reset_index(drop=True))

    # Leakage-safe 12-month drivers: for every forecast month t, only values at t-12 are used.
    for col in ["Brent_USD_per_barrel", "bdi_mean", "policy_rate", "zaf_food_inflation", "bdi_momentum_3m"]:
        panel[f"{col}_lag12"] = panel[col].shift(12)

    panel["food_cpi_log_growth"] = np.log(panel["food_cpi"]).diff()
    panel.to_csv(BASE / "merged_monthly_panel.csv", index=False)
    print("Saved merged_monthly_panel.csv", panel.shape)
    print(panel[["Date", "food_inflation", "food_cpi", "bdi_mean", "Brent_USD_per_barrel", "policy_rate"]].tail())


if __name__ == "__main__":
    main()
