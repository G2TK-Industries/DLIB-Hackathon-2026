"""
IndabaX Botswana 2026 — Feature Engineering Pipeline
======================================================
Merges 5 datasets (daily BDI, monthly Brent, monthly policy rate,
monthly FAO Botswana prices, monthly HCP cross-country panel) into
a single monthly panel with engineered features.

Design choices (documented for the Feature Engineering Report):
- BDI (daily) is aggregated to monthly using MULTIPLE features, not
  just a mean: level, volatility, trend, extreme-day counts, momentum.
- All contemporaneous (same-month) features are built first, for a
  clean 2000-2023 training panel where target and features are
  co-dated.
- Lag features (1, 3, 6, 12 months) are then added for every driver
  variable, because Jan-Dec 2024 forecasts CANNOT use 2024 values of
  BDI / Brent / policy rate (they don't exist -- data ends Dec 2023).
  Strategy chosen: (a) lagged-features-only as the primary strategy,
  with 12-month lag being the one usable at forecast time for a
  12-month-ahead horizon. Two-stage (forecast BDI/Brent first) is
  left as an optional extension in the modelling stage.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 140)

import os
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ---------------------------------------------------------------
# 1. BDI daily -> monthly multi-feature aggregation
# ---------------------------------------------------------------
bdi = pd.read_csv(f"{DATA}/01_baltic_dry_index_daily.csv", parse_dates=["Date"])
bdi = bdi.sort_values("Date").reset_index(drop=True)
bdi["month"] = bdi["Date"].dt.to_period("M")
bdi["daily_ret"] = bdi["BDI_Close"].pct_change()

def bdi_month_features(g):
    close = g["BDI_Close"].values
    rets = g["daily_ret"].dropna().values
    n = len(close)
    half = max(n // 2, 1)
    first_half_mean = close[:half].mean()
    second_half_mean = close[half:].mean() if n > half else close[-1]
    month_return = (close[-1] / close[0] - 1) if close[0] != 0 else np.nan
    extreme_days = np.sum(np.abs(rets) > 0.03) if len(rets) else 0
    pct_extreme = extreme_days / len(rets) if len(rets) else 0.0
    return pd.Series({
        "bdi_mean": close.mean(),
        "bdi_std": close.std(),
        "bdi_min": close.min(),
        "bdi_max": close.max(),
        "bdi_range": close.max() - close.min(),
        "bdi_cv": close.std() / close.mean() if close.mean() else np.nan,
        "bdi_month_return": month_return,
        "bdi_first_half_mean": first_half_mean,
        "bdi_second_half_mean": second_half_mean,
        "bdi_trend_shift": second_half_mean - first_half_mean,
        "bdi_extreme_day_count": extreme_days,
        "bdi_pct_extreme_days": pct_extreme,
        "bdi_n_days": n,
    })

bdi_monthly = bdi.groupby("month").apply(bdi_month_features).reset_index()
bdi_monthly["Date"] = bdi_monthly["month"].dt.to_timestamp()

# momentum: 3-month rolling change in monthly mean level, and rolling vol
bdi_monthly = bdi_monthly.sort_values("Date").reset_index(drop=True)
bdi_monthly["bdi_momentum_3m"] = bdi_monthly["bdi_mean"].pct_change(3)
bdi_monthly["bdi_rolling_vol_3m"] = bdi_monthly["bdi_mean"].rolling(3).std()
bdi_monthly["bdi_rolling_mean_6m"] = bdi_monthly["bdi_mean"].rolling(6).mean()
bdi_monthly = bdi_monthly.drop(columns=["month"])

print("BDI monthly features:", bdi_monthly.shape)

# ---------------------------------------------------------------
# 2. Brent crude (monthly)
# ---------------------------------------------------------------
brent = pd.read_csv(f"{DATA}/02_brent_crude_monthly.csv", parse_dates=["Date"])
brent["Date"] = brent["Date"].values.astype("datetime64[M]")  # normalise to month start
brent = brent.sort_values("Date").reset_index(drop=True)
brent["brent_mom_pct"] = brent["Brent_USD_per_barrel"].pct_change()
brent["brent_yoy_pct"] = brent["Brent_USD_per_barrel"].pct_change(12)
brent["brent_rolling_mean_3m"] = brent["Brent_USD_per_barrel"].rolling(3).mean()
brent["brent_rolling_vol_3m"] = brent["Brent_USD_per_barrel"].rolling(3).std()
print("Brent monthly:", brent.shape)

# ---------------------------------------------------------------
# 3. Botswana policy rate (monthly)
# ---------------------------------------------------------------
rate = pd.read_csv(f"{DATA}/03_botswana_policy_rate.csv", parse_dates=["Date"])
rate["Date"] = rate["Date"].values.astype("datetime64[M]")
rate = rate.sort_values("Date").reset_index(drop=True)
rate["policy_rate_change"] = rate["policy_rate"].diff()
rate["policy_rate_change_12m"] = rate["policy_rate"].diff(12)
print("Policy rate monthly:", rate.shape)

# ---------------------------------------------------------------
# 4. FAO Botswana prices (target + CPI) -- long -> wide
# ---------------------------------------------------------------
fao = pd.read_csv(f"{DATA}/04_fao_botswana_prices.csv", parse_dates=["Date"])
fao["Date"] = fao["Date"].values.astype("datetime64[M]")
fao_wide = fao.pivot_table(index="Date", columns="Item Code", values="Value").reset_index()
fao_wide = fao_wide.rename(columns={
    23012: "bwa_cpi_general",
    23013: "bwa_cpi_food",
    23014: "food_inflation",  # <-- TARGET
})
fao_wide["bwa_cpi_food_mom_pct"] = fao_wide["bwa_cpi_food"].pct_change()
fao_wide["bwa_cpi_general_mom_pct"] = fao_wide["bwa_cpi_general"].pct_change()
print("FAO Botswana wide:", fao_wide.shape, list(fao_wide.columns))

# ---------------------------------------------------------------
# 5. HCP cross-country panel -- long -> wide (per country, per indicator)
# ---------------------------------------------------------------
hcp = pd.read_csv(f"{DATA}/05_human_capital_project.csv", parse_dates=["Date"])
hcp["Date"] = hcp["Date"].values.astype("datetime64[M]")
hcp_bwa_free = hcp[hcp["REF_AREA"] != "BWA"]  # BWA already fully captured via dataset 4
ind_map = {"FAO_CP_23012": "cpi_general", "FAO_CP_23013": "cpi_food", "FAO_CP_23014": "food_inflation"}
hcp_bwa_free = hcp_bwa_free.copy()
hcp_bwa_free["colname"] = hcp_bwa_free["REF_AREA"].str.lower() + "_" + hcp_bwa_free["INDICATOR"].map(ind_map)
hcp_wide = hcp_bwa_free.pivot_table(index="Date", columns="colname", values="Value").reset_index()
print("HCP cross-country wide:", hcp_wide.shape, list(hcp_wide.columns))

# ---------------------------------------------------------------
# 6. Merge everything into one monthly panel
# ---------------------------------------------------------------
panel = fao_wide.merge(brent, on="Date", how="left") \
                 .merge(rate, on="Date", how="left") \
                 .merge(bdi_monthly, on="Date", how="left") \
                 .merge(hcp_wide, on="Date", how="left")

panel = panel.sort_values("Date").reset_index(drop=True)
print("\nMerged panel shape:", panel.shape)
print("Date range:", panel["Date"].min(), "to", panel["Date"].max())

# ---------------------------------------------------------------
# 7. Lag features (1, 3, 6, 12 months) for every driver variable
#    -- required because 2024 raw values of BDI/Brent/policy rate
#    do not exist. The 12-month lag is what will actually be usable
#    to build a genuine 12-month-ahead forecast without phantom
#    future features.
# ---------------------------------------------------------------
driver_cols = [c for c in panel.columns if c not in ("Date", "food_inflation")]
lag_periods = [1, 3, 6, 12]

lagged_frames = [panel[["Date", "food_inflation"]]]
for col in driver_cols:
    for lag in lag_periods:
        lagged_frames.append(panel[[col]].shift(lag).rename(columns={col: f"{col}_lag{lag}"}))

panel_full = pd.concat([panel] + [f for f in lagged_frames if "Date" not in f.columns] + [lagged_frames[0][["food_inflation"]].rename(columns={"food_inflation":"food_inflation_dup"})], axis=1)
# drop the helper dup column, keep original food_inflation from panel
panel_full = panel_full.drop(columns=["food_inflation_dup"])

print("\nFinal panel with lags shape:", panel_full.shape)
print("Missing values in target:", panel_full["food_inflation"].isna().sum())

# ---------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------
panel_full.to_csv(f"{OUT}/merged_monthly_panel.csv", index=False)
bdi_monthly.to_csv(f"{OUT}/bdi_monthly_features.csv", index=False)

print("\nSaved: merged_monthly_panel.csv, bdi_monthly_features.csv")
print("\nColumn count breakdown:")
print(" - Contemporaneous + engineered (non-lag):", len([c for c in panel.columns if c not in ('Date','food_inflation')]))
print(" - Lag features added:", len(driver_cols) * len(lag_periods))
print(" - Total columns in final panel:", panel_full.shape[1])

print("\nSample of final panel (last 5 rows, key columns):")
key_cols = ["Date", "food_inflation", "bwa_cpi_food", "Brent_USD_per_barrel",
            "policy_rate", "bdi_mean", "bdi_momentum_3m",
            "Brent_USD_per_barrel_lag12", "bdi_mean_lag12", "zaf_food_inflation"]
print(panel_full[key_cols].tail(5).to_string(index=False))
