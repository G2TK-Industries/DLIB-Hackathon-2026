"""
HCP Linkage Analysis
=====================
Note on data: dataset 5 ("Human Capital Project") does not contain
health/education outcome variables -- it contains the same 3 FAO
price indicators (General CPI, Food CPI, Food Inflation) for 4
peer countries (ZAF, NAM, KEN, ZWE). We therefore operationalise
"human capital indicators" as documented explicitly in the memo:
  (1) Botswana General CPI (general_cpi) - proxy for real
      household purchasing power available for health/education
      spending, i.e. the classic channel through which food-price
      shocks erode human capital investment.
  (2) South Africa food inflation (zaf_food_inflation) - Botswana's
      largest trading/labour-migration partner; a regional human-
      capital-pressure benchmark and leading indicator via the
      import channel (Botswana imports most food from/through SA).

Methods: OLS regression (statsmodels) + pairwise Granger causality.
Forward projection: use the Phase 1 SARIMAX 2024 food inflation
forecast to project the General CPI (cost-of-living / human capital
capacity) path for 2024, with a specific numeric example.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

panel = pd.read_csv(f"{OUT}/merged_monthly_panel.csv", parse_dates=["Date"]).set_index("Date")
print(panel.columns.tolist())
sarimax_fc = pd.read_csv(f"{OUT}/sarimax_improved_2024_forecast.csv")

df = panel[["food_inflation", "general_cpi", "zaf_food_inflation"]].dropna()
print("Rows used:", len(df), df.index.min().date(), "to", df.index.max().date())

# ---------------------------------------------------------------
# 1. OLS Regression: general_cpi growth (MoM %) ~ food_inflation
#    (does food inflation erode the growth of general purchasing power?)
# ---------------------------------------------------------------
df["general_cpi_mom_pct"] = df["general_cpi"].pct_change() * 100
reg_df = df.dropna()

X = sm.add_constant(reg_df["food_inflation"])
y = reg_df["general_cpi_mom_pct"]
ols_model = sm.OLS(y, X).fit()
print("\n--- OLS: General CPI MoM% ~ Food Inflation ---")
print(ols_model.summary())

coef = ols_model.params["food_inflation"]
pval = ols_model.pvalues["food_inflation"]
r2 = ols_model.rsquared

# ---------------------------------------------------------------
# 2. Granger causality: food_inflation <-> zaf_food_inflation
# ---------------------------------------------------------------
gc_df = df[["food_inflation", "zaf_food_inflation"]].dropna()

print("\n--- Granger causality: does ZAF food inflation Granger-cause BWA food inflation? ---")
gc1 = grangercausalitytests(gc_df[["food_inflation", "zaf_food_inflation"]], maxlag=3, verbose=True)

print("\n--- Granger causality: does BWA food inflation Granger-cause ZAF food inflation? ---")
gc2 = grangercausalitytests(gc_df[["zaf_food_inflation", "food_inflation"]], maxlag=3, verbose=True)

# extract lag-3 p-values (F-test) for reporting
p_zaf_causes_bwa = gc1[3][0]["ssr_ftest"][1]
p_bwa_causes_zaf = gc2[3][0]["ssr_ftest"][1]

# ---------------------------------------------------------------
# 3. Forward projection using Phase 1 SARIMAX 2024 forecast
# ---------------------------------------------------------------
last_cpi_general = df["general_cpi"].iloc[-1]  # Dec 2023 level
proj_food_inflation = sarimax_fc["forecast"].values  # 12 monthly YoY% values for 2024

# Apply the estimated regression relationship: predicted MoM% CPI growth
# driven by each month's forecast food inflation, compounded forward
predicted_mom = ols_model.params["const"] + coef * proj_food_inflation
cpi_path = [last_cpi_general]
for m in predicted_mom:
    cpi_path.append(cpi_path[-1] * (1 + m / 100))
cpi_path = cpi_path[1:]  # 12 projected monthly levels for 2024

cum_growth_2024 = (cpi_path[-1] / last_cpi_general - 1) * 100

print(f"\n--- Forward projection (2024) ---")
print(f"Dec-2023 General CPI level: {last_cpi_general:.2f}")
print(f"Projected Dec-2024 General CPI level: {cpi_path[-1]:.2f}")
print(f"Implied cumulative CPI growth over 2024: {cum_growth_2024:.2f}%")

results = {
    "ols_coef_food_inflation_on_cpi_mom": float(coef),
    "ols_pvalue": float(pval),
    "ols_r2": float(r2),
    "granger_p_zaf_causes_bwa_lag3": float(p_zaf_causes_bwa),
    "granger_p_bwa_causes_zaf_lag3": float(p_bwa_causes_zaf),
    "dec2023_cpi_general": float(last_cpi_general),
    "projected_dec2024_cpi_general": float(cpi_path[-1]),
    "projected_2024_cumulative_cpi_growth_pct": float(cum_growth_2024),
}
with open(f"{OUT}/hcp_linkage_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved -> hcp_linkage_results.json")

# ---------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------
# Chart 1: historical comovement -- BWA food inflation vs ZAF food inflation
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df.index, df["food_inflation"], label="Botswana food inflation (YoY %)", linewidth=1.8)
ax.plot(df.index, df["zaf_food_inflation"], label="South Africa food inflation (YoY %)", linewidth=1.3, alpha=0.8)
ax.set_title("Historical Comovement: Botswana vs South Africa Food Inflation (2001-2023)")
ax.set_ylabel("YoY % change")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/hcp_comovement_chart.png", dpi=140)
print("Saved -> hcp_comovement_chart.png")

# Chart 2: forward projection -- General CPI level path under 2024 forecast
fig, ax = plt.subplots(figsize=(9, 4.5))
hist_dates = df.index[-24:]
hist_cpi = df["general_cpi"].iloc[-24:]
future_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
ax.plot(hist_dates, hist_cpi, label="Historical General CPI (2022-2023)", color="tab:blue")
ax.plot(future_dates, cpi_path, label="Projected General CPI (2024, from Phase 1 forecast)",
        color="tab:red", linestyle="--", marker="o")
ax.axvline(pd.Timestamp("2024-01-01"), color="gray", linestyle=":", linewidth=1)
ax.set_title("Forward Projection: General CPI Path Implied by 2024 Food Inflation Forecast")
ax.set_ylabel("General CPI (2015=100)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/hcp_projection_chart.png", dpi=140)
print("Saved -> hcp_projection_chart.png")
