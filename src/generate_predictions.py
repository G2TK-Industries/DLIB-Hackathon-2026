import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os


# ==============================
# Paths
# ==============================

DATA_PATH = "merged_monthly.csv"
OUTPUT_PATH = "final_predictions.csv"


# ==============================
# Load Data
# ==============================

df = pd.read_csv(DATA_PATH)


# Convert year_month to datetime

df["year_month"] = pd.to_datetime(
    df["year_month"]
)


df = df.sort_values(
    "year_month"
)


# Set date index

df = df.set_index(
    "year_month"
)


# Target

target = "BDI_mean"


series = df[target]


print(series.head())


# ==============================
# Train Final SARIMAX
# ==============================

print("\nTraining SARIMAX...")


model = SARIMAX(
    series,
    order=(1,1,1),
    seasonal_order=(1,1,1,12),
    enforce_stationarity=False,
    enforce_invertibility=False
)


sarimax_model = model.fit()


print("Training completed")


# ==============================
# Forecast Future Months
# ==============================

forecast_periods = 12


forecast = sarimax_model.forecast(
    steps=forecast_periods
)


# ==============================
# Create Future Dates
# ==============================

future_dates = pd.date_range(
    start=series.index[-1],
    periods=forecast_periods + 1,
    freq="ME"
)[1:]


# ==============================
# Save Predictions
# ==============================

prediction_df = pd.DataFrame({

    "year_month": future_dates,
    "BDI_prediction": forecast.values

})


prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nFinal Predictions:")
print(prediction_df)


print(
    "\nSaved to:",
    OUTPUT_PATH
)


# ==============================
# Plot
# ==============================

plt.figure(figsize=(12,6))


plt.plot(
    series.index,
    series,
    label="Historical BDI"
)


plt.plot(
    prediction_df["year_month"],
    prediction_df["BDI_prediction"],
    marker="o",
    label="SARIMAX Forecast"
)


plt.title(
    "BDI Mean Forecast - SARIMAX"
)


plt.xlabel("Year")
plt.ylabel("BDI Mean")

plt.legend()
plt.grid()

plt.show()