import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_forecast(actual, forecast, dates, model_name):

    # Make all arrays equal length
    min_length = min(
        len(actual),
        len(forecast),
        len(dates)
    )

    actual = actual[:min_length]
    forecast = forecast[:min_length]
    dates = dates[:min_length]


    plt.figure(figsize=(12,6))

    plt.plot(
        dates,
        actual,
        label="Actual BDI",
        linewidth=2
    )

    plt.plot(
        dates,
        forecast,
        label=f"{model_name} Forecast",
        linewidth=2
    )

    plt.title(
        f"BDI Actual vs Forecast - {model_name}",
        fontsize=14
    )

    plt.xlabel("Date")
    plt.ylabel("BDI Mean Value")

    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)

    plt.savefig(
        f"plots/{model_name}_forecast.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



# ==============================
# Load evaluation forecast data
# ==============================

import pandas as pd

df = pd.read_csv("merged_monthly.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())


# Actual BDI
actual = df["BDI_mean"]



# Example SARIMAX forecast file
sarimax_pred = pd.read_csv(
    "sarimax_predictions.csv"
)

lstm_pred = pd.read_csv(
    "lstm_predictions.csv"
)
print("\n===== SARIMAX FORECAST COLUMNS =====")
print(sarimax_pred.columns.tolist())

print("\n===== SARIMAX FORECAST DATA =====")
print(sarimax_pred.head())

# Forecast dates
forecast_dates = pd.to_datetime(
    sarimax_pred["Date"]
)



# ==============================
# SARIMAX Plot
# ==============================

plot_forecast(
    actual.tail(len(sarimax_pred)),
    sarimax_pred["Forecast"],
    forecast_dates,
    "SARIMAX"
)



# ==============================
# LSTM Plot
# ==============================

plot_forecast(
    actual.tail(len(lstm_pred)),
    lstm_pred["Forecast"],
    forecast_dates,
    "LSTM"
)