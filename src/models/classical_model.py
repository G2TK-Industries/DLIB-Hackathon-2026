import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================
# Load Dataset
# ===============================

DATA_PATH = "data/processed/merged_monthly.csv"

df = pd.read_csv(DATA_PATH)


# Convert date column
df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values("year_month")

df = df.set_index("year_month")


print("Dataset loaded")
print(df.head())


# ===============================
# Select Target
# ===============================

target = "BDI_mean"


# ===============================
# Select SARIMAX Exogenous Features
# ===============================

features = [

    # BDI behaviour
    "BDI_volatility",
    "BDI_momentum",

    # Rolling averages
    "BDI_roll_3",
    "BDI_roll_6",
    "BDI_roll_12",

    # External economic variables
    "policy_rate_lag1",
    "policy_rate_lag3",
    "policy_rate_lag6",

    "Brent_lag1",
    "Brent_lag3",
    "Brent_lag6",

]


# keep only existing columns

features = [
    col for col in features 
    if col in df.columns
]


print("\nUsing Features:")
print(features)



# ===============================
# Remove Missing Values
# ===============================

model_df = df[[target] + features].dropna()


print("\nRows after cleaning:", len(model_df))


# ===============================
# Train/Test Split
# ===============================

# last 20% for testing

split = int(len(model_df)*0.8)


train = model_df.iloc[:split]
test = model_df.iloc[split:]


y_train = train[target]
y_test = test[target]


X_train = train[features]
X_test = test[features]



print("\nTraining size:", len(train))
print("Testing size:", len(test))


# ===============================
# SARIMAX MODEL
# ===============================

model = SARIMAX(

    y_train,

    exog=X_train,

    order=(1,1,1),

    seasonal_order=(1,1,1,12),

    enforce_stationarity=False,

    enforce_invertibility=False

)



print("\nTraining SARIMAX...")


results = model.fit(
    disp=False
)


print(results.summary())



# ===============================
# Forecast
# ===============================

forecast = results.predict(

    start=len(train),

    end=len(model_df)-1,

    exog=X_test

)



# ===============================
# Evaluation
# ===============================


mae = mean_absolute_error(
    y_test,
    forecast
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        forecast
    )
)



print("\n==============================")
print("SARIMAX BENCHMARK RESULTS")
print("==============================")

print("MAE :", mae)

print("RMSE:", rmse)



# ===============================
# Save Predictions
# ===============================


results_df = pd.DataFrame({

    "actual": y_test,

    "forecast": forecast

})


results_df.to_csv(
    "sarimax_predictions.csv"
)


print("\nPredictions saved:")
print("sarimax_predictions.csv")