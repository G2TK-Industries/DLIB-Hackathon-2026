import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ===============================
# Load feature dataset
# ===============================

df = pd.read_csv("merged_monthly.csv")

df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values("year_month")


print("Dataset shape:")
print(df.shape)


# ===============================
# Select features
# ===============================

target = "BDI_mean"


features = [
    col for col in df.columns 
    if col not in ["year_month", target]
]


print("\nFeatures used:")
print(features)


X = df[features]
y = df[[target]]


# ===============================
# Scale data
# ===============================

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()


X_scaled = scaler_X.fit_transform(X)

y_scaled = scaler_y.fit_transform(y)



# ===============================
# Create sequences for LSTM
# ===============================

def create_sequences(X, y, window=12):

    Xs = []
    ys = []

    for i in range(len(X)-window):

        Xs.append(
            X[i:i+window]
        )

        ys.append(
            y[i+window]
        )

    return np.array(Xs), np.array(ys)



WINDOW = 12


X_seq, y_seq = create_sequences(
    X_scaled,
    y_scaled,
    WINDOW
)


print("\nLSTM input shape:")
print(X_seq.shape)


# ===============================
# Train-test split
# ===============================

split = int(len(X_seq)*0.8)


X_train = X_seq[:split]
X_test  = X_seq[split:]


y_train = y_seq[:split]
y_test  = y_seq[split:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))



# ===============================
# Build LSTM Model
# ===============================


model = Sequential()


model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            WINDOW,
            X_train.shape[2]
        )
    )
)


model.add(
    Dropout(0.2)
)


model.add(
    LSTM(
        32
    )
)


model.add(
    Dropout(0.2)
)


model.add(
    Dense(1)
)



model.compile(
    optimizer="adam",
    loss="mse"
)


model.summary()



# ===============================
# Training
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)



# ===============================
# Prediction
# ===============================

pred_scaled = model.predict(X_test)


# Convert back to BDI values

pred = scaler_y.inverse_transform(
    pred_scaled
)


actual = scaler_y.inverse_transform(
    y_test
)



# ===============================
# Evaluation
# ===============================


mae = mean_absolute_error(
    actual,
    pred
)


rmse = np.sqrt(
    mean_squared_error(
        actual,
        pred
    )
)


print("\nMODEL PERFORMANCE")
print("----------------------")
print("MAE:", mae)
print("RMSE:", rmse)



# ===============================
# Save predictions
# ===============================


results = pd.DataFrame({

    "Actual_BDI": actual.flatten(),

    "Predicted_BDI": pred.flatten()

})


results.to_csv(
    "lstm_predictions.csv",
    index=False
)


print("\nPredictions saved!")