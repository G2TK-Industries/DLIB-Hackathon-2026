# Forecasting Botswana's Human Capital

## Overview

Human capital development is a key driver of Botswana's economic growth, productivity, innovation, and long-term national development.

Accurate forecasting of human capital trends can support evidence-based decision-making, policy planning, workforce development strategies, and investment prioritization in education, healthcare, and skills development.

This project focuses on forecasting Botswana's Human Capital using historical socioeconomic and economic indicators.

Two forecasting approaches were implemented and compared:

- **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)**
- **LSTM (Long Short-Term Memory Neural Network)**

The objective of this project is to identify the most suitable forecasting model for predicting future human capital trends in Botswana.

---


# Forecasting Models

## 1. SARIMAX Model

SARIMAX was implemented as the statistical forecasting approach.

SARIMAX combines:

- Autoregressive components
- Moving average components
- Differencing
- Seasonal patterns
- External variables


### Advantages

- Suitable for time-series forecasting
- Captures historical dependencies
- Handles external influencing variables
- Provides interpretable forecasts


---

## 2. LSTM Model

A Long Short-Term Memory neural network was implemented to capture nonlinear patterns and long-term dependencies.
The LSTM model learns sequential relationships from previous observations to predict future human capital values.

---

# Model Evaluation

The models were evaluated using:

| Metric | Description |
|---|---|
| RMSE | Measures prediction error magnitude |
| MAE | Measures average absolute forecasting error |
| R² Score | Measures explained variation |
| MAPE | Measures percentage forecasting error |


## Evaluation Results

| Model | RMSE | MAE | R² Score | MAPE (%) |
|---|---:|---:|---:|---:|
| SARIMAX | 1262.4350 | 959.4601 | -1.0149 | 46.4661 |
| LSTM | 1637.7253 | 1444.5431 | -2.2842 | 84.0315 |


---

# Model Performance Analysis

## SARIMAX Performance

SARIMAX achieved:

- Lower RMSE
- Lower MAE
- Lower MAPE

compared to LSTM.

This indicates that SARIMAX produced more accurate forecasts for the available Botswana human capital dataset.


## LSTM Performance

The LSTM model was able to learn nonlinear relationships but produced higher forecasting errors.

Possible reasons include:

- Limited historical data availability
- Small time-series sample size
- High model complexity compared to available data


## Selected Model

Based on the evaluation metrics:

**SARIMAX was selected as the final forecasting model.**

---

# How to Run the Models


## Step 1: Feature Engineering

Generate forecasting features:

```bash
python src/features.py
```

This creates:

- Lag features
- Rolling averages
- Growth indicators
- Trend features

---

# Step 2: Train SARIMAX Model

Run:

```bash
python src/models/classical_model.py
```

The model will:

- Load processed data
- Train SARIMAX
- Generate forecasts
- Save predictions
- Create visualization outputs


Output:

```
results/

└── SARIMAX_forecast.png
```

---

# Step 3: Train LSTM Model

Run:

```bash
python src/models/deep_model.py
```

The model will:

- Normalize input data
- Create time sequences
- Train the neural network
- Generate predictions
- Save the trained model


Output:

```
results/

└── LSTM_forecast.png
```

---

---

# Project Structure

```
Forecasting-Botswana-Human-Capital/

│
├── data/
│   └── human_capital_dataset.csv
│
├── src/
│   ├── features.py
│   ├── sarimax_model.py
│   ├── lstm_model.py
│   ├── evaluate.py
│   └── forecast.py
│
├── models/
│   ├── sarimax_model.pkl
│   └── lstm_model.h5
│
├── results/
│   ├── SARIMAX_forecast.png
│   ├── LSTM_forecast.png
│   ├── model_comparison.png
│   ├── residuals.png
│   └── future_forecast.png
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

## Programming Language

- Python


## Machine Learning Libraries

- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- TensorFlow / Keras


# Conclusion

This project demonstrates the application of statistical and deep learning approaches for forecasting Botswana's human capital development.

By comparing SARIMAX and LSTM models, SARIMAX achieved better forecasting performance based on RMSE, MAE, and MAPE evaluation metrics.

The developed forecasting framework provides insights into future human capital trends and can support data-driven planning and policy decisions for Botswana's development.

---

# Author

**G2TK Industries**
