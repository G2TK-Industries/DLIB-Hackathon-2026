# Forecasting Botswana's Human Capital

## Overview

Human capital development is a key driver of Botswana's economic growth, productivity, and long-term national development. Accurate forecasting of human capital trends can support evidence-based decision-making, policy planning, and strategic investment in education, skills development, and workforce improvement.

This project focuses on forecasting Botswana's Human Capital using historical socioeconomic and economic indicators.

Two forecasting approaches were implemented and compared:

- **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)**
- **LSTM (Long Short-Term Memory Neural Network)**

The objective of this project is to determine the most suitable forecasting approach for predicting future human capital trends in Botswana.

---

# Project Workflow

The project follows a complete time-series forecasting pipeline:

```
Data Collection

        ↓

Data Preprocessing

        ↓

Feature Engineering

        ↓

Model Training

        ↓

Forecast Generation

        ↓

Model Evaluation

        ↓

Future Human Capital Prediction
```

---

# Dataset

The dataset contains historical indicators related to Botswana's human capital development.

The target variable is:

```
Human Capital Index
```

Additional socioeconomic and economic indicators were used as predictive variables.

The dataset was prepared into a time-series forecasting format to capture historical patterns and future trends.

---

# Feature Engineering

To improve forecasting performance, additional time-series features were generated.

Created features include:

- Human Capital growth rate
- Human Capital change
- Rolling averages
- Trend indicators
- Lag variables



# Forecasting Models

## 1. SARIMAX Model

SARIMAX was implemented as a statistical forecasting model.

SARIMAX combines:

- Autoregressive components
- Moving average components
- Differencing
- Seasonal patterns
- External variables

Advantages:

- Suitable for time-series forecasting
- Captures historical dependencies
- Provides interpretable results


---

## 2. LSTM Model

A Long Short-Term Memory neural network was implemented to learn complex temporal relationships.

LSTM architecture:

```
Historical Time-Series Input

          ↓

       LSTM Layer

          ↓

      Dropout Layer

          ↓

     Dense Layer

          ↓

Human Capital Forecast
```

LSTM is capable of capturing nonlinear patterns and long-term dependencies within sequential data.

---

# Forecast Visualizations

## Actual vs SARIMAX Forecast

<img src="./results/SARIMAX_forecast.png" width="800">


## Actual vs LSTM Forecast

<img src="./results/LSTM_forecast.png" width="800">


---

# Model Evaluation

The models were evaluated using four forecasting metrics:

| Metric | Description |
|---|---|
| RMSE | Measures average prediction error magnitude |
| MAE | Measures average absolute forecasting error |
| R² Score | Measures how well the model explains variation |
| MAPE | Measures percentage forecasting error |


## Evaluation Results

| Model | RMSE | MAE | R² Score | MAPE (%) |
|---|---:|---:|---:|---:|
| SARIMAX | 1262.4350 | 959.4601 | -1.0149 | 46.4661 |
| LSTM | 1637.7253 | 1444.5431 | -2.2842 | 84.0315 |


---

# Model Performance Analysis

Based on the evaluation results:

### SARIMAX

- Achieved lower RMSE compared to LSTM
- Achieved lower MAE compared to LSTM
- Produced lower forecasting error percentage (MAPE)

### LSTM

- Captured nonlinear relationships but produced higher forecasting errors on the available dataset.

Based on the evaluation metrics, **SARIMAX was selected as the better-performing forecasting model for this dataset.**

Selected Model:

## SARIMAX

---

# Future Forecast

The selected SARIMAX model was used to forecast future human capital trends in Botswana.

Forecast output:

![Future Human Capital Forecast](images/future_forecast.png)


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


## Visualization

- Matplotlib


---

# Project Structure

```
Forecasting-Botswana-Human-Capital/

│
├── data/
│   └── human_capital_dataset.csv
│
├── src/
│   ├── feature_engineering.py
│   ├── sarimax_model.py
│   ├── lstm_model.py
│   ├── evaluate.py
│   └── forecast.py
│
├── models/
│   ├── sarimax_model.pkl
│   └── lstm_model.h5
│
├── images/
│   ├── sarimax_forecast.png
│   ├── lstm_forecast.png
│   ├── model_comparison.png
│   ├── residuals.png
│   └── future_forecast.png
│
└── README.md
```

---

# Conclusion

This project demonstrates the application of statistical and deep learning techniques for forecasting Botswana's human capital development.

By comparing SARIMAX and LSTM models, the project identified SARIMAX as the most suitable forecasting approach based on RMSE, MAE, and MAPE performance.

The developed forecasting framework can support future decision-making by providing insights into expected human capital trends and enabling proactive development planning.

---

# Author

**G2TK Industries**

DLIB Hackathon 2026
