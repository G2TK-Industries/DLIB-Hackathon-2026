# Forecasting Botswana Food Inflation Using Economic Indicators

## DLIB Hackathon 2026 Submission

## Overview

This project develops a data-driven forecasting framework for predicting **Botswana's monthly food inflation** using historical economic indicators, commodity market signals, and regional inflation relationships.

The project combines:

1. **Classical time-series forecasting**
   - SARIMAX / SARIMA modelling with exogenous economic drivers

2. **Deep learning forecasting**
   - LSTM neural network modelling of Food CPI growth dynamics

3. **Human capital linkage analysis**
   - Statistical analysis of how food-price shocks influence broader cost-of-living pressures

The objective is to provide an accurate and interpretable forecasting system that can support economic planning, policy analysis, and human capital decision-making.

---

# Project Objectives

The main objectives are:

- Forecast Botswana food inflation for 2024.
- Compare classical statistical models against deep learning approaches.
- Incorporate economic drivers such as:
  - Oil prices
  - Baltic Dry Index
  - Monetary policy rate
  - Regional food inflation
- Analyse inflation spillovers between Botswana and neighbouring economies.
- Estimate how food inflation affects household purchasing power through CPI impacts.

---

# Repository Structure

```
DLIB-Hackathon-2026/

│
├── data/
│   ├── 01_baltic_dry_index_daily.csv
│   ├── 02_brent_crude_monthly.csv
│   ├── 03_botswana_policy_rate.csv
│   ├── 04_fao_botswana_prices.csv
│   └── 05_human_capital_project.csv
│
├── dataprocessed/
│   └── merged_monthly_panel.csv
│
├── outputs/
│   ├── sarimax_improved_2024_forecast.csv
│   ├── lstm_cpi_2024_forecast.csv
│   ├── best_model_predictions.csv
│   ├── classical_metrics_improved.json
│   ├── dl_metrics_improved.json
│   ├── hcp_linkage_results.json
│   ├── classical_diagnostics_improved.png
│   ├── dl_diagnostics_improved.png
│   ├── hcp_comovement_chart.png
│   └── hcp_projection_chart.png
│
├── src/
│   ├── 01_prepare_data_improved.py
│   ├── 01_feature_engineering.py
│   ├── 02_classical_model_improved.py
│   ├── 03_dl_model_improved.py
│   └── 04_hcp_linkage.py
│
├── requirements.txt
└── README.md
```

---

# Dataset Description

The project integrates multiple economic datasets.

## Input Variables

### Botswana Economic Indicators

| Variable | Description |
|---|---|
| Food CPI | Botswana food price index |
| Food Inflation | Target variable (% YoY) |
| General CPI | Overall consumer price index |
| Policy Rate | Monetary policy indicator |

---

### Global Market Indicators

| Variable | Description |
|---|---|
| Brent Crude Oil | Global energy price indicator |
| Baltic Dry Index | Global shipping demand indicator |

---

### Regional Inflation Indicators

| Variable | Country |
|---|---|
| South Africa Food Inflation | ZAF |
| Namibia Food Inflation | NAM |
| Kenya Food Inflation | KEN |
| Zimbabwe Food Inflation | ZWE |

---

## Data Frequency

All datasets are converted into a monthly panel.

Historical period:

```
January 2001 - December 2023
```

Forecast period:

```
January 2024 - December 2024
```

---

# Data Processing Pipeline

The project follows the following workflow:

```
Raw Economic Data

        |
        v

Data Cleaning

        |
        v

Monthly Alignment

        |
        v

Feature Engineering

        |
        v

Model Training

        |
        v

2024 Forecast Generation

        |
        v

Human Capital Analysis
```

---

# Model Implementations

The project contains two clearly separated forecasting implementations:

---

# 1. Classical Model Implementation - SARIMAX

## File

```
src/02_classical_model_improved.py
```

## Model Type

**SARIMAX**
(Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)

The classical model forecasts Botswana food inflation using:

- Historical inflation patterns
- Seasonal behaviour
- External economic drivers

---

## SARIMAX Formulations

Two model families are evaluated.

---

## Model A: Direct Food Inflation SARIMAX

Target:

```
food_inflation (% YoY)
```

External regressors:

```
Brent_USD_per_barrel_lag12
bdi_mean_lag12
policy_rate_lag12
zaf_food_inflation_lag12
bdi_momentum_3m_lag12
```

These lagged variables prevent the model from using future information.

---

## Model B: Food CPI Growth SARIMA

Instead of directly forecasting inflation:

```
Food CPI
    |
    v
Monthly log growth
    |
    v
SARIMA forecast
    |
    v
Reconstructed Food CPI
    |
    v
YoY Food Inflation
```

This improves stationarity and preserves the official inflation identity:

```
Food Inflation =
100 × (Food CPI(t) / Food CPI(t-12) - 1)
```

---

## Model Selection

Models are compared using rolling-origin validation.

Validation origins:

```
2019-12
2020-12
2021-12
2022-12
```

Evaluation metrics:

- RMSE
- MAE
- sMAPE

The final model is selected using:

```
Recent weighted RMSE
```

rather than AIC alone.

---

## Running the Classical Model

From the repository root:

```bash
python src/02_classical_model_improved.py
```

---

## Classical Model Outputs

Generated files:

```
outputs/

sarimax_improved_2024_forecast.csv

classical_metrics_improved.json

classical_diagnostics_improved.png
```

---

# 2. Deep Learning Model Implementation - LSTM

## File

```
src/03_dl_model_improved.py
```

---

## Model Type

**Long Short-Term Memory Neural Network (LSTM)**

Framework:

```
PyTorch
```

The model forecasts future Food CPI growth and converts predictions into official food inflation values.

---

# LSTM Architecture

```
Input Sequence

24 months of Food CPI log growth

        |
        v

LSTM Layer

Hidden Units = 16

        |
        v

Dropout

0.15

        |
        v

Fully Connected Layer

        |
        v

12 Month Future Growth Forecast
```

---

## Target Transformation

The model predicts:

```
Monthly log Food CPI growth
```

instead of raw inflation.

Forecast reconstruction:

```
Predicted CPI Growth

        |
        v

Future Food CPI Levels

        |
        v

YoY Food Inflation
```

---

## Training Configuration

| Parameter | Value |
|-|-|
| Input window | 24 months |
| Forecast horizon | 12 months |
| Hidden units | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Loss function | MSE |
| Seeds | 7, 19, 42, 73, 101 |
| Ensemble | 5 LSTM models |

---

## Validation Strategy

Strict holdout testing:

Training:

```
Before December 2022
```

Testing:

```
January 2023 - December 2023
```

After validation, the model is retrained using all available historical samples before generating the 2024 forecast.

---

## Running the LSTM Model

From the repository root:

```bash
python src/03_dl_model_improved.py
```

---

## LSTM Outputs

Generated files:

```
outputs/

lstm_cpi_2024_forecast.csv

best_model_predictions.csv

dl_metrics_improved.json

dl_diagnostics_improved.png
```

---

# Human Capital Linkage Analysis

## File

```
src/04_hcp_linkage.py
```

The project analyses how food inflation affects human capital through household purchasing power.

---

# 1. OLS Regression Analysis

Model:

```
General CPI Monthly Growth
        ~
Food Inflation
```

Results:

| Metric | Value |
|-|-:|
| R² | 0.036 |
| Food Inflation coefficient | 0.0226 |
| p-value | 0.002 |

Interpretation:

Food inflation has a statistically significant positive relationship with overall CPI growth, indicating that rising food prices increase cost-of-living pressure.

---

# 2. Granger Causality Analysis

## South Africa → Botswana Food Inflation

Lag 3:

```
p-value = 0.0009
```

Conclusion:

South African food inflation provides significant predictive information for Botswana food inflation.

This supports the regional import-price transmission channel.

---

## Botswana → South Africa Food Inflation

Lag 3:

```
p-value = 0.1079
```

Conclusion:

There is insufficient evidence that Botswana food inflation drives South African food inflation.

---

# 3. Forward Human Capital Projection

Using the 2024 food inflation forecast:

Starting value:

```
December 2023 General CPI:
146.86
```

Projected:

```
December 2024 General CPI:
155.99
```

Expected cumulative growth:

```
6.22%
```

Interpretation:

Persistent food inflation can reduce household purchasing power and potentially limit spending capacity for education, healthcare, and other human capital investments.

---

# Installation

## Requirements

Python:

```
Python 3.12+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Complete Pipeline

Execute scripts in order:

```bash
python src/01_prepare_data_improved.py

python src/01_feature_engineering.py

python src/02_classical_model_improved.py

python src/03_dl_model_improved.py

python src/04_hcp_linkage.py
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- PyTorch
- SciPy

---



---

# Authors

Developed for:

**DLIB Hackathon 2026**

Project Theme:

**Forecasting Botswana Food Inflation and Understanding Economic Impacts**