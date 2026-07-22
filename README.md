# IndabaX Botswana 2026 — Forecasting Food Price Inflation

Team: <team name>
Members: <names>

## Project Description
Two-model forecast (classical baseline + deep learning) of Botswana's monthly
food price inflation (FAO item 23014) for Jan-Dec 2024, using 5 macro/HCP datasets.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Python 3.10+ required.

## Reproduce the pipeline
```bash
python src/data_pipeline.py              # merge 5 raw datasets -> data/processed/
python src/generate_predictions.py --model classical
python src/generate_predictions.py --model deep
python src/evaluate.py                    # RMSE/MAE + residual diagnostics for both
```

## Repository Structure
- `data/raw/` — original 5 datasets (unmodified)
- `data/processed/` — merged monthly panel
- `src/` — data pipeline, feature engineering, both models, evaluation
- `notebooks/` — exploratory + model development notebooks
- `reports/` — Deliverables 1.1b, 1.1c, 1.2a, 1.2b
- `submissions/predictions.csv` — Deliverable 1.1a
- `phase2/` — scenario analysis, policy brief, slide deck (post-advancement)

## Deliverables Map
| Ref | File |
|---|---|
| 1.1a | submissions/predictions.csv |
| 1.1b | reports/feature_engineering_report.pdf |
| 1.1c | reports/model_comparison_report.pdf |
| 1.1d | this repository |
| 1.2a | reports/hcp_linkage_memo.pdf |
| 1.2b | reports/hcp_visualisations/ |

## Feature-Forecasting Strategy for 2024
<state your choice: lagged-only / VAR / two-stage, and why>

## Model Comparison Summary
<classical RMSE vs deep RMSE, which won, why>
