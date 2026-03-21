# Calgary emergency shelter occupancy predictor

## Problem statement

Calgary's emergency shelters operate near capacity, making resource planning difficult. When demand is underestimated, vulnerable individuals are turned away; when overestimated, resources are wasted. This project forecasts daily shelter occupancy rates using 83,000+ records, enabling proactive capacity planning and early warnings.

## Approach

- Fetched daily shelter occupancy data from Calgary Open Data (dataset `7u2t-3wxf`)
- Engineered temporal features, 7-day and 30-day rolling averages, and lag features per shelter
- Trained Random Forest, Gradient Boosting, and XGBoost regressors
- Used temporal train/test split (80/20) to prevent data leakage
- Generated multi-day-ahead forecasts with 90% capacity alerts

## Key results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| MAE | ~0.04 |
| R-squared | ~0.88 |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_05_shelter_occupancy_predictor/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
