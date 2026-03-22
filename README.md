<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Shelter%20Occupancy%20Predictor&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Forecasting%20Calgary%20emergency%20shelter%20demand%20from%2083K%2B%20records&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Prophet-Forecasting-3b5998?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XGBoost-0.88_R²-blue?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Calgary's emergency shelters operate near capacity, making resource planning difficult. When demand is underestimated, vulnerable individuals are turned away; when overestimated, resources are wasted.
>
> **Solution** -- This project forecasts daily shelter occupancy rates using 83,000+ records, combining Prophet for trend decomposition with XGBoost for multi-day-ahead predictions and 90% capacity alerts.
>
> **Impact** -- Enables proactive capacity planning, early overcrowding warnings, and optimized resource allocation for Calgary's most vulnerable populations.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| MAE | ~0.04 |
| R-squared | ~0.88 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Daily shelter   │────>│  Rolling avgs    │────>│  Model suite   │────>│  Streamlit      │
│  Data (Socrata) │     │  occupancy data  │     │  7d / 30d        │     │  Prophet       │     │  dashboard      │
│  83K+ records   │     │  Per-shelter     │     │  Lag features    │     │  XGBoost       │     │  Forecast view  │
│  Dataset 7u2t   │     │  cleaning        │     │  Temporal feats  │     │  RF / GB       │     │  Capacity alert │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_05_shelter_occupancy_predictor/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── shelter_occupancy_raw.csv   # Cached occupancy data
├── models/                         # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & preprocessing
    └── model.py                    # Model training & forecasting
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/shelter-occupancy-predictor.git
cd shelter-occupancy-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Emergency Shelter Occupancy](https://data.calgary.ca/) (dataset `7u2t-3wxf`) |
| Records | 83,000+ |
| Access method | Socrata API (sodapy) |
| Key fields | Date, shelter name, capacity, overnight count, occupancy rate |
| Target variable | Daily occupancy rate (0.0 -- 1.0) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Prophet-3b5998?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and cleaning

- Fetched daily shelter occupancy data from Calgary Open Data (dataset `7u2t-3wxf`)
- Cleaned and standardized shelter names and date formats
- Computed occupancy rate as overnight count divided by capacity

### Feature engineering

- Engineered temporal features: day-of-week, month, season, holidays
- Created 7-day and 30-day rolling averages and lag features per shelter
- Built per-shelter feature matrices to capture facility-specific patterns

### Model training and evaluation

- Trained Random Forest, Gradient Boosting, and XGBoost regressors
- Used temporal train/test split (80/20) to prevent data leakage
- XGBoost achieved the best R-squared of ~0.88 with MAE of ~0.04

### Forecasting and alerts

- Generated multi-day-ahead forecasts with iterative prediction
- Implemented 90% capacity alerts to flag shelters approaching overflow
- Prophet used for trend decomposition and seasonal pattern analysis

### Interactive dashboard

- Built a Streamlit dashboard with per-shelter forecast views and capacity alerts
- Visualizations include occupancy trends, forecast confidence intervals, and alert indicators

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing shelter occupancy data
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
