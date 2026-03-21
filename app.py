"""
Streamlit application for Calgary Emergency Shelter Occupancy Predictor.

Provides interactive dashboards for exploring shelter occupancy data,
analyzing individual shelters, forecasting future demand, and reviewing
model performance metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure the src package is importable
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import load_and_prepare, compute_shelter_summary
from src.model import (
    train_model,
    train_all_models,
    save_model,
    load_model,
    encode_categorical,
    prepare_features_target,
    DEFAULT_FEATURES,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Emergency Shelter Occupancy Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading shelter occupancy data...")
def get_data() -> pd.DataFrame:
    """Load and prepare the shelter occupancy dataset."""
    df = load_and_prepare(use_cache=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Computing shelter summaries...")
def get_shelter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-shelter summary statistics."""
    return compute_shelter_summary(df)


@st.cache_resource(show_spinner="Training models (this may take a moment)...")
def get_trained_models(_df: pd.DataFrame):
    """Train all models and cache the results."""
    return train_all_models(_df)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Occupancy Dashboard",
        "Shelter Analysis",
        "Demand Forecasting",
        "Model Performance",
        "About",
    ],
)

# Load data
try:
    df = get_data()
except Exception as exc:
    st.error(
        f"Failed to load data: {exc}\n\n"
        "Please ensure the dataset CSV is available in the data/ folder, "
        "or that you have an internet connection for the initial download."
    )
    st.stop()

# =========================================================================
# PAGE: Occupancy Dashboard
# =========================================================================
if page == "Occupancy Dashboard":
    st.title("Calgary Emergency Shelter Occupancy Dashboard")
    st.markdown(
        "An overview of shelter occupancy across Calgary, based on daily "
        "reporting data from the City of Calgary Open Data portal."
    )

    # --- KPI Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    total_shelters = df["shelter"].nunique()
    avg_occupancy = df["occupancy_rate"].mean()
    total_capacity = int(df.groupby("date")["capacity"].sum().mean())
    total_records = len(df)

    col1.metric("Total Shelters", f"{total_shelters}")
    col2.metric("Avg Occupancy Rate", f"{avg_occupancy:.1%}")
    col3.metric("Avg Daily Capacity", f"{total_capacity:,}")
    col4.metric("Total Records", f"{total_records:,}")

    st.markdown("---")

    # --- Occupancy Over Time ---
    st.subheader("Overall Occupancy Over Time")

    daily_avg = (
        df.groupby("date")
        .agg(avg_occupancy=("occupancy_rate", "mean"), total_overnight=("overnight", "sum"))
        .reset_index()
    )

    time_metric = st.radio(
        "Display metric", ["Average Occupancy Rate", "Total Overnight Count"], horizontal=True
    )
    if time_metric == "Average Occupancy Rate":
        fig = px.line(
            daily_avg, x="date", y="avg_occupancy",
            labels={"date": "Date", "avg_occupancy": "Avg Occupancy Rate"},
        )
        fig.update_yaxes(tickformat=".0%")
    else:
        fig = px.line(
            daily_avg, x="date", y="total_overnight",
            labels={"date": "Date", "total_overnight": "Total Overnight"},
        )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # --- Shelter Type Breakdown ---
    st.subheader("Occupancy by Shelter Type")

    type_summary = (
        df.groupby("sheltertype")
        .agg(
            avg_occupancy=("occupancy_rate", "mean"),
            avg_overnight=("overnight", "mean"),
            avg_capacity=("capacity", "mean"),
            count=("date", "count"),
        )
        .reset_index()
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = px.bar(
            type_summary.sort_values("avg_occupancy", ascending=False),
            x="sheltertype",
            y="avg_occupancy",
            color="sheltertype",
            labels={"sheltertype": "Shelter Type", "avg_occupancy": "Avg Occupancy Rate"},
            title="Average Occupancy Rate by Shelter Type",
        )
        fig_bar.update_yaxes(tickformat=".0%")
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_pie = px.pie(
            type_summary,
            values="count",
            names="sheltertype",
            title="Record Distribution by Shelter Type",
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)


# =========================================================================
# PAGE: Shelter Analysis
# =========================================================================
elif page == "Shelter Analysis":
    st.title("Individual Shelter Analysis")

    shelter_list = sorted(df["shelter"].dropna().unique())

    # --- Single Shelter View ---
    st.subheader("Single Shelter Occupancy Trend")
    selected_shelter = st.selectbox("Select a shelter", shelter_list)

    shelter_df = df[df["shelter"] == selected_shelter].sort_values("date")

    if shelter_df.empty:
        st.warning("No data found for the selected shelter.")
    else:
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Mean Occupancy", f"{shelter_df['occupancy_rate'].mean():.1%}")
        info_col2.metric("Mean Capacity", f"{shelter_df['capacity'].mean():.0f}")
        info_col3.metric("Data Points", f"{len(shelter_df):,}")

        fig_shelter = px.line(
            shelter_df, x="date", y="occupancy_rate",
            title=f"Occupancy Rate: {selected_shelter}",
            labels={"date": "Date", "occupancy_rate": "Occupancy Rate"},
        )
        fig_shelter.update_yaxes(tickformat=".0%")
        fig_shelter.add_hline(y=0.9, line_dash="dash", line_color="red",
                              annotation_text="90% threshold")
        fig_shelter.update_layout(height=400)
        st.plotly_chart(fig_shelter, use_container_width=True)

    st.markdown("---")

    # --- Capacity Utilization Heatmap ---
    st.subheader("Monthly Capacity Utilization Heatmap")

    heatmap_type = st.selectbox(
        "Group by", ["Shelter Type", "Individual Shelter (top 15)"]
    )

    if heatmap_type == "Shelter Type":
        heat_data = (
            df.groupby(["month", "sheltertype"])["occupancy_rate"]
            .mean()
            .reset_index()
            .pivot(index="sheltertype", columns="month", values="occupancy_rate")
        )
    else:
        top_shelters = df["shelter"].value_counts().head(15).index
        heat_data = (
            df[df["shelter"].isin(top_shelters)]
            .groupby(["month", "shelter"])["occupancy_rate"]
            .mean()
            .reset_index()
            .pivot(index="shelter", columns="month", values="occupancy_rate")
        )

    heat_data.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][c - 1]
        if 1 <= c <= 12 else str(c)
        for c in heat_data.columns
    ]

    fig_heat = px.imshow(
        heat_data,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels={"color": "Occupancy Rate"},
        title="Average Occupancy Rate by Month",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- Compare Shelters ---
    st.subheader("Compare Shelters Side by Side")
    compare_shelters = st.multiselect(
        "Select shelters to compare", shelter_list, default=shelter_list[:3]
    )

    if compare_shelters:
        compare_df = df[df["shelter"].isin(compare_shelters)]
        monthly_compare = (
            compare_df.groupby([pd.Grouper(key="date", freq="MS"), "shelter"])
            ["occupancy_rate"]
            .mean()
            .reset_index()
        )
        fig_compare = px.line(
            monthly_compare, x="date", y="occupancy_rate", color="shelter",
            title="Monthly Occupancy Comparison",
            labels={"date": "Date", "occupancy_rate": "Occupancy Rate"},
        )
        fig_compare.update_yaxes(tickformat=".0%")
        fig_compare.update_layout(height=450)
        st.plotly_chart(fig_compare, use_container_width=True)


# =========================================================================
# PAGE: Demand Forecasting
# =========================================================================
elif page == "Demand Forecasting":
    st.title("Shelter Demand Forecasting")
    st.markdown(
        "Predict future occupancy to support proactive shelter resource "
        "allocation and identify shelters approaching capacity."
    )

    # Train or load a model
    try:
        model_result = load_model("best_shelter_model.joblib")
        st.success("Loaded pre-trained model from disk.")
    except FileNotFoundError:
        st.info("No pre-trained model found. Training a Random Forest model...")
        model_result = train_model(df, model_name="random_forest")
        save_model(model_result, "best_shelter_model.joblib")
        st.success("Model trained and saved.")

    st.markdown("---")

    # Forecast horizon
    forecast_days = st.radio("Forecast horizon", [7, 30], horizontal=True)

    # Select shelter for forecast
    shelter_list = sorted(df["shelter"].dropna().unique())
    forecast_shelter = st.selectbox("Select shelter to forecast", shelter_list)

    shelter_data = df[df["shelter"] == forecast_shelter].sort_values("date")

    if len(shelter_data) < 30:
        st.warning("Insufficient historical data for this shelter.")
    else:
        # Build future date range
        last_date = shelter_data["date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Use the most recent row as a template for static features
        template = shelter_data.iloc[-1].copy()

        future_rows = []
        # Seed rolling values from the latest available data
        recent_7 = shelter_data["occupancy_rate"].tail(7).mean()
        recent_30 = shelter_data["occupancy_rate"].tail(30).mean()
        last_occ = shelter_data["occupancy_rate"].iloc[-1]
        lag7_series = shelter_data["occupancy_rate"].tail(7).tolist()

        for i, fdate in enumerate(future_dates):
            row = {
                "day_of_week": fdate.dayofweek,
                "month": fdate.month,
                "year": fdate.year,
                "day_of_month": fdate.day,
                "capacity": template["capacity"],
                "rolling_7d_occupancy": recent_7,
                "rolling_30d_occupancy": recent_30,
                "lag_1d_occupancy": last_occ,
                "lag_7d_occupancy": lag7_series[i % len(lag7_series)] if lag7_series else recent_7,
            }
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)

        # Encode shelter type
        if "sheltertype" in shelter_data.columns:
            stype = shelter_data["sheltertype"].iloc[-1]
            encoders = model_result.get("encoders", {})
            if "sheltertype" in encoders:
                le = encoders["sheltertype"]
                if stype in le.classes_:
                    future_df["sheltertype_encoded"] = le.transform([stype])[0]
                else:
                    future_df["sheltertype_encoded"] = 0
            else:
                future_df["sheltertype_encoded"] = 0

        # Align feature columns
        feature_cols = model_result.get("feature_cols", DEFAULT_FEATURES)
        for col in feature_cols:
            if col not in future_df.columns:
                future_df[col] = 0

        predictions = model_result["model"].predict(future_df[feature_cols])
        predictions = np.clip(predictions, 0, None)

        forecast_result = pd.DataFrame({
            "date": future_dates,
            "predicted_occupancy": predictions,
        })

        # Plot historical + forecast
        hist_tail = shelter_data.tail(90)[["date", "occupancy_rate"]].rename(
            columns={"occupancy_rate": "value"}
        )
        hist_tail["type"] = "Historical"

        fore = forecast_result.rename(columns={"predicted_occupancy": "value"})
        fore["type"] = "Forecast"

        combined = pd.concat([hist_tail, fore], ignore_index=True)

        fig_fc = px.line(
            combined, x="date", y="value", color="type",
            title=f"Occupancy Forecast: {forecast_shelter} ({forecast_days} days)",
            labels={"date": "Date", "value": "Occupancy Rate"},
            color_discrete_map={"Historical": "#636EFA", "Forecast": "#EF553B"},
        )
        fig_fc.update_yaxes(tickformat=".0%")
        fig_fc.add_hline(y=0.9, line_dash="dash", line_color="orange",
                         annotation_text="90% capacity threshold")
        fig_fc.update_layout(height=450)
        st.plotly_chart(fig_fc, use_container_width=True)

        # Capacity alerts
        st.subheader("Capacity Alerts")
        high_days = forecast_result[forecast_result["predicted_occupancy"] > 0.9]
        if len(high_days) > 0:
            st.error(
                f"**{len(high_days)} of {forecast_days} forecasted days exceed 90% occupancy!** "
                f"This shelter may need additional resources."
            )
            st.dataframe(
                high_days.style.format({"predicted_occupancy": "{:.1%}"}),
                use_container_width=True,
            )
        else:
            st.success(
                "No days in the forecast period exceed the 90% occupancy threshold."
            )


# =========================================================================
# PAGE: Model Performance
# =========================================================================
elif page == "Model Performance":
    st.title("Model Performance & Comparison")
    st.markdown(
        "Train multiple regression models and compare their predictive accuracy "
        "on shelter occupancy rates."
    )

    if st.button("Train / Retrain All Models"):
        st.cache_resource.clear()

    results = get_trained_models(df)

    # --- Comparison Table ---
    st.subheader("Model Comparison")
    comparison_rows = []
    for name, res in results.items():
        comparison_rows.append({
            "Model": name,
            "Train MAE": res["train_metrics"]["MAE"],
            "Train RMSE": res["train_metrics"]["RMSE"],
            "Train R2": res["train_metrics"]["R2"],
            "Test MAE": res["test_metrics"]["MAE"],
            "Test RMSE": res["test_metrics"]["RMSE"],
            "Test R2": res["test_metrics"]["R2"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df.style.highlight_max(
        subset=["Test R2"], color="#a8d08d"
    ).highlight_min(
        subset=["Test MAE", "Test RMSE"], color="#a8d08d"
    ), use_container_width=True)

    st.markdown("---")

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    selected_model_name = st.selectbox("Select model", list(results.keys()))
    fi = results[selected_model_name]["feature_importance"]

    fig_fi = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title=f"Feature Importance: {selected_model_name}",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # --- Actual vs Predicted Scatter ---
    st.subheader("Actual vs Predicted (Test Set)")
    y_test = results[selected_model_name]["y_test"]
    y_pred = results[selected_model_name]["y_pred_test"]

    scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted",
        title=f"Actual vs Predicted: {selected_model_name}",
        opacity=0.3,
    )
    # Perfect prediction line
    max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect Prediction",
            line={"dash": "dash", "color": "red"},
        )
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Residual Distribution ---
    st.subheader("Residual Distribution")
    residuals = y_test - y_pred
    fig_resid = px.histogram(
        x=residuals, nbins=60,
        title=f"Residuals: {selected_model_name}",
        labels={"x": "Residual (Actual - Predicted)", "count": "Frequency"},
    )
    fig_resid.update_layout(height=350)
    st.plotly_chart(fig_resid, use_container_width=True)


# =========================================================================
# PAGE: About
# =========================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Problem Statement

    Homelessness remains one of the most pressing social challenges facing
    Calgary. Emergency shelters serve as a critical safety net, providing
    overnight accommodation for individuals and families experiencing
    housing instability. However, shelter operators and municipal planners
    often struggle with **unpredictable demand fluctuations** driven by
    seasonal weather patterns, economic conditions, and policy changes.

    When shelters reach capacity, vulnerable populations are turned away,
    leading to dangerous exposure during Calgary's harsh winters. Conversely,
    over-allocating resources to underutilized facilities diverts funding
    from other essential services.

    **This project builds a predictive model to forecast daily shelter
    occupancy**, enabling proactive resource allocation, early capacity
    warnings, and data-driven planning for Calgary's emergency shelter system.

    ## Dataset

    - **Source**: [City of Calgary Open Data Portal](https://data.calgary.ca/)
    - **Dataset**: Emergency Shelters Daily Occupancy (ID: `7u2t-3wxf`)
    - **Records**: ~82,869 daily observations
    - **Columns**: date, year, month, city, shelter type, shelter name,
      organization, shelter, capacity, overnight count

    ## Methodology

    1. **Data Collection**: Fetched via Socrata Open Data API (SODA) with
       local caching for reproducibility.
    2. **Feature Engineering**: Temporal features (day-of-week, month,
       day-of-month), rolling averages (7-day, 30-day), lag features
       (1-day, 7-day), and shelter metadata encoding.
    3. **Modeling**: Three regression algorithms compared ---
       Random Forest, Gradient Boosting, and XGBoost --- using a temporal
       train/test split to prevent data leakage.
    4. **Evaluation**: Models assessed on MAE, RMSE, and R-squared using
       held-out future data.
    5. **Forecasting**: The best model generates multi-day-ahead occupancy
       predictions with capacity alerts at the 90% threshold.

    ## Features of This Application

    - **Occupancy Dashboard**: High-level KPIs and trends across all shelters.
    - **Shelter Analysis**: Deep dives into individual shelters with
      heatmaps and side-by-side comparisons.
    - **Demand Forecasting**: 7-day and 30-day occupancy predictions with
      capacity alert notifications.
    - **Model Performance**: Transparent comparison of model accuracy with
      feature importance and residual analysis.

    ## Technology Stack

    - **Python**: pandas, NumPy, scikit-learn, XGBoost
    - **Visualization**: Plotly
    - **Application**: Streamlit
    - **Data Access**: sodapy (Socrata Open Data API)

    ## How to Run

    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)

    st.markdown("---")
    st.caption(
        "Data provided by the City of Calgary Open Data Portal. "
        "This project is for educational and portfolio purposes."
    )
