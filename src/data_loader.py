"""
Data loader for Calgary Emergency Shelter Daily Occupancy dataset.

Fetches data from Calgary Open Data portal using the Socrata API (sodapy),
caches it locally, and provides preprocessing and feature engineering
utilities for shelter occupancy prediction.

Dataset: Emergency Shelters Daily Occupancy
Dataset ID: 7u2t-3wxf
Records: ~82,869
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from sodapy import Socrata
except ImportError:
    Socrata = None

logger = logging.getLogger(__name__)

# Constants
DATASET_ID = "7u2t-3wxf"
DOMAIN = "data.calgary.ca"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "shelter_occupancy_raw.csv"
PROCESSED_FILE = DATA_DIR / "shelter_occupancy_processed.csv"
EXPECTED_COLUMNS = [
    "date", "year", "month", "city", "sheltertype", "sheltername",
    "organization", "shelter", "capacity", "overnight",
]
TOTAL_RECORDS = 82869


def fetch_data(use_cache: bool = True, limit: int = TOTAL_RECORDS + 5000) -> pd.DataFrame:
    """
    Fetch Emergency Shelter Occupancy data from Calgary Open Data.

    Parameters
    ----------
    use_cache : bool
        If True, load from local CSV cache when available instead of
        making an API call. Defaults to True.
    limit : int
        Maximum number of records to fetch from the API.

    Returns
    -------
    pd.DataFrame
        Raw shelter occupancy data.

    Raises
    ------
    ImportError
        If sodapy is not installed and cache is unavailable.
    RuntimeError
        If data cannot be fetched or loaded.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        logger.info("Loading cached data from %s", CACHE_FILE)
        df = pd.read_csv(CACHE_FILE)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    if Socrata is None:
        raise ImportError(
            "sodapy is required to fetch data from the API. "
            "Install it with: pip install sodapy"
        )

    logger.info("Fetching data from Calgary Open Data (dataset %s)...", DATASET_ID)
    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(DATASET_ID, limit=limit)
        client.close()

        df = pd.DataFrame.from_records(results)
        logger.info("Fetched %d records from the API.", len(df))

        # Save to cache
        df.to_csv(CACHE_FILE, index=False)
        logger.info("Cached raw data to %s", CACHE_FILE)
    except Exception as exc:
        logger.error("Failed to fetch data from Socrata API: %s", exc)
        if CACHE_FILE.exists():
            logger.warning("Falling back to cached data.")
            return pd.read_csv(CACHE_FILE)
        raise RuntimeError(
            f"Failed to fetch data from Socrata API: {exc}"
        ) from exc

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw shelter occupancy data.

    Steps:
        1. Parse the date column to datetime.
        2. Convert capacity and overnight columns to numeric.
        3. Compute occupancy_rate = overnight / capacity.
        4. Extract temporal features: day_of_week, month, year, day_of_month.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from fetch_data().

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with additional columns.
    """
    df = df.copy()

    # Standardize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Convert numeric columns
    for col in ["capacity", "overnight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing capacity or overnight
    df = df.dropna(subset=["capacity", "overnight"])

    # Compute occupancy rate (guard against division by zero)
    df["occupancy_rate"] = np.where(
        df["capacity"] > 0,
        df["overnight"] / df["capacity"],
        np.nan,
    )

    # Extract temporal features
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_month"] = df["date"].dt.day

    # Sort by shelter and date for rolling calculations
    df = df.sort_values(["shelter", "date"]).reset_index(drop=True)

    logger.info(
        "Preprocessing complete. Shape: %s, Date range: %s to %s",
        df.shape,
        df["date"].min().date(),
        df["date"].max().date(),
    )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling average occupancy features per shelter.

    Adds:
        - rolling_7d_occupancy: 7-day rolling mean occupancy rate per shelter.
        - rolling_30d_occupancy: 30-day rolling mean occupancy rate per shelter.
        - lag_1d_occupancy: Previous day occupancy rate.
        - lag_7d_occupancy: Occupancy rate from 7 days ago.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with occupancy_rate column.

    Returns
    -------
    pd.DataFrame
        Dataframe augmented with rolling and lag features.
    """
    df = df.copy()
    df = df.sort_values(["shelter", "date"]).reset_index(drop=True)

    grouped = df.groupby("shelter")["occupancy_rate"]

    # Rolling averages (min_periods=1 to avoid too many NaNs at the start)
    df["rolling_7d_occupancy"] = grouped.transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["rolling_30d_occupancy"] = grouped.transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # Lag features
    df["lag_1d_occupancy"] = grouped.shift(1)
    df["lag_7d_occupancy"] = grouped.shift(7)

    logger.info("Added rolling and lag features.")
    return df


def compute_shelter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shelter-level summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    pd.DataFrame
        One row per shelter with summary stats including mean, median,
        std, min, max occupancy rate, total nights of data, and mean capacity.
    """
    summary = df.groupby(["shelter", "sheltertype", "organization"]).agg(
        mean_occupancy=("occupancy_rate", "mean"),
        median_occupancy=("occupancy_rate", "median"),
        std_occupancy=("occupancy_rate", "std"),
        min_occupancy=("occupancy_rate", "min"),
        max_occupancy=("occupancy_rate", "max"),
        mean_capacity=("capacity", "mean"),
        mean_overnight=("overnight", "mean"),
        total_nights=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()

    logger.info("Computed summary for %d shelters.", len(summary))
    return summary


def load_and_prepare(use_cache: bool = True) -> pd.DataFrame:
    """
    Full pipeline: fetch, preprocess, and add features.

    This is the primary entry point for downstream consumers.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached raw data.

    Returns
    -------
    pd.DataFrame
        Fully prepared dataframe ready for modeling or visualization.
    """
    if PROCESSED_FILE.exists() and use_cache:
        logger.info("Loading processed data from %s", PROCESSED_FILE)
        df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
        logger.info("Loaded %d processed records.", len(df))
        return df

    df = fetch_data(use_cache=use_cache)
    df = preprocess(df)
    df = add_rolling_features(df)

    # Save processed data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    logger.info("Saved processed data to %s", PROCESSED_FILE)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_and_prepare(use_cache=True)
    print(f"Loaded {len(data)} records with columns: {list(data.columns)}")
    print(data.head())
    print("\nShelter summary:")
    summary = compute_shelter_summary(data)
    print(summary.head(10))
