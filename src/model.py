"""
Machine learning models for predicting emergency shelter occupancy.

Implements regression models (RandomForest, XGBoost, GradientBoosting) to
forecast shelter occupancy rates and overnight counts. Features include
temporal indicators, shelter metadata, capacity, and rolling/lag statistics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Default feature columns used for training
DEFAULT_FEATURES = [
    "day_of_week",
    "month",
    "year",
    "day_of_month",
    "capacity",
    "rolling_7d_occupancy",
    "rolling_30d_occupancy",
    "lag_1d_occupancy",
    "lag_7d_occupancy",
    "sheltertype_encoded",
]


def encode_categorical(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns for model consumption.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list of str, optional
        Columns to encode. Defaults to ["sheltertype"].

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        The dataframe with encoded columns appended (suffixed with
        ``_encoded``) and a dictionary mapping column names to their
        fitted LabelEncoder instances.
    """
    if columns is None:
        columns = ["sheltertype"]

    df = df.copy()
    encoders = {}

    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found in dataframe. Skipping.", col)
            continue
        le = LabelEncoder()
        encoded_col = f"{col}_encoded"
        df[encoded_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(
            "Encoded '%s' -> '%s' with %d classes.", col, encoded_col, len(le.classes_)
        )

    return df, encoders


def temporal_train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data respecting temporal order (no data leakage from the future).

    The most recent ``test_fraction`` of observations (by date) form the
    test set; the remainder form the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe sorted by date.
    test_fraction : float
        Proportion of data to hold out for testing.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Training and test dataframes.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    logger.info(
        "Temporal split: %d train rows (up to %s), %d test rows (from %s).",
        len(train),
        train["date"].max().date(),
        len(test),
        test["date"].min().date(),
    )
    return train, test


def prepare_features_target(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "occupancy_rate",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and target vector, dropping rows with NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list of str, optional
        Feature column names. Defaults to DEFAULT_FEATURES.
    target_col : str
        Name of the target column.

    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Feature matrix X and target vector y.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning("Missing feature columns (will be skipped): %s", missing)

    subset = df[available + [target_col]].dropna()
    X = subset[available]
    y = subset[target_col]
    return X, y


def get_model(name: str, **kwargs):
    """
    Instantiate a regression model by name.

    Parameters
    ----------
    name : str
        One of "random_forest", "xgboost", "gradient_boosting".
    **kwargs
        Additional keyword arguments forwarded to the model constructor.

    Returns
    -------
    sklearn-compatible estimator

    Raises
    ------
    ValueError
        If the model name is not recognized.
    """
    defaults = {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    }

    name_lower = name.lower().replace(" ", "_")

    if name_lower == "random_forest":
        params = {**defaults["random_forest"], **kwargs}
        return RandomForestRegressor(**params)
    elif name_lower == "xgboost":
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        params = {**defaults["xgboost"], **kwargs}
        return XGBRegressor(**params)
    elif name_lower == "gradient_boosting":
        params = {**defaults["gradient_boosting"], **kwargs}
        return GradientBoostingRegressor(**params)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: "
            "random_forest, xgboost, gradient_boosting"
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, and R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def get_feature_importance(
    model, feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a fitted model.

    Parameters
    ----------
    model : fitted estimator
        Must have a ``feature_importances_`` attribute.
    feature_names : list of str
        Names of the features used during training.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ``feature`` and ``importance``, sorted
        descending by importance.
    """
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def train_model(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    target_col: str = "occupancy_rate",
    feature_cols: Optional[List[str]] = None,
    test_fraction: float = 0.2,
    **model_kwargs,
) -> Dict:
    """
    Full training pipeline: encode, split, train, evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with rolling/lag features.
    model_name : str
        Name of the model to train.
    target_col : str
        Target column name.
    feature_cols : list of str, optional
        Feature columns. Defaults to DEFAULT_FEATURES.
    test_fraction : float
        Fraction of data held out for testing.
    **model_kwargs
        Extra arguments forwarded to the model constructor.

    Returns
    -------
    dict
        Dictionary containing the fitted model, encoders, metrics,
        feature importance, and predictions on the test set.
    """
    # Encode categoricals
    df_enc, encoders = encode_categorical(df, columns=["sheltertype"])

    # Temporal split
    train_df, test_df = temporal_train_test_split(df_enc, test_fraction=test_fraction)

    # Prepare features
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES

    X_train, y_train = prepare_features_target(train_df, feature_cols, target_col)
    X_test, y_test = prepare_features_target(test_df, feature_cols, target_col)

    logger.info(
        "Training %s on %d samples, testing on %d samples.",
        model_name,
        len(X_train),
        len(X_test),
    )

    # Train
    model = get_model(model_name, **model_kwargs)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = evaluate(y_train, y_pred_train)
    test_metrics = evaluate(y_test, y_pred_test)

    logger.info("Train metrics: %s", train_metrics)
    logger.info("Test metrics:  %s", test_metrics)

    # Feature importance
    fi = get_feature_importance(model, list(X_train.columns))

    return {
        "model": model,
        "model_name": model_name,
        "encoders": encoders,
        "feature_cols": list(X_train.columns),
        "target_col": target_col,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": fi,
        "y_test": y_test.values,
        "y_pred_test": y_pred_test,
        "y_train": y_train.values,
        "y_pred_train": y_pred_train,
    }


def save_model(result: Dict, filename: str = "shelter_model.joblib") -> Path:
    """
    Save a trained model and its metadata to disk.

    Parameters
    ----------
    result : dict
        Output of train_model().
    filename : str
        Filename for the saved artifact.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(result, filepath)
    logger.info("Model saved to %s", filepath)
    return filepath


def load_model(filename: str = "shelter_model.joblib") -> Dict:
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    filename : str
        Filename of the saved artifact.

    Returns
    -------
    dict
        The result dictionary produced by train_model().

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No saved model found at {filepath}")
    result = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return result


def train_all_models(
    df: pd.DataFrame,
    target_col: str = "occupancy_rate",
    test_fraction: float = 0.2,
) -> Dict[str, Dict]:
    """
    Train and compare all available models.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.
    target_col : str
        Target column.
    test_fraction : float
        Test split fraction.

    Returns
    -------
    dict
        Mapping of model name to its training result dictionary.
    """
    model_names = ["random_forest", "gradient_boosting"]
    if XGBRegressor is not None:
        model_names.append("xgboost")
    else:
        logger.warning("xgboost not installed; skipping XGBoost model.")

    results = {}
    for name in model_names:
        logger.info("Training %s...", name)
        try:
            result = train_model(
                df,
                model_name=name,
                target_col=target_col,
                test_fraction=test_fraction,
            )
            results[name] = result
        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data_loader import load_and_prepare

    df = load_and_prepare(use_cache=True)
    results = train_all_models(df)

    print("\n=== Model Comparison ===")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Train: {res['train_metrics']}")
        print(f"  Test:  {res['test_metrics']}")

    # Save the best model (by test R2)
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["R2"])
    print(f"\nBest model: {best_name}")
    save_model(results[best_name], "best_shelter_model.joblib")
