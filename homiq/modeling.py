"""Leakage-safe preprocessing and linear regression helpers."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price_inr"

BASELINE_NUMERIC_FEATURES = ["total_sqft", "bhk", "bath"]
IMPROVED_NUMERIC_FEATURES = BASELINE_NUMERIC_FEATURES + ["balcony"]
IMPROVED_CATEGORICAL_FEATURES = ["location", "area_type"]
IMPROVED_FEATURES = IMPROVED_NUMERIC_FEATURES + IMPROVED_CATEGORICAL_FEATURES


def build_pipeline(
    numeric_features: list[str],
    categorical_features: Optional[list[str]] = None,
) -> Pipeline:
    """Build a model with preprocessing fit only on each training fold."""
    categorical_features = categorical_features or []
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=15,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("regressor", LinearRegression()),
        ]
    )


def validate_feature_columns(frame: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(
            "The Bengaluru housing dataset is missing expected model features: "
            f"{missing}. Check that the selected dataset is Bengaluru House "
            "Price Data from Kaggle."
        )


def cross_validation_summary(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict[str, float]:
    """Return mean cross-validated RMSE, MAE, and R-squared on training data."""
    from sklearn.model_selection import cross_validate

    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    return {
        "rmse": float(-np.mean(scores["test_rmse"])),
        "mae": float(-np.mean(scores["test_mae"])),
        "r2": float(np.mean(scores["test_r2"])),
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics in the target's original INR units."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_model_artifact(
    estimator: Pipeline,
    features: list[str],
    categorical_features: list[str],
    metrics: dict[str, Any],
    frame: pd.DataFrame,
    analytics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Package the fitted model and dataset-driven UI metadata."""
    category_options = {}
    category_defaults = {}
    ui_categorical_features = [
        column for column in ["location", "area_type"] if column in frame
    ]
    for column in ui_categorical_features:
        values = sorted(frame[column].dropna().astype(str).unique().tolist())
        category_options[column] = values
        mode = frame[column].dropna().astype(str).mode()
        category_defaults[column] = (
            mode.iloc[0] if not mode.empty else (values[0] if values else "")
        )

    numeric_features = frame.select_dtypes(include="number").columns.tolist()
    return {
        "model": estimator,
        "features": features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "metrics": metrics,
        "numeric_defaults": {
            column: float(frame[column].median()) for column in numeric_features
        },
        "category_options": category_options,
        "category_defaults": category_defaults,
        "numeric_ranges": {
            column: [float(frame[column].min()), float(frame[column].max())]
            for column in numeric_features
        },
        "analytics": analytics or {},
        "data_source": "Bengaluru House Price Data (Kaggle, CC0)",
        "market": "Bengaluru, India",
        "price_unit": "INR",
    }
