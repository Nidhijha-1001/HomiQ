"""Train, evaluate, and save HomiQ's Bengaluru house-price model."""

import json
from pathlib import Path
from typing import Union

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from homiq.currency import get_exchange_rate
from homiq.data import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET,
    describe_dataset,
    iqr_outlier_counts,
    load_indian_housing,
)
from homiq.modeling import (
    BASELINE_NUMERIC_FEATURES,
    IMPROVED_CATEGORICAL_FEATURES,
    IMPROVED_FEATURES,
    IMPROVED_NUMERIC_FEATURES,
    build_model_artifact,
    build_pipeline,
    cross_validation_summary,
    evaluate_regression,
    validate_feature_columns,
)

MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42


def _make_plots(
    frame: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_correlations: pd.Series,
    usd_per_inr: float,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    actual_usd = y_test.to_numpy() * usd_per_inr
    predicted_usd = y_pred * usd_per_inr
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(actual_usd, predicted_usd, alpha=0.65, edgecolor="none")
    lower = min(float(actual_usd.min()), float(predicted_usd.min()))
    upper = max(float(actual_usd.max()), float(predicted_usd.max()))
    axis.plot([lower, upper], [lower, upper], "--", color="crimson", label="Ideal")
    axis.set(
        title="Actual vs. predicted Bengaluru listing prices",
        xlabel="Actual listed price (USD)",
        ylabel="Predicted listed price (USD)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    correlation_columns = [
        TARGET,
        *[name for name in IMPROVED_NUMERIC_FEATURES if name in frame.columns],
    ]
    correlation = frame[correlation_columns].corr(numeric_only=True)
    correlation = correlation.rename(columns={TARGET: "Listed price (USD)"})
    correlation = correlation.rename(index={TARGET: "Listed price (USD)"})
    fig, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        ax=axis,
    )
    axis.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    residuals_usd = (y_test.to_numpy() - y_pred) * usd_per_inr
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(frame[TARGET] * usd_per_inr, kde=True, ax=axes[0])
    axes[0].set(title="Listed price distribution", xlabel="Listed price (USD)")
    axes[1].scatter(predicted_usd, residuals_usd, alpha=0.65, edgecolor="none")
    axes[1].axhline(0, color="crimson", linestyle="--")
    axes[1].set(
        title="Residuals vs. predicted price",
        xlabel="Predicted listed price (USD)",
        ylabel="Residual (USD)",
    )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "regression_diagnostics.png", dpi=160)
    plt.close(fig)

    top_correlations = feature_correlations.abs().sort_values(ascending=False).head(8)
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(
        x=feature_correlations.loc[top_correlations.index].values,
        y=top_correlations.index,
        color="#4c78a8",
        ax=axis,
    )
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set(
        title="Strongest numeric correlations with listed price",
        xlabel="Pearson correlation",
        ylabel="Feature",
    )
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_correlations.png", dpi=160)
    plt.close(fig)


def train_model(data_path: Union[Path, str] = DEFAULT_DATA_PATH) -> dict:
    frame = load_indian_housing(data_path)
    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        frame = frame.drop_duplicates().reset_index(drop=True)

    exchange_rate = get_exchange_rate()
    if exchange_rate.error:
        print(
            "Live exchange rate unavailable; static charts use the configured "
            f"fallback ({exchange_rate.usd_per_inr:.6f} USD per INR): "
            f"{exchange_rate.error}"
        )
    print("Bengaluru House Price Data exploratory summary")
    describe_dataset(frame, exchange_rate.usd_per_inr)
    print("\nIQR outlier counts (reported, not automatically removed):")
    print(iqr_outlier_counts(frame).head(15).to_string())

    validate_feature_columns(frame, IMPROVED_FEATURES)
    X = frame[IMPROVED_FEATURES].copy()
    y = frame[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    baseline = build_pipeline(BASELINE_NUMERIC_FEATURES)
    improved = build_pipeline(
        IMPROVED_NUMERIC_FEATURES, IMPROVED_CATEGORICAL_FEATURES
    )
    baseline_cv = cross_validation_summary(
        baseline, X_train[BASELINE_NUMERIC_FEATURES], y_train
    )
    improved_cv = cross_validation_summary(improved, X_train, y_train)
    selected_name = "improved" if improved_cv["r2"] > baseline_cv["r2"] else "baseline"
    selected_features = (
        IMPROVED_FEATURES if selected_name == "improved" else BASELINE_NUMERIC_FEATURES
    )
    selected_categories = (
        IMPROVED_CATEGORICAL_FEATURES if selected_name == "improved" else []
    )
    selected_X_train = X_train[selected_features]
    selected_X_test = X_test[selected_features]
    final_model = improved if selected_name == "improved" else baseline
    final_model.fit(selected_X_train, y_train)
    predictions = final_model.predict(selected_X_test)
    test_metrics = evaluate_regression(y_test, predictions)
    importance = permutation_importance(
        final_model,
        selected_X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    numeric_correlations = frame[IMPROVED_NUMERIC_FEATURES + [TARGET]].corr(
        numeric_only=True
    )[TARGET].drop(TARGET)
    market_data = frame.copy()
    test_data = X_test.copy()
    test_data["ActualPrice"] = y_test
    test_data["PredictedPrice"] = predictions
    analytics = {
        "market_rows": market_data.to_dict(orient="records"),
        "test_rows": test_data.to_dict(orient="records"),
        "price_values_inr": frame[TARGET].astype(float).tolist(),
        "permutation_importance": [
            {
                "feature": feature,
                "mae_increase": float(mean),
                "mae_increase_std": float(std),
            }
            for feature, mean, std in zip(
                selected_features,
                importance.importances_mean,
                importance.importances_std,
            )
        ],
    }

    reports = {
        "dataset": {
            "name": "Bengaluru House Price Data",
            "source": "Kaggle dataset 20710 by AmitabhaChakraborty",
            "source_url": "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data",
            "license": "CC0: Public Domain",
            "geographic_scope": "Bengaluru, India",
            "rows_after_target_cleanup": int(len(frame)),
            "duplicate_rows_removed": duplicate_count,
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "target_unit": "INR",
        },
        "cross_validation": {
            "folds": 5,
            "baseline": baseline_cv,
            "improved": improved_cv,
            "target_unit": "INR",
            "selection_metric": "mean five-fold R-squared on training data",
        },
        "selected_model": selected_name,
        "features": selected_features,
        "test": test_metrics,
        "target_unit": "INR",
        "display_exchange_rate": {
            "usd_per_inr": exchange_rate.usd_per_inr,
            "rate_date": exchange_rate.rate_date,
            "source": exchange_rate.source,
        },
        "top_numeric_correlations_with_listed_price": {
            key: float(value)
            for key, value in numeric_correlations.reindex(
                numeric_correlations.abs()
                .sort_values(ascending=False)
                .index[:8]
            ).items()
        },
    }
    _make_plots(
        frame,
        y_test,
        predictions,
        numeric_correlations,
        exchange_rate.usd_per_inr,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_model_artifact(
        final_model,
        selected_features,
        selected_categories,
        reports,
        X_train,
        analytics,
    )
    joblib.dump(artifact, MODEL_PATH)

    print("\nCross-validation comparison (USD display; R² is unitless):")
    display_cv = {
        name: {
            "rmse_usd": scores["rmse"] * exchange_rate.usd_per_inr,
            "mae_usd": scores["mae"] * exchange_rate.usd_per_inr,
            "r2": scores["r2"],
        }
        for name, scores in (
            ("baseline", baseline_cv),
            ("improved", improved_cv),
        )
    }
    print(json.dumps(display_cv, indent=2))
    print(f"\nSelected model: {selected_name}")
    print("Held-out test metrics (USD display; R² is unitless):")
    display_test_metrics = {
        "mse_usd_squared": test_metrics["mse"] * exchange_rate.usd_per_inr**2,
        "rmse_usd": test_metrics["rmse"] * exchange_rate.usd_per_inr,
        "mae_usd": test_metrics["mae"] * exchange_rate.usd_per_inr,
        "r2": test_metrics["r2"],
    }
    print(json.dumps(display_test_metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved reports and plots: {REPORTS_DIR}")
    return reports


if __name__ == "__main__":
    train_model()
