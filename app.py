"""Premium Streamlit experience for HomiQ's Bengaluru house-price model."""

from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from homiq.analytics import (
    actual_vs_predicted_chart,
    correlation_heatmap,
    feature_importance_chart,
    filter_market_data,
    price_distribution_chart,
)
from homiq.currency import ExchangeRate, format_usd, get_exchange_rate, inr_to_usd
from homiq.data import PROJECT_ROOT, TARGET
from homiq.train import train_model
from homiq.ui import apply_theme, render_brand, render_stat

MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.joblib"
INTEGER_FEATURES = {"bhk", "bath", "balcony"}
NAV_ITEMS = [
    ("Overview", "overview"),
    ("Price estimator", "estimate"),
    ("Market insights", "insights"),
]
USD_NOTE = (
    "Prices displayed in USD. Converted from INR using the latest available "
    "exchange rate."
)

st.set_page_config(
    page_title="HomiQ | Bengaluru property intelligence",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_model()

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception:
        train_model()
        artifact = joblib.load(MODEL_PATH)

    invalid = (
        artifact.get("market") != "Bengaluru, India"
        or artifact.get("target") != TARGET
        or not artifact.get("analytics", {}).get("market_rows")
    )
    if invalid:
        try:
            train_model()
            artifact = joblib.load(MODEL_PATH)
        except Exception as exc:  # pragma: no cover - surfaced in app UI
            raise ValueError(
                "The saved model is not the current Bengaluru/INR model and an "
                "automatic rebuild failed. Run `python train.py` from the project "
                "root to replace the stale artifact."
            ) from exc

    if (
        artifact.get("market") != "Bengaluru, India"
        or artifact.get("target") != TARGET
        or not artifact.get("analytics", {}).get("market_rows")
    ):
        raise ValueError(
            "The saved model is not the current Bengaluru/INR model. Run "
            "`python train.py` to replace the older artifact."
        )
    return artifact


@st.cache_data(ttl=3600)
def load_exchange_rate() -> ExchangeRate:
    return get_exchange_rate()


def _navigate(page: str) -> None:
    st.session_state["current_page"] = page


def _clear_prediction() -> None:
    for key in ("last_prediction", "last_prediction_signature", "last_prediction_inputs"):
        st.session_state.pop(key, None)


def _format_money(value_inr: float, exchange_rate: ExchangeRate) -> str:
    return format_usd(inr_to_usd(value_inr, exchange_rate.usd_per_inr))


def _input_signature(values: dict[str, Any]) -> tuple:
    return tuple(
        (key, "<missing>" if pd.isna(value) else value)
        for key, value in sorted(values.items())
    )


def _slider(artifact: dict[str, Any], feature: str, label: str) -> float:
    default = artifact["numeric_defaults"][feature]
    observed_low, observed_high = artifact["numeric_ranges"][feature]
    low, high = float(observed_low), float(observed_high)

    if feature in INTEGER_FEATURES:
        min_value = int(np.floor(low))
        max_value = int(np.ceil(high))
        if max_value <= min_value:
            max_value = min_value + 1
        value = int(np.clip(round(default), min_value, max_value))
        selected = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            help=f"Observed training range: {min_value:,}–{max_value:,}.",
            key=f"input_{feature}",
        )
        return float(selected)

    step = 50.0
    low = float(np.floor(low / step) * step)
    high = float(np.ceil(high / step) * step)
    if high <= low:
        high = low + step
    value = low + round((float(default) - low) / step) * step
    value = float(np.clip(value, low, high))
    selected = st.slider(
        label,
        min_value=low,
        max_value=high,
        value=value,
        step=step,
        help=(
            f"Observed training range: {observed_low:,.0f}–"
            f"{observed_high:,.0f} sq. ft."
        ),
        key=f"input_{feature}",
    )
    return float(selected)


def _category_input(
    artifact: dict[str, Any], feature: str, label: str
) -> str:
    options = artifact["category_options"][feature]
    selected = st.selectbox(
        label,
        options,
        index=options.index(artifact["category_defaults"][feature]),
        help="Type to search values recorded in the Bengaluru training data.",
        key=f"input_{feature}",
    )
    return selected


def _render_navigation() -> None:
    current = st.session_state.get("current_page", "overview")
    columns = st.columns([1.5, 1, 1, 1.15])
    with columns[0]:
        render_brand()
    for column, (label, page) in zip(columns[1:], NAV_ITEMS):
        with column:
            st.button(
                label,
                key=f"nav_{page}",
                type="primary" if current == page else "secondary",
                use_container_width=True,
                on_click=_navigate,
                args=(page,),
            )
    st.markdown(
        '<div style="height:1px;background:#e8edf5;margin:2px 0 16px"></div>',
        unsafe_allow_html=True,
    )


def _render_currency_status(exchange_rate: ExchangeRate) -> None:
    st.caption(USD_NOTE)
    st.caption(
        f"Rate: 1 INR = ${exchange_rate.usd_per_inr:.5f} · "
        f"{exchange_rate.source} · {exchange_rate.rate_date}"
    )
    if exchange_rate.error:
        st.warning(
            "The live exchange-rate service is unavailable. Prices use the "
            "configured HOMIQ_USD_PER_INR fallback."
        )


def _hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-copy">
            <div class="eyebrow">INDIA PROPERTY INTELLIGENCE</div>
            <h1>Discover What<br/>Your Home Is Worth.</h1>
            <p>Explore a data-led estimate and the listing details shaping home
            values across Bengaluru.</p>
          </div>
          <div class="hero-art" aria-hidden="true">
            <svg viewBox="0 0 540 330" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="368" cy="150" r="115" fill="#45C7D8" fill-opacity=".08"/>
              <circle cx="368" cy="150" r="84" stroke="#90E5EB" stroke-opacity=".22"/>
              <g class="hero-glow">
                <path d="M226 158 348 65l125 93v116H226V158Z" fill="#F8FBFF"/>
                <path d="m205 161 143-111 145 111" stroke="#69D4E1" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M253 151h187v123H253z" fill="#E9F0FA"/>
                <path d="M322 195h48v79h-48z" fill="#4263EB"/>
                <path d="M276 173h29v32h-29zm104 0h29v32h-29z" fill="#77DDE5"/>
                <path d="M288 174v30m-11-15h29m84-15v30m-11-15h29" stroke="#F8FBFF" stroke-width="3"/>
                <path d="M220 273h253" stroke="#87DCE4" stroke-width="5" stroke-linecap="round"/>
              </g>
            </svg>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    first, second = st.columns([1, 1])
    with first:
        st.button(
            "Estimate your home value  →",
            key="hero_estimate",
            type="primary",
            on_click=_navigate,
            args=("estimate",),
        )
    with second:
        st.button(
            "Explore market insights",
            key="hero_insights",
            type="secondary",
            on_click=_navigate,
            args=("insights",),
        )


def _render_overview(
    artifact: dict[str, Any], exchange_rate: ExchangeRate
) -> None:
    metrics = artifact["metrics"]["test"]
    market_rows = artifact["analytics"]["market_rows"]
    _hero()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    first, second, third, fourth = st.columns(4)
    with first:
        render_stat(
            "LISTINGS ANALYZED",
            f"{len(market_rows):,}",
            "Bengaluru housing records",
        )
    with second:
        render_stat("HELD-OUT R²", f"{metrics['r2']:.3f}", "Measured on 20% test split")
    with third:
        render_stat("MODEL FEATURES", f"{len(artifact['features'])}", "Used by selected model")
    with fourth:
        render_stat(
            "TYPICAL ERROR",
            _format_money(metrics["mae"], exchange_rate),
            "Test-set mean absolute error",
        )

    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    st.subheader("A clearer picture of home value")
    st.markdown(
        '<p class="section-kicker">Explore listing data, receive a transparent '
        'estimate, and see how the model performs on records it did not train on.</p>',
        unsafe_allow_html=True,
    )
    feature_columns = st.columns(3)
    features = [
        ("⌕", "A place that feels specific", "Search Bengaluru localities found in the training data."),
        ("↗", "A transparent estimate", "Compare a model estimate with recorded local listing prices."),
        ("▦", "Insights behind the number", "Explore held-out predictions, correlations, and model metrics."),
    ]
    for column, (icon, title, copy) in zip(feature_columns, features):
        with column:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
                f"<h3>{title}</h3><p>{copy}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-strip"><strong>Market scope.</strong> The model uses '
        'historical Bengaluru listings only. It does not cover Delhi NCR, Mumbai, '
        'or other Indian cities, and it is not a current-market appraisal.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Displayed test metrics are converted from the model's original INR "
        "evaluation values using the exchange rate shown above."
    )


def _render_estimator(
    artifact: dict[str, Any], exchange_rate: ExchangeRate
) -> None:
    features = artifact["features"]
    values: dict[str, Any] = {}

    st.markdown('<div class="eyebrow">PERSONALIZED ESTIMATE</div>', unsafe_allow_html=True)
    st.title("Tell us about the home")
    st.markdown(
        '<p class="section-kicker">Start with the essentials. Your property '
        'summary updates as you adjust each detail.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        values["location"] = _category_input(artifact, "location", "Bengaluru locality")
        values["total_sqft"] = _slider(
            artifact, "total_sqft", "Property area (sq. ft.)"
        )
        values["bhk"] = _slider(artifact, "bhk", "Bedrooms (BHK)")
    with right:
        values["area_type"] = _category_input(artifact, "area_type", "Area type")
        values["bath"] = _slider(artifact, "bath", "Bathrooms")
        values["balcony"] = _slider(artifact, "balcony", "Balconies")

    if "location" not in features:
        st.info(
            "The selected model did not use locality as a predictor. The locality "
            "is shown for your summary and market comparison only."
        )
    if "area_type" not in features:
        st.info(
            "The selected model did not use area type as a predictor; it is shown "
            "for your property summary only."
        )

    signature = _input_signature(values)
    summary = st.columns(4)
    summary_items = [
        ("LOCALITY", values["location"]),
        ("AREA", f"{values['total_sqft']:,.0f} sq. ft."),
        ("BEDROOMS", f"{values['bhk']:.0f} BHK"),
        ("BATHROOMS", f"{values['bath']:.0f}"),
    ]
    for column, (label, value) in zip(summary, summary_items):
        with column:
            st.markdown(
                f'<div class="summary-card"><div class="summary-label">{label}</div>'
                f'<div class="summary-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button(
        "Predict house price",
        key="predict_price",
        type="primary",
        use_container_width=True,
    ):
        model_row = pd.DataFrame(
            [{feature: values[feature] for feature in features}],
            columns=features,
        )
        with st.spinner("Estimating from the trained model…"):
            prediction = float(artifact["model"].predict(model_row)[0])
        if not np.isfinite(prediction) or prediction <= 0:
            st.error(
                "The model did not return a valid positive estimate for these "
                "details. Try another property combination."
            )
            _clear_prediction()
        else:
            st.session_state["last_prediction"] = prediction
            st.session_state["last_prediction_signature"] = signature
            st.session_state["last_prediction_inputs"] = dict(values)
            st.toast("Your estimate is ready.", icon="🏠")

    saved_signature = st.session_state.get("last_prediction_signature")
    saved_inputs = st.session_state.get("last_prediction_inputs")
    saved_price = st.session_state.get("last_prediction")
    if saved_price is None:
        st.markdown(
            '<div class="empty-card">Your estimate will appear here after you '
            'predict a home.</div>',
            unsafe_allow_html=True,
        )
        return
    if signature != saved_signature:
        st.info("Your property details changed. Predict again to refresh the estimate.")
        return

    estimated_usd = inr_to_usd(saved_price, exchange_rate.usd_per_inr)
    area = float(saved_inputs["total_sqft"])
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-card"><div class="result-label">MODEL ESTIMATE · '
        f'{saved_inputs["location"]} · {area:,.0f} sq. ft.</div>'
        f'<div class="result-price">{format_usd(estimated_usd)}</div>'
        f'<div class="result-note">Historical Bengaluru listing estimate · '
        "not a formal appraisal</div></div>",
        unsafe_allow_html=True,
    )

    price_col, median_col = st.columns(2)
    with price_col:
        st.metric("Estimated price per sq. ft.", format_usd(estimated_usd / area))
    market = pd.DataFrame(artifact["analytics"]["market_rows"])
    local_market = market[market["location"] == saved_inputs["location"]]
    if local_market.empty:
        local_market = market
        comparison_label = "Bengaluru median listing price"
    else:
        comparison_label = f"{saved_inputs['location']} median listing price"
    with median_col:
        st.metric(
            comparison_label,
            _format_money(float(local_market[TARGET].median()), exchange_rate),
        )

    st.plotly_chart(
        price_distribution_chart(
            local_market[TARGET],
            usd_per_inr=exchange_rate.usd_per_inr,
            predicted_price_inr=float(saved_price),
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "The chart shows observed listing prices in the selected locality; the "
        "estimate marker is a model prediction, not a confidence interval. HomiQ "
        "does not report an uncertainty range because one has not been calibrated."
    )
    st.button(
        "Predict another property",
        key="predict_another",
        type="secondary",
        on_click=_clear_prediction,
    )


def _render_insights(
    artifact: dict[str, Any], exchange_rate: ExchangeRate
) -> None:
    st.markdown('<div class="eyebrow">DATA, NOT GUESSWORK</div>', unsafe_allow_html=True)
    st.title("A closer look at Bengaluru listings")
    st.markdown(
        '<p class="section-kicker">Filter recorded properties and inspect the '
        'trained model’s held-out predictions and evaluation metrics.</p>',
        unsafe_allow_html=True,
    )

    market = pd.DataFrame(artifact["analytics"]["market_rows"])
    test = pd.DataFrame(artifact["analytics"]["test_rows"])
    location_options = ["All locations"] + sorted(
        market["location"].dropna().astype(str).unique().tolist()
    )
    area_type_options = ["All area types"] + sorted(
        market["area_type"].dropna().astype(str).unique().tolist()
    )
    bhk_min = int(np.floor(market["bhk"].min()))
    bhk_max = int(np.ceil(market["bhk"].max()))
    filters = st.columns([1.4, 1.25, 1.25])
    with filters[0]:
        location = st.selectbox("Locality", location_options)
    with filters[1]:
        area_type = st.selectbox("Area type", area_type_options)
    with filters[2]:
        bhk_range = st.slider(
            "Bedrooms (BHK)",
            min_value=bhk_min,
            max_value=max(bhk_max, bhk_min + 1),
            value=(bhk_min, bhk_max),
            help="Filter the recorded bedrooms/BHK count.",
        )

    filtered_market = filter_market_data(market, location, area_type, bhk_range)
    filtered_test = filter_market_data(test, location, area_type, bhk_range)
    st.caption(
        f"Showing {len(filtered_market):,} of {len(market):,} listings · "
        f"{len(filtered_test):,} matching held-out test records."
    )
    metrics = artifact["metrics"]["test"]
    rate = exchange_rate.usd_per_inr
    metric_data = [
        ("TEST R²", f"{metrics['r2']:.3f}"),
        ("MEAN ABSOLUTE ERROR", format_usd(metrics["mae"] * rate)),
        ("ROOT MEAN SQUARED ERROR", format_usd(metrics["rmse"] * rate)),
        ("MEAN SQUARED ERROR (USD²)", format_usd(metrics["mse"] * rate * rate)),
    ]
    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, metric_data):
        with column:
            st.metric(label, value)
    st.caption(
        "R² is unitless. Monetary metrics are converted from the model's INR "
        "evaluation values; filters only change the visualized sample."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Model predictions")
        if filtered_test.empty:
            st.info("No held-out listings match these filters. Broaden the selection.")
        else:
            st.plotly_chart(
                actual_vs_predicted_chart(filtered_test, usd_per_inr=rate),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right:
        st.markdown("#### Recorded listing prices")
        if filtered_market.empty:
            st.info("No listings match these filters. Broaden the selection.")
        else:
            st.plotly_chart(
                price_distribution_chart(filtered_market[TARGET], usd_per_inr=rate),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Numeric relationships")
        if len(filtered_market) < 2:
            st.info("Select more listings to calculate feature correlations.")
        else:
            numeric_features = [
                feature
                for feature in artifact["features"]
                if feature not in artifact["categorical_features"]
            ]
            st.plotly_chart(
                correlation_heatmap(
                    filtered_market,
                    list(dict.fromkeys(numeric_features + [TARGET])),
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right:
        st.markdown("#### What moves the model?")
        st.plotly_chart(
            feature_importance_chart(
                pd.DataFrame(artifact["analytics"]["permutation_importance"]),
                usd_per_inr=rate,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    st.caption(
        "Permutation importance is the measured change in held-out MAE after "
        "randomly shuffling one feature (five repeats). It describes this fitted "
        "model and sample; it is not causal and was not used to select the model."
    )


def main() -> None:
    apply_theme()
    _render_navigation()
    exchange_rate = load_exchange_rate()
    _render_currency_status(exchange_rate)
    artifact = load_model_artifact()
    page = st.session_state.get("current_page", "overview")

    if page == "estimate":
        _render_estimator(artifact, exchange_rate)
    elif page == "insights":
        _render_insights(artifact, exchange_rate)
    else:
        _render_overview(artifact, exchange_rate)

    st.markdown(
        '<div class="footer-note">HomiQ · Bengaluru, India · Historical listing '
        "data for learning and exploration</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
