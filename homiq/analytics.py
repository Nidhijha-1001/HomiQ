"""Plotly charts for Bengaluru market data and held-out model diagnostics."""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from homiq.data import TARGET

PALETTE = ["#4263eb", "#45c7d8", "#8ba1ff", "#152b50"]


def _polish(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#263753", size=11),
        title=dict(font=dict(size=15, color="#17243d"), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#101c36", font_color="white"),
    )
    figure.update_xaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#edf1f6", zeroline=False)
    return figure


def actual_vs_predicted_chart(
    frame: pd.DataFrame, usd_per_inr: float = 1.0
) -> go.Figure:
    display = frame.copy()
    display["ActualPriceUSD"] = display["ActualPrice"] * usd_per_inr
    display["PredictedPriceUSD"] = display["PredictedPrice"] * usd_per_inr
    hover_fields = [
        name
        for name in ["location", "area_type", "bhk", "total_sqft"]
        if name in display
    ]
    figure = px.scatter(
        display,
        x="ActualPriceUSD",
        y="PredictedPriceUSD",
        hover_data=hover_fields,
        color_discrete_sequence=PALETTE,
        opacity=0.72,
        title="Actual vs. predicted listed prices",
        labels={
            "ActualPriceUSD": "Actual price (USD)",
            "PredictedPriceUSD": "Predicted price (USD)",
        },
    )
    lower = min(display["ActualPriceUSD"].min(), display["PredictedPriceUSD"].min())
    upper = max(display["ActualPriceUSD"].max(), display["PredictedPriceUSD"].max())
    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Ideal",
            line=dict(color="#98a5b8", dash="dash"),
            hoverinfo="skip",
        )
    )
    figure.update_layout(legend=dict(title=None))
    return _polish(figure)


def price_distribution_chart(
    prices_inr: pd.Series,
    usd_per_inr: float = 1.0,
    predicted_price_inr: Optional[float] = None,
) -> go.Figure:
    prices_usd = prices_inr.astype(float) * usd_per_inr
    figure = px.histogram(
        x=prices_usd,
        nbins=35,
        title="Listed price distribution",
        labels={"x": "Recorded listed price (USD)", "y": "Listings"},
        color_discrete_sequence=[PALETTE[0]],
        opacity=0.86,
    )
    if predicted_price_inr is not None:
        figure.add_vline(
            x=predicted_price_inr * usd_per_inr,
            line_color=PALETTE[1],
            line_width=3,
            annotation_text="Your estimate",
            annotation_position="top right",
        )
    return _polish(figure, height=320)


def correlation_heatmap(frame: pd.DataFrame, features: list[str]) -> go.Figure:
    columns = [column for column in features if column in frame]
    correlation = frame[columns].corr(numeric_only=True)
    display_labels = [
        "Listed price (USD)" if column == TARGET else column.replace("_", " ").title()
        for column in correlation.columns
    ]
    figure = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(),
            x=display_labels,
            y=display_labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, "#4263eb"], [0.5, "#f7f9fd"], [1, "#45c7d8"]],
            colorbar=dict(title="r"),
            hovertemplate="%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title="Feature correlation", xaxis=dict(tickangle=-35))
    return _polish(figure, height=440)


def feature_importance_chart(
    frame: pd.DataFrame, usd_per_inr: float = 1.0
) -> go.Figure:
    ranked = frame.sort_values("mae_increase", ascending=True).tail(12).copy()
    ranked["mae_increase_usd"] = ranked["mae_increase"] * usd_per_inr
    ranked["mae_increase_std_usd"] = ranked["mae_increase_std"] * usd_per_inr
    figure = go.Figure(
        go.Bar(
            x=ranked["mae_increase_usd"],
            y=ranked["feature"].str.replace(r"(?<!^)(?=[A-Z])", " ", regex=True),
            orientation="h",
            error_x=dict(
                type="data", array=ranked["mae_increase_std_usd"], visible=True
            ),
            marker_color=[
                PALETTE[0] if value >= 0 else "#ef8354"
                for value in ranked["mae_increase_usd"]
            ],
            hovertemplate=(
                "%{y}<br>Mean absolute error increase: $%{x:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Permutation importance on held-out listings",
        xaxis_title="Increase in MAE when shuffled (USD)",
        yaxis_title=None,
        showlegend=False,
    )
    return _polish(figure, height=410)


def filter_market_data(
    frame: pd.DataFrame,
    location: str,
    area_type: str,
    bhk_range: tuple[float, float],
) -> pd.DataFrame:
    filtered = frame
    if location != "All locations" and "location" in filtered:
        filtered = filtered[filtered["location"] == location]
    if area_type != "All area types" and "area_type" in filtered:
        filtered = filtered[filtered["area_type"] == area_type]
    if "bhk" in filtered:
        filtered = filtered[filtered["bhk"].between(*bhk_range)]
    return filtered.copy()
