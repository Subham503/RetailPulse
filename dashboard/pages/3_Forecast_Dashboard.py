"""Page 3 — Forecast Dashboard (Production Safe Version)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.utils import load_forecast, not_found_msg


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Forecast Dashboard")
st.divider()


# =============================
# LOAD DATA
# =============================
forecast = load_forecast()

if forecast is None or forecast.empty:
    not_found_msg("src.models.forecasting")
    st.stop()

forecast = forecast.copy()


# =============================
# HARD TYPE FIX (IMPORTANT)
# =============================
forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
forecast = forecast.dropna(subset=["ds"]).sort_values("ds")


# Split safely
actual = forecast[forecast["type"] == "actual"].copy()
pred   = forecast[forecast["type"] == "forecast"].copy()


# Convert numeric safely
num_cols = ["yhat", "yhat_lower", "yhat_upper"]
for col in num_cols:
    if col in pred.columns:
        pred[col] = pd.to_numeric(pred[col], errors="coerce")

if "y_actual" in actual.columns:
    actual["y_actual"] = pd.to_numeric(actual["y_actual"], errors="coerce")


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("Options")
    show_ci  = st.checkbox("Show confidence interval", value=True)
    show_tbl = st.checkbox("Show forecast table", value=False)
    horizon  = st.slider("Horizon (days)", 7, 90, 30)


pred_t = pred.head(horizon).copy().reset_index(drop=True)


# =============================
# SAFE METRICS
# =============================
avg_forecast = float(pred_t["yhat"].mean()) if "yhat" in pred_t else 0.0
peak_forecast = float(pred_t["yhat"].max()) if "yhat" in pred_t else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical Days", f"{len(actual):,}")
c2.metric("Forecast Horizon", f"{horizon} days")
c3.metric("Avg Forecast/Day", f"£{avg_forecast:,.0f}")
c4.metric("Peak Forecast", f"£{peak_forecast:,.0f}")

st.divider()


# =============================
# MAIN PLOT
# =============================
st.subheader("Demand Forecast — Historical + Predicted")

fig = go.Figure()


# -----------------------------
# ACTUAL LINE
# -----------------------------
if not actual.empty:
    fig.add_trace(go.Scatter(
        x=actual["ds"],
        y=actual["y_actual"],
        name="Actual",
        line=dict(color="#4F46E5", width=1.5)
    ))


# -----------------------------
# CONFIDENCE INTERVAL
# -----------------------------
if (
    show_ci
    and "yhat_lower" in pred_t.columns
    and "yhat_upper" in pred_t.columns
):
    xs = list(pred_t["ds"]) + list(pred_t["ds"][::-1])
    ys = list(pred_t["yhat_upper"]) + list(pred_t["yhat_lower"][::-1])

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        fill="toself",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI"
    ))


# -----------------------------
# FORECAST LINE
# -----------------------------
if "yhat" in pred_t.columns:
    fig.add_trace(go.Scatter(
        x=pred_t["ds"],
        y=pred_t["yhat"],
        name="Forecast",
        line=dict(color="#EF4444", width=2, dash="dash")
    ))


# -----------------------------
# VERTICAL LINE (FINAL FIX - NO CRASH VERSION)
# -----------------------------
if not actual.empty:
    fig.add_shape(
        type="line",
        x0=actual["ds"].max(),
        x1=actual["ds"].max(),
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", dash="dot")
    )

    fig.add_annotation(
        x=actual["ds"].max(),
        y=1,
        yref="paper",
        text="Forecast start",
        showarrow=False,
        font=dict(color="gray")
    )


fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Revenue (£)",
    hovermode="x unified",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# =============================
# BAR CHART
# =============================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Next {horizon} Days — Daily Forecast")

    if "yhat" in pred_t.columns:
        st.plotly_chart(
            px.bar(
                pred_t,
                x="ds",
                y="yhat",
                color="yhat",
                color_continuous_scale="Reds",
                labels={"ds": "Date", "yhat": "Forecast (£)"}
            ),
            use_container_width=True
        )


# =============================
# WEEKLY CHART
# =============================
with col2:
    st.subheader("Historical Weekly Revenue")

    if not actual.empty and "y_actual" in actual.columns:
        wk = actual.copy()
        wk["Week"] = wk["ds"].dt.to_period("W").apply(lambda x: x.start_time)

        wk = wk.groupby("Week")["y_actual"].sum().reset_index()

        st.plotly_chart(
            px.area(
                wk,
                x="Week",
                y="y_actual",
                labels={"y_actual": "Weekly Revenue (£)"}
            ),
            use_container_width=True
        )


# =============================
# TABLE
# =============================
if show_tbl and not pred_t.empty:
    st.subheader(f"Forecast Table — Next {horizon} Days")

    tbl = pred_t[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    tbl.columns = ["Date", "Forecast (£)", "Lower CI", "Upper CI"]

    for c in ["Forecast (£)", "Lower CI", "Upper CI"]:
        tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
        tbl[c] = tbl[c].apply(lambda x: f"£{x:,.2f}" if pd.notnull(x) else "N/A")

    st.dataframe(tbl.reset_index(drop=True), use_container_width=True)