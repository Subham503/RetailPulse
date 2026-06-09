"""Shared data loading utilities for Streamlit dashboard."""

import pandas as pd
import streamlit as st
from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def _path(filename: str) -> Path:
    return PROCESSED_DIR / filename


# ── Cached loaders ─────────────────────────────────────────────────────────

@st.cache_data
def load_sales() -> pd.DataFrame | None:
    p = _path("cleaned_data.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_rfm() -> pd.DataFrame | None:
    # Prefer churn-enriched version
    for fname in ("rfm_with_churn.csv", "rfm_clustered.csv", "rfm.csv"):
        p = _path(fname)
        if p.exists():
            return pd.read_csv(p)
    return None


@st.cache_data
def load_forecast() -> pd.DataFrame | None:
    p = _path("forecast.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["ds"])
    return df


@st.cache_data
def load_inventory() -> pd.DataFrame | None:
    p = _path("inventory_metrics.csv")
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── UI helpers ─────────────────────────────────────────────────────────────

def not_found_msg(step: str) -> None:
    st.warning(
        f"⚠️ Data not found. Run `python -m {step}` first.",
        icon="⚠️",
    )


def metric_row(metrics: list[tuple]) -> None:
    """Render a row of st.metric cards. metrics = [(label, value, delta?), ...]"""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            if len(m) == 3:
                col.metric(m[0], m[1], m[2])
            else:
                col.metric(m[0], m[1])