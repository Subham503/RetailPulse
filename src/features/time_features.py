"""
Time Series Feature Engineering.

Builds daily + weekly sales aggregations,
adds lag features and rolling statistics for forecasting.

Usage:
    python -m src.features.time_features
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.config import PROCESSED_DIR, FEATURES_DIR
from src.utils.logging_config import setup_logger

INPUT_PATH        = PROCESSED_DIR / "cleaned_data.parquet"
DAILY_OUTPUT      = FEATURES_DIR  / "daily_sales.csv"
WEEKLY_OUTPUT     = FEATURES_DIR  / "weekly_sales.csv"
SKU_DAILY_OUTPUT  = FEATURES_DIR  / "sku_daily_sales.csv"


# ── Aggregations ───────────────────────────────────────────────────────────

def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily total sales.

    Returns DataFrame with columns: ds (date), y (revenue),
    orders (invoice count), items (qty sum).
    """
    daily = (
        df.groupby(df["InvoiceDate"].dt.normalize())
        .agg(
            y      = ("TotalPrice", "sum"),
            orders = ("Invoice",    "nunique"),
            items  = ("Quantity",   "sum"),
        )
        .reset_index()
        .rename(columns={"InvoiceDate": "ds"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    daily["y"] = daily["y"].round(2)
    logger.info(f"Daily sales: {len(daily)} days  |  {daily['ds'].min().date()} → {daily['ds'].max().date()}")
    return daily


def build_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to ISO weekly total sales."""
    df = df.copy()
    df["Week"] = df["InvoiceDate"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = (
        df.groupby("Week")
        .agg(
            y      = ("TotalPrice", "sum"),
            orders = ("Invoice",    "nunique"),
            items  = ("Quantity",   "sum"),
        )
        .reset_index()
        .rename(columns={"Week": "ds"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    weekly["y"] = weekly["y"].round(2)
    logger.info(f"Weekly sales: {len(weekly)} weeks")
    return weekly


def build_sku_daily_sales(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Daily sales per SKU (top N by total revenue).
    Used for product-level forecasting.
    """
    top_skus = (
        df.groupby("StockCode")["TotalPrice"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )

    sku_df = df[df["StockCode"].isin(top_skus)].copy()

    sku_daily = (
        sku_df.groupby(["StockCode", sku_df["InvoiceDate"].dt.normalize()])
        .agg(y=("TotalPrice", "sum"), qty=("Quantity", "sum"))
        .reset_index()
        .rename(columns={"InvoiceDate": "ds"})
        .sort_values(["StockCode", "ds"])
        .reset_index(drop=True)
    )

    logger.info(f"SKU daily sales: {len(top_skus)} SKUs × daily aggregation")
    return sku_daily


# ── Lag & Rolling Features ─────────────────────────────────────────────────

LAG_DAYS     = [1, 7, 14, 28]
ROLLING_WINS = [7, 14, 28]


def add_lag_features(df: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    """Add lag features for target column. df must be sorted by ds."""
    df = df.sort_values("ds").copy()
    for lag in LAG_DAYS:
        df[f"lag_{lag}d"] = df[target].shift(lag)
    logger.info(f"Lag features added: {[f'lag_{l}d' for l in LAG_DAYS]}")
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    """Add rolling mean, std, max for target column."""
    df = df.sort_values("ds").copy()
    for w in ROLLING_WINS:
        df[f"roll_mean_{w}d"] = df[target].shift(1).rolling(w, min_periods=1).mean().round(2)
        df[f"roll_std_{w}d"]  = df[target].shift(1).rolling(w, min_periods=1).std().round(2)
        df[f"roll_max_{w}d"]  = df[target].shift(1).rolling(w, min_periods=1).max().round(2)
    logger.info(f"Rolling features added: windows = {ROLLING_WINS}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/temporal features from ds column."""
    df = df.copy()
    df["day_of_week"]  = df["ds"].dt.dayofweek        # 0=Mon
    df["day_of_month"] = df["ds"].dt.day
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"]        = df["ds"].dt.month
    df["quarter"]      = df["ds"].dt.quarter
    df["year"]         = df["ds"].dt.year
    df["is_weekend"]   = (df["ds"].dt.dayofweek >= 5).astype(int)
    df["is_month_end"] = df["ds"].dt.is_month_end.astype(int)

    # Q4 flag (Oct-Dec) — retail peak season
    df["is_q4"] = (df["month"] >= 10).astype(int)

    logger.info("Calendar features added: day_of_week, month, quarter, is_weekend, is_q4, ...")
    return df


# ── Full Pipeline ──────────────────────────────────────────────────────────

def build_time_features(df: pd.DataFrame) -> dict:
    """Build all time series feature sets. Returns dict of DataFrames."""
    daily  = build_daily_sales(df)
    weekly = build_weekly_sales(df)
    sku    = build_sku_daily_sales(df)

    # Enrich daily with lag + rolling + calendar
    daily_enriched = daily.copy()
    daily_enriched = add_lag_features(daily_enriched)
    daily_enriched = add_rolling_features(daily_enriched)
    daily_enriched = add_calendar_features(daily_enriched)

    return {
        "daily":         daily,
        "daily_enriched": daily_enriched,
        "weekly":        weekly,
        "sku_daily":     sku,
    }


def run() -> dict:
    setup_logger()
    logger.info("=" * 50)
    logger.info("STEP 3b/6 — TIME SERIES FEATURES")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make clean` first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded: {len(df):,} rows")

    result = build_time_features(df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    result["daily"].to_csv(DAILY_OUTPUT,          index=False)
    result["weekly"].to_csv(WEEKLY_OUTPUT,         index=False)
    result["sku_daily"].to_csv(SKU_DAILY_OUTPUT,   index=False)

    logger.success(f"Saved → {DAILY_OUTPUT}")
    logger.success(f"Saved → {WEEKLY_OUTPUT}")
    logger.success(f"Saved → {SKU_DAILY_OUTPUT}")

    daily = result["daily"]
    logger.info("─" * 50)
    logger.info(f"  Daily avg revenue  : £{daily['y'].mean():>10,.2f}")
    logger.info(f"  Daily max revenue  : £{daily['y'].max():>10,.2f}")
    logger.info(f"  Daily avg orders   : {daily['orders'].mean():>10.1f}")
    logger.info(f"  Total days         : {len(daily):>10,}")
    logger.info("─" * 50)

    return result


if __name__ == "__main__":
    run()
