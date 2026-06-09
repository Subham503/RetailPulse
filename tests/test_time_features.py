"""Unit tests for src/features/time_features.py"""

import pandas as pd
import numpy as np
import pytest
from src.features.time_features import (
    build_daily_sales,
    build_weekly_sales,
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
)


@pytest.fixture
def clean_df():
    dates = pd.date_range("2011-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "InvoiceDate": dates.repeat(3),
        "Invoice":     [f"I{i}" for i in range(90)],
        "TotalPrice":  np.random.uniform(10, 200, 90).round(2),
        "Quantity":    np.random.randint(1, 10, 90),
        "CustomerID":  ["C1", "C2", "C3"] * 30,
        "StockCode":   ["SKU1", "SKU2", "SKU3"] * 30,
        "Country":     ["UK"] * 90,
    })


def test_build_daily_sales_shape(clean_df):
    daily = build_daily_sales(clean_df)
    assert len(daily) == 30
    assert "ds" in daily.columns
    assert "y" in daily.columns
    assert "orders" in daily.columns


def test_build_daily_sales_sorted(clean_df):
    daily = build_daily_sales(clean_df)
    assert daily["ds"].is_monotonic_increasing


def test_build_weekly_sales(clean_df):
    weekly = build_weekly_sales(clean_df)
    assert "ds" in weekly.columns
    assert "y" in weekly.columns
    assert len(weekly) <= 30  # fewer rows than daily


def test_add_lag_features(clean_df):
    daily = build_daily_sales(clean_df)
    result = add_lag_features(daily)
    assert "lag_1d" in result.columns
    assert "lag_7d" in result.columns
    assert "lag_28d" in result.columns
    # First row lag_1d must be NaN
    assert pd.isna(result["lag_1d"].iloc[0])


def test_add_rolling_features(clean_df):
    daily = build_daily_sales(clean_df)
    result = add_rolling_features(daily)
    assert "roll_mean_7d" in result.columns
    assert "roll_std_7d" in result.columns
    assert "roll_max_28d" in result.columns


def test_add_calendar_features(clean_df):
    daily = build_daily_sales(clean_df)
    result = add_calendar_features(daily)
    for col in ["day_of_week", "month", "quarter", "is_weekend", "is_q4"]:
        assert col in result.columns
    assert result["is_weekend"].isin([0, 1]).all()
    assert result["is_q4"].isin([0, 1]).all()