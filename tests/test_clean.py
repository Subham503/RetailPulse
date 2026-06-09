"""Unit tests for src/data/clean.py"""

import pandas as pd
import numpy as np
import pytest
from src.data.clean import (
    drop_missing_customer,
    remove_cancellations,
    fix_dtypes,
    remove_invalid_quantities,
    add_derived_columns,
    drop_duplicates,
    clean,
)


@pytest.fixture
def sample_df():
    # CustomerID already renamed by ingest.py — no space
    return pd.DataFrame({
        "Invoice":     ["536365", "C536366", "536367", "536368", "536365"],
        "StockCode":   ["85123A", "85123B", "85123C", "POST",   "85123A"],
        "Description": ["Item A", "Item B", "Item C", "Postage", "Item A"],
        "Quantity":    [6,         3,         -1,        1,         6],
        "InvoiceDate": pd.to_datetime(["2010-01-01 08:00"] * 5),
        "Price":       [2.55, 3.39, 0.0, 5.0, 2.55],
        "CustomerID":  ["17850", None, "17851", "17852", "17850"],
        "Country":     ["UK"] * 5,
    })


def test_drop_missing_customer(sample_df):
    df = drop_missing_customer(sample_df)
    assert df["CustomerID"].notna().all()
    assert len(df) == 4


def test_remove_cancellations(sample_df):
    df = remove_cancellations(sample_df)
    assert not df["Invoice"].str.startswith("C").any()


def test_fix_dtypes(sample_df):
    df = fix_dtypes(sample_df)
    assert df["InvoiceDate"].dtype == "datetime64[ns]"
    assert df["Quantity"].dtype in [np.float64, np.int64]
    assert df["Price"].dtype in [np.float64, np.int64]


def test_remove_invalid_quantities(sample_df):
    df = fix_dtypes(sample_df)
    df = remove_invalid_quantities(df)
    assert (df["Quantity"] > 0).all()
    assert (df["Price"] > 0).all()


def test_add_derived_columns(sample_df):
    df = fix_dtypes(sample_df)
    df = remove_invalid_quantities(df)
    df = add_derived_columns(df)
    assert "TotalPrice" in df.columns
    assert "Year" in df.columns
    assert "Month" in df.columns
    assert (df["TotalPrice"] > 0).all()


def test_drop_duplicates(sample_df):
    df = sample_df.rename(columns={"Customer ID": "CustomerID"})
    before = len(df)
    df = drop_duplicates(df)
    assert len(df) < before  # row 0 and row 4 are duplicates


def test_full_pipeline_reduces_rows(sample_df):
    df = clean(sample_df)
    assert len(df) < len(sample_df)
    assert "TotalPrice" in df.columns
    assert df["Quantity"].min() > 0
    assert df["Price"].min() > 0