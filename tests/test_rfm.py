"""Unit tests for src/features/rfm.py"""

import pandas as pd
import numpy as np
import pytest
from src.features.rfm import compute_rfm, score_rfm, assign_segments, build_rfm


@pytest.fixture
def clean_df():
    """Minimal cleaned DataFrame for RFM testing."""
    return pd.DataFrame({
        "CustomerID":  ["C1", "C1", "C2", "C3", "C3", "C3"],
        "Invoice":     ["I1", "I2", "I3", "I4", "I5", "I6"],
        "InvoiceDate": pd.to_datetime([
            "2011-10-01", "2011-11-15",   # C1: 2 orders
            "2011-06-01",                  # C2: 1 order (older)
            "2011-11-01", "2011-11-20", "2011-12-01",  # C3: 3 orders (most recent)
        ]),
        "TotalPrice": [100.0, 200.0, 50.0, 80.0, 120.0, 300.0],
        "Quantity":   [2, 4, 1, 3, 2, 5],
        "Country":    ["UK"] * 6,
    })


def test_compute_rfm_shape(clean_df):
    rfm = compute_rfm(clean_df)
    assert len(rfm) == 3  # 3 unique customers
    assert set(["CustomerID", "Recency", "Frequency", "Monetary"]).issubset(rfm.columns)


def test_compute_rfm_values(clean_df):
    snapshot = pd.Timestamp("2011-12-10")
    rfm = compute_rfm(clean_df, snapshot_date=snapshot)

    c1 = rfm[rfm["CustomerID"] == "C1"].iloc[0]
    c2 = rfm[rfm["CustomerID"] == "C2"].iloc[0]
    c3 = rfm[rfm["CustomerID"] == "C3"].iloc[0]

    # C3 most recent (last order 2011-12-01)
    assert c3["Recency"] < c1["Recency"] < c2["Recency"]

    # C3 most frequent (3 orders)
    assert c3["Frequency"] == 3
    assert c1["Frequency"] == 2
    assert c2["Frequency"] == 1

    # C3 highest monetary
    assert c3["Monetary"] == pytest.approx(500.0)
    assert c1["Monetary"] == pytest.approx(300.0)


def test_score_rfm_columns(clean_df):
    rfm = compute_rfm(clean_df)
    rfm = score_rfm(rfm)
    for col in ["R_Score", "F_Score", "M_Score", "RFM_Score", "RFM_Total"]:
        assert col in rfm.columns


def test_score_rfm_range(clean_df):
    rfm = compute_rfm(clean_df)
    rfm = score_rfm(rfm)
    assert rfm["R_Score"].between(1, 5).all()
    assert rfm["F_Score"].between(1, 5).all()
    assert rfm["M_Score"].between(1, 5).all()
    assert rfm["RFM_Total"].between(3, 15).all()


def test_assign_segments_no_nulls(clean_df):
    rfm = compute_rfm(clean_df)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)
    assert rfm["Segment"].notna().all()
    assert "Segment" in rfm.columns


def test_build_rfm_full_pipeline(clean_df):
    rfm = build_rfm(clean_df)
    expected_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                     "R_Score", "F_Score", "M_Score", "RFM_Total", "Segment"]
    for col in expected_cols:
        assert col in rfm.columns
    assert len(rfm) == 3