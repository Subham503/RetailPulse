"""
RFM Feature Engineering.

Computes Recency, Frequency, Monetary per customer,
scores them 1-5, assigns segment labels.

Usage:
    python -m src.features.rfm
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.config import PROCESSED_DIR
from src.utils.logging_config import setup_logger

INPUT_PATH  = PROCESSED_DIR / "cleaned_data.parquet"
OUTPUT_PATH = PROCESSED_DIR / "rfm.csv"


# ── RFM Computation ────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute RFM metrics per customer.

    Parameters
    ----------
    df            : cleaned DataFrame with InvoiceDate, Invoice, TotalPrice, CustomerID
    snapshot_date : reference date for recency (default: max date + 1 day)

    Returns
    -------
    DataFrame with columns: CustomerID, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    logger.info(f"Snapshot date: {snapshot_date.date()}")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency = ("Invoice",     "nunique"),
            Monetary  = ("TotalPrice",  "sum"),
        )
        .reset_index()
    )

    rfm["Monetary"] = rfm["Monetary"].round(2)

    logger.info(f"RFM computed for {len(rfm):,} customers")
    return rfm


# ── RFM Scoring ────────────────────────────────────────────────────────────

def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add R/F/M scores (1–5) and combined RFM_Total.

    Score logic:
      R: lower recency  = better = higher score
      F: higher frequency = better = higher score
      M: higher monetary  = better = higher score
    """
    rfm = rfm.copy()

    # R score: 5=most recent, 1=least recent
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)

    # F score: 5=most frequent
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # M score: 5=highest spend
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    logger.info("RFM scores (1–5) added: R_Score, F_Score, M_Score, RFM_Total")
    return rfm


# ── Segment Labels ─────────────────────────────────────────────────────────

SEGMENT_MAP = {
    "Champions":        lambda r, f, m: (r >= 4) & (f >= 4) & (m >= 4),
    "Loyal":            lambda r, f, m: (r >= 3) & (f >= 3),
    "Potential Loyal":  lambda r, f, m: (r >= 3) & (f <= 2),
    "At-Risk":          lambda r, f, m: (r <= 2) & (f >= 3),
    "Lost":             lambda r, f, m: (r <= 2) & (f <= 2),
}


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign business segment label to each customer."""
    rfm = rfm.copy()
    rfm["Segment"] = "Other"

    # Apply in priority order (first match wins)
    for label, condition in SEGMENT_MAP.items():
        mask = condition(rfm["R_Score"], rfm["F_Score"], rfm["M_Score"])
        rfm.loc[mask & (rfm["Segment"] == "Other"), "Segment"] = label

    counts = rfm["Segment"].value_counts()
    for seg, count in counts.items():
        pct = count / len(rfm) * 100
        logger.info(f"  {seg:<20}: {count:>5,}  ({pct:.1f}%)")

    return rfm


# ── Additional Customer Features ───────────────────────────────────────────

def add_customer_features(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add avg order value, avg items per order, country."""
    # Avg order value
    aov = (
        df.groupby(["CustomerID", "Invoice"])["TotalPrice"]
        .sum()
        .groupby("CustomerID")
        .mean()
        .round(2)
        .rename("AvgOrderValue")
    )

    # Avg items per order
    aip = (
        df.groupby(["CustomerID", "Invoice"])["Quantity"]
        .sum()
        .groupby("CustomerID")
        .mean()
        .round(1)
        .rename("AvgItemsPerOrder")
    )

    # Primary country
    country = (
        df.groupby("CustomerID")["Country"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("Country")
    )

    rfm = rfm.merge(aov,     on="CustomerID", how="left")
    rfm = rfm.merge(aip,     on="CustomerID", how="left")
    rfm = rfm.merge(country, on="CustomerID", how="left")

    logger.info("Extra features added: AvgOrderValue, AvgItemsPerOrder, Country")
    return rfm


# ── Full Pipeline ──────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Full RFM pipeline: compute → score → segment → extra features."""
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)
    rfm = add_customer_features(rfm, df)
    return rfm


def run() -> pd.DataFrame:
    setup_logger()
    logger.info("=" * 50)
    logger.info("STEP 3a/6 — RFM FEATURES")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make clean` first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded: {len(df):,} rows")

    rfm = build_rfm(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Saved → {OUTPUT_PATH}  ({len(rfm):,} customers)")

    # Stats summary
    logger.info("─" * 50)
    logger.info(f"  Recency   — mean: {rfm['Recency'].mean():.0f}d  median: {rfm['Recency'].median():.0f}d")
    logger.info(f"  Frequency — mean: {rfm['Frequency'].mean():.1f}   max: {rfm['Frequency'].max()}")
    logger.info(f"  Monetary  — mean: £{rfm['Monetary'].mean():,.0f}  max: £{rfm['Monetary'].max():,.0f}")
    logger.info("─" * 50)

    return rfm


if __name__ == "__main__":
    run()