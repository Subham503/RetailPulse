"""
RFM Feature Engineering.

Computes Recency, Frequency, and Monetary metrics per customer,
assigns RFM scores (1-5), and segments customers.

Usage:
    python -m src.features.rfm
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.config import PROCESSED_DIR, FEATURES_DIR
from src.utils.logging_config import setup_logger

INPUT_PATH  = PROCESSED_DIR / "cleaned_data.parquet"
OUTPUT_PATH = PROCESSED_DIR / "rfm.csv"


def compute_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary values for each customer.
    
    Recency: days since last purchase relative to snapshot_date.
    Frequency: number of unique orders/invoices.
    Monetary: total money spent.
    """
    if snapshot_date is None:
        # If not provided, snapshot is 1 day after the latest transaction
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)
        
    # Group by customer and compute RFM metrics
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency    = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency  = ("Invoice", "nunique"),
            Monetary   = ("TotalPrice", "sum"),
        )
        .reset_index()
    )
    
    logger.info(f"Computed RFM for {len(rfm)} customers.")
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign R, F, and M scores from 1 to 5.
    R score: 5 is most recent, 1 is least recent (lower Recency is better).
    F score: 5 is most frequent, 1 is least frequent (higher Frequency is better).
    M score: 5 is highest spend, 1 is lowest spend (higher Monetary is better).
    """
    df = rfm.copy()
    
    # Use rank(method='first') to handle ties and small datasets safely with qcut
    df["R_Score"] = pd.qcut(df["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Total"] = df["R_Score"] + df["F_Score"] + df["M_Score"]
    
    logger.info("Assigned RFM scores.")
    return df


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a customer segment label based on R and F scores.
    """
    df = rfm.copy()
    
    def get_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        
        # Segment definitions mapping R and F scores
        if r >= 4 and f >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f >= 1:
            return "Recent Customers"
        elif r >= 3 and f >= 1:
            return "Potential Loyalists"
        elif r >= 2 and f >= 2:
            return "Customers Needing Attention"
        elif r >= 2 and f >= 1:
            return "About to Sleep"
        elif r >= 1 and f >= 4:
            return "Cant Lose Them"
        elif r >= 1 and f >= 3:
            return "At Risk"
        else:
            return "Hibernating"
            
    df["Segment"] = df.apply(get_segment, axis=1)
    
    logger.info("Assigned customer segments.")
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full RFM feature engineering pipeline.
    """
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    rfm = assign_segments(rfm)
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
    logger.success(f"Saved → {OUTPUT_PATH}")

    # Display segment distribution
    logger.info("Segment Distribution:")
    dist = rfm["Segment"].value_counts()
    for seg, count in dist.items():
         logger.info(f"  {seg:<30}: {count:>5} ({count/len(rfm):.1%})")

    return rfm


if __name__ == "__main__":
    run()
