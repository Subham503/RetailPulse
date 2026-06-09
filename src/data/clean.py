"""
Data cleaning pipeline for Online Retail II.

Steps:
  1. Drop rows missing CustomerID
  2. Remove cancellations (Invoice starts with 'C')
  3. Fix / coerce dtypes
  4. Remove non-positive Quantity and Price
  5. Derive TotalPrice
  6. Drop duplicates
  7. Report stats

Usage:
    python -m src.data.clean
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from src.utils.config import PROCESSED_DIR
from src.utils.logging_config import setup_logger

INPUT_PATH  = PROCESSED_DIR / "raw_combined.parquet"
OUTPUT_PATH = PROCESSED_DIR / "cleaned_data.parquet"


# ── Individual cleaning steps ──────────────────────────────────────────────

def drop_missing_customer(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    logger.info(f"  drop missing CustomerID : -{before - len(df):>8,} rows → {len(df):,}")
    return df


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["Invoice"].astype(str).str.startswith("C")
    df = df[~mask]
    logger.info(f"  remove cancellations    : -{before - len(df):>8,} rows → {len(df):,}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"]  = df["CustomerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["Quantity"]    = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"]       = pd.to_numeric(df["Price"],    errors="coerce")
    df["Invoice"]     = df["Invoice"].astype(str).str.strip()
    df["StockCode"]   = df["StockCode"].astype(str).str.strip().str.upper()
    df["Country"]     = df["Country"].astype(str).str.strip()
    logger.info("  fix dtypes              : done")
    return df


def remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    logger.info(f"  remove qty/price ≤ 0   : -{before - len(df):>8,} rows → {len(df):,}")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalPrice"] = (df["Quantity"] * df["Price"]).round(2)
    df["Year"]       = df["InvoiceDate"].dt.year
    df["Month"]      = df["InvoiceDate"].dt.month
    df["DayOfWeek"]  = df["InvoiceDate"].dt.dayofweek   # 0=Monday
    df["Hour"]       = df["InvoiceDate"].dt.hour
    df["Date"]       = df["InvoiceDate"].dt.date
    logger.info("  add derived columns     : TotalPrice, Year, Month, DayOfWeek, Hour, Date")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"  drop duplicates         : -{before - len(df):>8,} rows → {len(df):,}")
    return df


def drop_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    logger.info(f"  drop remaining nulls    : -{before - len(df):>8,} rows → {len(df):,}")
    return df


def remove_test_skus(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known test / non-product StockCodes."""
    test_codes = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK"}
    before = len(df)
    df = df[~df["StockCode"].isin(test_codes)]
    logger.info(f"  remove test SKUs        : -{before - len(df):>8,} rows → {len(df):,}")
    return df


# ── Full pipeline ──────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full cleaning pipeline. Returns cleaned DataFrame."""
    logger.info(f"Input shape: {df.shape}")
    df = (
        df
        .pipe(drop_missing_customer)
        .pipe(remove_cancellations)
        .pipe(fix_dtypes)
        .pipe(remove_invalid_quantities)
        .pipe(remove_test_skus)
        .pipe(drop_duplicates)
        .pipe(drop_remaining_nulls)
        .pipe(add_derived_columns)
    )
    return df


def data_quality_report(df: pd.DataFrame) -> None:
    """Log basic DQ stats after cleaning."""
    logger.info("─" * 50)
    logger.info("DATA QUALITY REPORT")
    logger.info("─" * 50)
    logger.info(f"  Total rows         : {len(df):,}")
    logger.info(f"  Unique customers   : {df['CustomerID'].nunique():,}")
    logger.info(f"  Unique invoices    : {df['Invoice'].nunique():,}")
    logger.info(f"  Unique SKUs        : {df['StockCode'].nunique():,}")
    logger.info(f"  Countries          : {df['Country'].nunique():,}")
    logger.info(f"  Date range         : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    logger.info(f"  Total revenue      : £{df['TotalPrice'].sum():>12,.2f}")
    logger.info(f"  Avg order value    : £{df.groupby('Invoice')['TotalPrice'].sum().mean():>8,.2f}")
    logger.info(f"  Null count         : {df.isnull().sum().sum()}")
    logger.info("─" * 50)


def run() -> pd.DataFrame:
    """Full cleaning step: load parquet → clean → save → return."""
    setup_logger()
    logger.info("=" * 50)
    logger.info("STEP 2/6 — DATA CLEANING")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make ingest` first.")
        sys.exit(1)

    df_raw = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded raw: {len(df_raw):,} rows")

    df_clean = clean(df_raw)

    data_quality_report(df_clean)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow")
    logger.success(f"Saved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")

    return df_clean


if __name__ == "__main__":
    run()