"""
Inventory Optimization — Reorder point + safety stock per SKU.

Formula:
  Safety Stock  = Z * σ_demand * √(lead_time)
  Reorder Point = (avg_daily_demand * lead_time) + safety_stock

Usage:
    python -m src.models.inventory
"""

import sys
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.utils.config import PROCESSED_DIR, LEAD_TIME_DAYS, SERVICE_LEVEL
from src.utils.logging_config import setup_logger

INPUT_PATH  = PROCESSED_DIR / "cleaned_data.parquet"
OUTPUT_PATH = PROCESSED_DIR / "inventory_metrics.csv"


def compute_z_score(service_level: float) -> float:
    """Z-score for given service level (e.g. 0.95 → 1.645)."""
    return float(stats.norm.ppf(service_level))


def compute_sku_demand_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily demand stats per SKU."""
    # Daily qty per SKU
    sku_daily = (
        df.groupby(["StockCode", df["InvoiceDate"].dt.normalize()])["Quantity"]
        .sum()
        .reset_index()
    )
    sku_daily.columns = ["StockCode", "Date", "DailyQty"]

    # Demand stats
    stats_df = (
        sku_daily.groupby("StockCode")["DailyQty"]
        .agg(
            avg_daily_demand="mean",
            std_daily_demand="std",
            max_daily_demand="max",
            total_sold="sum",
            days_with_sales="count",
        )
        .reset_index()
        .fillna({"std_daily_demand": 0})
        .round(2)
    )

    return stats_df


def compute_reorder_metrics(
    stats_df: pd.DataFrame,
    lead_time: int,
    z: float,
) -> pd.DataFrame:
    """Add safety stock, reorder point, EOQ proxy."""
    df = stats_df.copy()

    df["safety_stock"] = (
        z * df["std_daily_demand"] * np.sqrt(lead_time)
    ).round(0).astype(int)

    df["lead_time_demand"] = (
        df["avg_daily_demand"] * lead_time
    ).round(0).astype(int)

    df["reorder_point"] = df["lead_time_demand"] + df["safety_stock"]

    # Risk tier
    demand_75 = df["avg_daily_demand"].quantile(0.75)
    demand_50 = df["avg_daily_demand"].quantile(0.50)

    def risk_tier(row):
        if row["avg_daily_demand"] >= demand_75:
            return "High"
        elif row["avg_daily_demand"] >= demand_50:
            return "Medium"
        else:
            return "Low"

    df["risk_tier"] = df.apply(risk_tier, axis=1)

    return df


def add_sku_metadata(stats_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach description + primary country per SKU."""
    desc = df.groupby("StockCode")["Description"].first().reset_index()
    country = (
        df.groupby("StockCode")["Country"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Country": "primary_country"})
    )
    revenue = (
        df.groupby("StockCode")["TotalPrice"]
        .sum()
        .round(2)
        .reset_index()
        .rename(columns={"TotalPrice": "total_revenue"})
    )

    result = stats_df.merge(desc,    on="StockCode", how="left")
    result = result.merge(country,   on="StockCode", how="left")
    result = result.merge(revenue,   on="StockCode", how="left")

    return result


def run() -> pd.DataFrame:
    setup_logger()

    logger.info("=" * 50)
    logger.info("MODEL 4/4 — INVENTORY OPTIMIZATION")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make clean` first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    logger.info(f"Loaded: {len(df):,} rows  |  {df['StockCode'].nunique():,} SKUs")

    z = compute_z_score(SERVICE_LEVEL)
    logger.info(f"Service level: {SERVICE_LEVEL:.0%}  |  Z-score: {z:.3f}  |  Lead time: {LEAD_TIME_DAYS}d")

    stats_df = compute_sku_demand_stats(df)
    metrics  = compute_reorder_metrics(stats_df, LEAD_TIME_DAYS, z)
    metrics  = add_sku_metadata(metrics, df)

    # Sort by risk + revenue
    metrics = metrics.sort_values(
        ["risk_tier", "total_revenue"],
        ascending=[True, False]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_PATH, index=False)
    logger.success(f"Saved → {OUTPUT_PATH}  ({len(metrics):,} SKUs)")

    # Summary
    tier_counts = metrics["risk_tier"].value_counts()
    logger.info("─" * 50)
    for tier, count in tier_counts.items():
        logger.info(f"  {tier:<8} risk SKUs : {count:>5,}")
    logger.info(f"  Avg reorder point : {metrics['reorder_point'].mean():.0f} units")
    logger.info(f"  Max reorder point : {metrics['reorder_point'].max():.0f} units")
    logger.info("─" * 50)

    return metrics


if __name__ == "__main__":
    run()