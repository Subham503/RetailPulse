"""GET /inventory/risk — return inventory risk table."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.utils.config import PROCESSED_DIR

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────
class SKURisk(BaseModel):
    sku:               str
    description:       str | None
    avg_daily_demand:  float
    safety_stock:      int
    reorder_point:     int
    total_sold:        float
    risk_tier:         str
    total_revenue:     float | None

class InventoryRiskResponse(BaseModel):
    total_skus:      int
    high_risk_count: int
    filtered_count:  int
    skus:            list[SKURisk]


# ── Cache ──────────────────────────────────────────────────────────────────
_inv_cache: pd.DataFrame | None = None

def _get_inventory() -> pd.DataFrame:
    global _inv_cache
    if _inv_cache is None:
        p = PROCESSED_DIR / "inventory_metrics.csv"
        if not p.exists():
            raise HTTPException(500, "Inventory data not found. Run `python -m src.models.inventory`.")
        _inv_cache = pd.read_csv(p)
    return _inv_cache


# ── Endpoint ───────────────────────────────────────────────────────────────
@router.get("/risk", response_model=InventoryRiskResponse)
def get_inventory_risk(
    tier:  str = Query(default=None,  description="Filter by risk tier: High, Medium, Low"),
    limit: int = Query(default=50,    ge=1, le=1000, description="Max SKUs to return"),
    sort:  str = Query(default="demand", description="Sort by: demand | reorder | revenue"),
):
    """Return inventory risk table with reorder points and safety stock."""
    inv = _get_inventory()

    filtered = inv.copy()
    if tier:
        filtered = filtered[filtered["risk_tier"].str.lower() == tier.lower()]

    sort_map = {
        "demand":  "avg_daily_demand",
        "reorder": "reorder_point",
        "revenue": "total_revenue",
    }
    sort_col = sort_map.get(sort, "avg_daily_demand")
    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=False)

    filtered = filtered.head(limit)

    skus = [
        SKURisk(
            sku              = str(row["StockCode"]),
            description      = str(row["Description"])   if "Description"   in row and pd.notna(row["Description"])   else None,
            avg_daily_demand = round(float(row["avg_daily_demand"]), 2),
            safety_stock     = int(row["safety_stock"]),
            reorder_point    = int(row["reorder_point"]),
            total_sold       = float(row["total_sold"]),
            risk_tier        = str(row["risk_tier"]),
            total_revenue    = round(float(row["total_revenue"]), 2) if "total_revenue" in row and pd.notna(row["total_revenue"]) else None,
        )
        for _, row in filtered.iterrows()
    ]

    return InventoryRiskResponse(
        total_skus      = len(inv),
        high_risk_count = int((inv["risk_tier"] == "High").sum()),
        filtered_count  = len(filtered),
        skus            = skus,
    )