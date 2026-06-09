"""GET /segments — return RFM segments for all customers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.utils.config import PROCESSED_DIR

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────
class CustomerSegment(BaseModel):
    customer_id:  str
    segment:      str
    recency:      int
    frequency:    int
    monetary:     float
    rfm_total:    int | None
    churn_prob:   float | None

class SegmentsResponse(BaseModel):
    total:     int
    segments:  list[str]
    customers: list[CustomerSegment]


# ── Cache ──────────────────────────────────────────────────────────────────
_rfm_cache: pd.DataFrame | None = None

def _get_rfm() -> pd.DataFrame:
    global _rfm_cache
    if _rfm_cache is None:
        for fname in ("rfm_with_churn.csv", "rfm_clustered.csv", "rfm.csv"):
            p = PROCESSED_DIR / fname
            if p.exists():
                _rfm_cache = pd.read_csv(p)
                _rfm_cache["CustomerID"] = _rfm_cache["CustomerID"].astype(str)
                break
    if _rfm_cache is None:
        raise HTTPException(500, "RFM data not found. Run pipeline first.")
    return _rfm_cache


# ── Endpoint ───────────────────────────────────────────────────────────────
@router.get("/segments", response_model=SegmentsResponse)
def get_segments(
    segment: str = Query(default=None, description="Filter by segment name"),
    limit:   int = Query(default=100,  ge=1, le=5000, description="Max rows to return"),
):
    """Return RFM segment for all (or filtered) customers."""
    rfm = _get_rfm()

    if segment:
        filtered = rfm[rfm.get("Segment", rfm.iloc[:, 0]).astype(str).str.lower() == segment.lower()]
    else:
        filtered = rfm

    filtered = filtered.head(limit)

    customers = [
        CustomerSegment(
            customer_id = str(row["CustomerID"]),
            segment     = str(row.get("Segment", "Unknown")),
            recency     = int(row["Recency"]),
            frequency   = int(row["Frequency"]),
            monetary    = round(float(row["Monetary"]), 2),
            rfm_total   = int(row["RFM_Total"])  if "RFM_Total"  in row and pd.notna(row["RFM_Total"])  else None,
            churn_prob  = round(float(row["ChurnProb"]), 4) if "ChurnProb" in row and pd.notna(row["ChurnProb"]) else None,
        )
        for _, row in filtered.iterrows()
    ]

    all_segs = sorted(rfm["Segment"].dropna().unique().tolist()) if "Segment" in rfm.columns else []

    return SegmentsResponse(
        total     = len(filtered),
        segments  = all_segs,
        customers = customers,
    )