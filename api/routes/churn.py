"""POST /predict/churn — return churn probability for a customer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.config import PROCESSED_DIR, MODELS_DIR

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────
class ChurnRequest(BaseModel):
    customer_id: str

class ChurnResponse(BaseModel):
    customer_id:  str
    churn_prob:   float
    churn_label:  int
    segment:      str | None
    recency_days: int
    frequency:    int
    monetary:     float
    risk_level:   str


# ── Data loader (lazy, cached at module level) ─────────────────────────────
_rfm_cache: pd.DataFrame | None = None

def _get_rfm() -> pd.DataFrame:
    global _rfm_cache
    if _rfm_cache is None:
        for fname in ("rfm_with_churn.csv", "rfm_clustered.csv", "rfm.csv"):
            p = PROCESSED_DIR / fname
            if p.exists():
                _rfm_cache = pd.read_csv(p)
                _rfm_cache["CustomerID"] = _rfm_cache["CustomerID"].astype(str).str.strip()
                break
    if _rfm_cache is None:
        raise HTTPException(500, "RFM data not found. Run the pipeline first.")
    return _rfm_cache


# ── Endpoint ───────────────────────────────────────────────────────────────
@router.post("/churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    """Return churn probability for a given customer ID."""
    rfm = _get_rfm()
    row = rfm[rfm["CustomerID"] == req.customer_id.strip()]

    if row.empty:
        raise HTTPException(404, f"Customer '{req.customer_id}' not found.")

    r = row.iloc[0]

    churn_prob  = float(r.get("ChurnProb",  1.0 if r["Recency"] >= 90 else 0.0))
    churn_label = int(r.get("ChurnLabel",   1 if churn_prob >= 0.5 else 0))

    if churn_prob >= 0.8:
        risk = "High"
    elif churn_prob >= 0.5:
        risk = "Medium"
    else:
        risk = "Low"

    return ChurnResponse(
        customer_id  = req.customer_id,
        churn_prob   = round(churn_prob, 4),
        churn_label  = churn_label,
        segment      = str(r.get("Segment", "Unknown")),
        recency_days = int(r["Recency"]),
        frequency    = int(r["Frequency"]),
        monetary     = round(float(r["Monetary"]), 2),
        risk_level   = risk,
    )