"""POST /predict/forecast — return N-day demand forecast."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from src.utils.config import PROCESSED_DIR

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    date:       str
    forecast:   float
    lower_ci:   float
    upper_ci:   float

class ForecastResponse(BaseModel):
    horizon_days: int
    avg_daily:    float
    total:        float
    forecast:     list[ForecastPoint]


# ── Cache ──────────────────────────────────────────────────────────────────
_forecast_cache: pd.DataFrame | None = None

def _get_forecast() -> pd.DataFrame:
    global _forecast_cache
    if _forecast_cache is None:
        p = PROCESSED_DIR / "forecast.csv"
        if not p.exists():
            raise HTTPException(500, "Forecast data not found. Run `python -m src.models.forecasting`.")
        df = pd.read_csv(p, parse_dates=["ds"])
        _forecast_cache = df[df["type"] == "forecast"].copy()
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            _forecast_cache[col] = pd.to_numeric(_forecast_cache[col], errors="coerce").clip(lower=0)
    return _forecast_cache


# ── Endpoint ───────────────────────────────────────────────────────────────
@router.post("/forecast", response_model=ForecastResponse)
def predict_forecast(
    days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days (1–90)")
):
    """Return demand forecast for the next N days."""
    fc = _get_forecast().head(days)

    if fc.empty:
        raise HTTPException(404, "No forecast data available.")

    points = [
        ForecastPoint(
            date     = row["ds"].strftime("%Y-%m-%d"),
            forecast = round(float(row["yhat"]),       2),
            lower_ci = round(float(row["yhat_lower"]), 2),
            upper_ci = round(float(row["yhat_upper"]), 2),
        )
        for _, row in fc.iterrows()
    ]

    return ForecastResponse(
        horizon_days = days,
        avg_daily    = round(float(fc["yhat"].mean()), 2),
        total        = round(float(fc["yhat"].sum()),  2),
        forecast     = points,
    )