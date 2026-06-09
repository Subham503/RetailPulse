"""
RetailPulse — FastAPI REST API

Endpoints:
  GET  /health
  POST /predict/churn
  POST /predict/forecast
  GET  /segments
  GET  /inventory/risk

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import churn, forecast, segments, inventory

app = FastAPI(
    title="RetailPulse API",
    description="ML-powered retail analytics — churn, forecasting, segmentation, inventory",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(churn.router,     prefix="/predict",   tags=["Churn"])
app.include_router(forecast.router,  prefix="/predict",   tags=["Forecast"])
app.include_router(segments.router,                       tags=["Segments"])
app.include_router(inventory.router, prefix="/inventory", tags=["Inventory"])


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check."""
    return {"status": "ok", "service": "RetailPulse API", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)