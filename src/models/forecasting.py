"""
Demand Forecasting — ARIMA / SARIMA on daily sales.

Steps:
  1. Load daily_sales.csv
  2. Try SARIMA(1,1,1)(1,1,1,7) — weekly seasonality
  3. Fallback to ARIMA(1,1,1) if SARIMA fails
  4. Forecast 90 days ahead with confidence intervals
  5. Compute MAPE on in-sample fit
  6. Save forecast CSV + model

Usage:
    python -m src.models.forecasting
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from loguru import logger

import mlflow
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.config import FEATURES_DIR, PROCESSED_DIR, MODELS_DIR
from src.utils.config import FORECAST_HORIZON_DAYS
from src.utils.logging_config import setup_logger
from src.utils.mlflow_utils import setup_mlflow

warnings.filterwarnings("ignore")

INPUT_PATH   = FEATURES_DIR  / "daily_sales.csv"
OUTPUT_PATH  = PROCESSED_DIR / "forecast.csv"


def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error — ignores zeros."""
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def train_sarima(series: pd.Series, horizon: int) -> dict:
    """Fit SARIMA(1,1,1)(1,1,1,7). Returns forecast dict or raises on failure."""
    logger.info("Fitting SARIMA(1,1,1)(1,1,1,7)...")
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    forecast_res = fitted.get_forecast(steps=horizon)
    forecast_mean = forecast_res.predicted_mean
    conf_int      = forecast_res.conf_int(alpha=0.05)

    mape = compute_mape(series, fitted.fittedvalues)
    logger.info(f"SARIMA train MAPE: {mape:.2f}%")

    return {
        "model_type": "SARIMA",
        "fitted":     fitted,
        "forecast":   forecast_mean,
        "lower_ci":   conf_int.iloc[:, 0],
        "upper_ci":   conf_int.iloc[:, 1],
        "mape":       mape,
    }


def train_arima(series: pd.Series, horizon: int) -> dict:
    """Fallback ARIMA(1,1,1)."""
    logger.info("Fitting ARIMA(1,1,1)...")
    model = ARIMA(series, order=(1, 1, 1))
    fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=horizon)
    forecast_mean = forecast_res.predicted_mean
    conf_int      = forecast_res.conf_int(alpha=0.05)

    mape = compute_mape(series, fitted.fittedvalues)
    logger.info(f"ARIMA train MAPE: {mape:.2f}%")

    return {
        "model_type": "ARIMA",
        "fitted":     fitted,
        "forecast":   forecast_mean,
        "lower_ci":   conf_int.iloc[:, 0],
        "upper_ci":   conf_int.iloc[:, 1],
        "mape":       mape,
    }


def build_forecast_df(daily: pd.DataFrame, result: dict, horizon: int) -> pd.DataFrame:
    """Combine historical actuals + forecast into one DataFrame."""
    last_date  = pd.to_datetime(daily["ds"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    forecast_df = pd.DataFrame({
        "ds":       future_dates,
        "yhat":     result["forecast"].values,
        "yhat_lower": result["lower_ci"].values,
        "yhat_upper": result["upper_ci"].values,
        "type":     "forecast",
    })

    hist_df = daily[["ds", "y"]].copy()
    hist_df["ds"]   = pd.to_datetime(hist_df["ds"])
    hist_df["yhat"] = np.nan
    hist_df["yhat_lower"] = np.nan
    hist_df["yhat_upper"] = np.nan
    hist_df["type"] = "actual"
    hist_df = hist_df.rename(columns={"y": "y_actual"})

    # Combine
    combined = pd.concat([hist_df, forecast_df], ignore_index=True)
    combined[["yhat", "yhat_lower", "yhat_upper"]] = combined[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0).round(2)

    return combined


def run() -> dict:
    setup_logger()
    setup_mlflow()

    logger.info("=" * 50)
    logger.info("MODEL 2/4 — DEMAND FORECASTING (ARIMA/SARIMA)")
    logger.info("=" * 50)

    if not INPUT_PATH.exists():
        logger.error(f"Input not found: {INPUT_PATH}. Run `make features` first.")
        sys.exit(1)

    daily = pd.read_csv(INPUT_PATH, parse_dates=["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)
    logger.info(f"Loaded: {len(daily)} days of sales data")

    series = daily.set_index("ds")["y"].asfreq("D").fillna(0)

    # Try SARIMA → fallback ARIMA
    try:
        result = train_sarima(series, FORECAST_HORIZON_DAYS)
    except Exception as e:
        logger.warning(f"SARIMA failed ({e}). Falling back to ARIMA.")
        result = train_arima(series, FORECAST_HORIZON_DAYS)

    with mlflow.start_run(run_name=f"forecasting_{result['model_type'].lower()}"):
        mlflow.log_param("model_type",      result["model_type"])
        mlflow.log_param("horizon_days",    FORECAST_HORIZON_DAYS)
        mlflow.log_param("train_days",      len(daily))
        mlflow.log_metric("train_mape",     round(result["mape"], 4))

    # Build + save forecast DataFrame
    forecast_df = build_forecast_df(daily, result, FORECAST_HORIZON_DAYS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["fitted"], MODELS_DIR / "forecast_model.joblib")

    logger.success(f"Forecast → {OUTPUT_PATH}  ({FORECAST_HORIZON_DAYS} days ahead)")
    logger.success(f"Model    → {MODELS_DIR}/forecast_model.joblib")
    logger.info(f"MAPE     : {result['mape']:.2f}%  (target ≤ 15%)")

    return result


if __name__ == "__main__":
    run()