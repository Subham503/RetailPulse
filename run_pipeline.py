"""
RetailPulse — Master Pipeline Orchestrator

Runs all 6 pipeline steps in sequence:
  1. Ingest      → data/processed/raw_combined.parquet
  2. Clean       → data/processed/cleaned_data.parquet
  3. Features    → data/processed/rfm.csv + data/features/daily_sales.csv
  4. Models      → models/*.joblib + MLflow runs
  5. Forecast    → data/processed/forecast.csv
  6. Inventory   → data/processed/inventory_metrics.csv

Usage:
    python run_pipeline.py                 # full run
    python run_pipeline.py --from clean    # resume from a step
    python run_pipeline.py --step ingest   # single step only
"""

import argparse
import time
from loguru import logger
from src.utils.logging_config import setup_logger

STEPS = ["ingest", "clean", "features", "segmentation", "forecasting", "churn", "inventory"]


def run_step(name: str) -> None:
    start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"  RUNNING: {name.upper()}")
    logger.info(f"{'='*60}")

    if name == "ingest":
        from src.data.ingest import run
        run()
    elif name == "clean":
        from src.data.clean import run
        run()
    elif name == "features":
        from src.features.rfm import run as run_rfm
        from src.features.time_features import run as run_time
        run_rfm()
        run_time()
    elif name == "segmentation":
        from src.models.segmentation import run
        run()
    elif name == "forecasting":
        from src.models.forecasting import run
        run()
    elif name == "churn":
        from src.models.churn import run
        run()
    elif name == "inventory":
        from src.models.inventory import run
        run()

    elapsed = time.time() - start
    logger.success(f"  ✅ {name.upper()} done in {elapsed:.1f}s")


def main(from_step: str = None, single_step: str = None) -> None:
    setup_logger(log_file="logs/pipeline.log")
    pipeline_start = time.time()

    logger.info("🛒 RetailPulse Pipeline Starting")
    logger.info(f"Steps: {' → '.join(STEPS)}")

    if single_step:
        run_step(single_step)
    else:
        start_idx = STEPS.index(from_step) if from_step else 0
        for step in STEPS[start_idx:]:
            run_step(step)

    total = time.time() - pipeline_start
    logger.success(f"\n🎉 Pipeline complete in {total/60:.1f} min")
    logger.info("Next steps:")
    logger.info("  make dashboard   → Streamlit on :8501")
    logger.info("  make api         → FastAPI on :8000")
    logger.info("  make mlflow      → MLflow UI on :5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailPulse Pipeline")
    parser.add_argument("--from",  dest="from_step",    choices=STEPS, default=None,
                        help="Resume from a specific step")
    parser.add_argument("--step",  dest="single_step",  choices=STEPS, default=None,
                        help="Run one step only")
    args = parser.parse_args()
    main(args.from_step, args.single_step)
