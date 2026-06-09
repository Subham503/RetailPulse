"""
Evidently AI — Data Drift + Model Performance Monitoring.

Compares reference data (first 70%) vs current data (last 30%)
to detect data drift and generate HTML reports.

Usage:
    python -m monitoring.drift_report
    python -m monitoring.drift_report --report all
    python -m monitoring.drift_report --report drift
    python -m monitoring.drift_report --report performance
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from loguru import logger
from src.utils.config import PROCESSED_DIR
from src.utils.logging_config import setup_logger

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RFM data, split into reference (70%) and current (30%)."""
    for fname in ("rfm_with_churn.csv", "rfm_clustered.csv", "rfm.csv"):
        p = PROCESSED_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        logger.error("No RFM data found. Run pipeline first.")
        sys.exit(1)

    split = int(len(df) * 0.7)
    reference = df.iloc[:split].copy()
    current   = df.iloc[split:].copy()

    logger.info(f"Reference: {len(reference)} rows  |  Current: {len(current)} rows")
    return reference, current


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns for drift detection."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    exclude = ["Churned", "ChurnLabel", "Cluster", "R_Score", "F_Score", "M_Score"]
    return [c for c in numeric_cols if c not in exclude]


def run_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate Evidently data drift report."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error("evidently not installed. Run: pip install evidently")
        return

    cols = get_feature_cols(reference)
    ref = reference[cols].copy()
    cur = current[cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "data_drift_report.html"
    report.save_html(str(out))
    logger.success(f"Data drift report → {out}")

    # Extract drift summary
    result = report.as_dict()
    try:
        drift_metrics = result["metrics"][0]["result"]
        n_drifted = drift_metrics.get("number_of_drifted_columns", "N/A")
        share      = drift_metrics.get("share_of_drifted_columns",  "N/A")
        logger.info(f"Drifted columns: {n_drifted}  |  Drift share: {share}")
    except Exception:
        pass


def run_data_quality_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate Evidently data quality report."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataQualityPreset
    except ImportError:
        logger.error("evidently not installed.")
        return

    cols = get_feature_cols(reference)
    ref = reference[cols].copy()
    cur = current[cols].copy()

    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "data_quality_report.html"
    report.save_html(str(out))
    logger.success(f"Data quality report → {out}")


def run_target_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate target drift report if churn labels exist."""
    if "Churned" not in reference.columns:
        logger.warning("No 'Churned' column — skipping target drift.")
        return

    try:
        from evidently.report import Report
        from evidently.metric_preset import TargetDriftPreset
    except ImportError:
        logger.error("evidently not installed.")
        return

    cols  = get_feature_cols(reference) + ["Churned"]
    ref   = reference[cols].copy()
    cur   = current[cols].copy()

    from evidently.pipeline.column_mapping import ColumnMapping
    cm = ColumnMapping()
    cm.target = "Churned"
    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=cm)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "target_drift_report.html"
    report.save_html(str(out))
    logger.success(f"Target drift report → {out}")


def run_churn_performance_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    """Generate classification performance report if predictions exist."""
    if "ChurnProb" not in reference.columns or "Churned" not in reference.columns:
        logger.warning("ChurnProb or Churned column missing — skipping performance report.")
        return

    try:
        from evidently.report import Report
        from evidently.metric_preset import ClassificationPreset
        from evidently.pipeline.column_mapping import ColumnMapping
    except ImportError:
        logger.error("evidently not installed.")
        return

    cols = ["Recency", "Frequency", "Monetary", "Churned", "ChurnProb"]
    cols = [c for c in cols if c in reference.columns]
    ref  = reference[cols].copy()
    cur  = current[cols].copy()

    # Binarize prediction
    ref["prediction"] = (ref["ChurnProb"] >= 0.5).astype(int)
    cur["prediction"] = (cur["ChurnProb"] >= 0.5).astype(int)

    mapping = ColumnMapping(
        target="Churned",
        prediction="prediction",
        numerical_features=["Recency", "Frequency", "Monetary"],
    )

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "churn_performance_report.html"
    report.save_html(str(out))
    logger.success(f"Churn performance report → {out}")


def run(report_type: str = "all") -> None:
    setup_logger()
    logger.info("=" * 50)
    logger.info("STEP 6/6 — MONITORING (Evidently AI)")
    logger.info("=" * 50)

    reference, current = load_data()

    if report_type in ("all", "drift"):
        run_data_drift_report(reference, current)

    if report_type in ("all", "quality"):
        run_data_quality_report(reference, current)

    if report_type in ("all", "target"):
        run_target_drift_report(reference, current)

    if report_type in ("all", "performance"):
        run_churn_performance_report(reference, current)

    logger.success(f"Reports saved to: {REPORTS_DIR}")
    logger.info("Open .html files in browser to view interactive reports.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailPulse Monitoring")
    parser.add_argument(
        "--report",
        choices=["all", "drift", "quality", "target", "performance"],
        default="all",
        help="Which report to generate",
    )
    args = parser.parse_args()
    run(args.report)