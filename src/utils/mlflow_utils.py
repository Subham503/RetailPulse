"""MLflow setup and helper utilities."""

import mlflow
from loguru import logger
from src.utils.config import MLFLOW_EXPERIMENT, ROOT_DIR


def setup_mlflow() -> None:
    """Initialize MLflow with SQLite backend."""
    db_path = ROOT_DIR / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow → uri: {tracking_uri}  experiment: {MLFLOW_EXPERIMENT}")


def log_dataframe_info(df, name: str = "dataset") -> None:
    """Log basic DataFrame metadata as MLflow params."""
    mlflow.log_param(f"{name}_rows", len(df))
    mlflow.log_param(f"{name}_cols", len(df.columns))