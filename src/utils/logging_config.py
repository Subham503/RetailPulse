"""Centralized logging configuration using loguru."""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure loguru logger for the project."""
    logger.remove()  # remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler
    logger.add(sys.stdout, format=fmt, level=log_level, colorize=True)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=fmt,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )

    return logger
