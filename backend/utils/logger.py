"""Logging utility for consistent application logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    Args:
        name: Logger name.
    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=500_000, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name.
    Returns:
        Logger instance.
    """
    return setup_logger(name)
