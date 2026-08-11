"""
Logging utilities for research pipeline.

Provides structured logging to both console and file with configurable levels.

Named ``logging_utils`` rather than ``logging``: a module named ``logging.py``
in this package shadows the standard library when a script inside utils/ is run
directly, which breaks the pandas import chain.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up structured logging for an experiment.

    Parameters
    ----------
    log_dir : str
        Directory where log files will be saved.
    experiment_name : str
        Name of the experiment (used in log filename).
    level : int, default=logging.INFO
        Logging level.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    # Create log directory if needed
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Format for logs
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_dict(logger: logging.Logger, data: dict, prefix: str = "") -> None:
    """
    Log a dictionary with clear formatting.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    data : dict
        Dictionary to log.
    prefix : str, optional
        String to prefix each line with.
    """
    for key, value in data.items():
        logger.info(f"{prefix}{key}: {value}")
