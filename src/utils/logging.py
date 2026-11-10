"""Logging setup for disaster AI audit experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure the root logger for the project.

    Call once at the start of a script. Subsequent calls are no-ops.

    Parameters
    ----------
    level : str
        Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    log_file : str | Path | None
        If given, also write logs to this file.
    fmt : str
        Log format string.
    datefmt : str
        Date format string.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers
    root.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "urllib3", "matplotlib", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
