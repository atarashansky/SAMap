"""Logging configuration for SAMap."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "samap") -> logging.Logger:
    """Get a logger instance for SAMap.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "samap"

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Auto-configure if no handlers exist
    if not logger.handlers and not logger.parent.handlers if logger.parent else not logger.handlers:
        _setup_default_handler(logger)

    return logger


def _setup_default_handler(logger: logging.Logger) -> None:
    """Set up default handler for a logger."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def setup_logging(
    level: int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = logging.INFO,
    format_string: str | None = None,
) -> None:
    """Configure logging for SAMap.

    Parameters
    ----------
    level : int or str, optional
        Logging level, by default logging.INFO
    format_string : str, optional
        Custom format string, by default None (uses default format)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    fmt = format_string or _LOG_FORMAT
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))

    logger = get_logger()
    logger.addHandler(handler)
    logger.setLevel(level)


# Create and configure module-level logger
logger = get_logger()
