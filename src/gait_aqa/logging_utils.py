"""Logging setup for CLI workflows."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


class _StdlibLoguruAdapter:
    """Tiny adapter that accepts Loguru-style `{}` formatting."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, message: str, *args: Any) -> None:
        self._logger.debug(self._format(message, args))

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(self._format(message, args))

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(self._format(message, args))

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(self._format(message, args))

    @staticmethod
    def _format(message: str, args: tuple[Any, ...]) -> str:
        return message.format(*args) if args else message


def setup_logging(
    log_file: str | Path = "output/logs/gait_aqa.log",
    level: str = "INFO",
) -> Any:
    """Configure console and file logging.

    Loguru is used when installed. A small standard-library fallback keeps the
    project runnable in minimal environments before dependencies are synced.
    """
    output = Path(log_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    level_name = level.upper()

    try:
        from loguru import logger  # type: ignore
    except ModuleNotFoundError:
        logging.basicConfig(
            level=getattr(logging, level_name, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr),
                logging.FileHandler(output, encoding="utf-8"),
            ],
            force=True,
        )
        return _StdlibLoguruAdapter(logging.getLogger("gait_aqa"))

    logger.remove()
    logger.add(
        sys.stderr,
        level=level_name,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    logger.add(
        output,
        level=level_name,
        rotation="5 MB",
        retention=10,
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.info("Logging initialized: {}", output)
    return logger


def get_logger(name: str = "gait_aqa") -> Any:
    """Return a logger compatible with Loguru's basic methods."""
    try:
        from loguru import logger  # type: ignore
    except ModuleNotFoundError:
        return _StdlibLoguruAdapter(logging.getLogger(name))
    return logger
