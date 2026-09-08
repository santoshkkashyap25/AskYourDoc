"""Logging configuration module for file and console logging."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config import settings


def setup_logging() -> logging.Logger:
    """Configures structured file and console logging."""
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    app_log_file = settings.LOG_DIR / "app.log"
    error_log_file = settings.LOG_DIR / "error.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates on reload
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating App Log File (10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        filename=str(app_log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Rotating Error Log File (5MB max, 3 backups)
    error_handler = RotatingFileHandler(
        filename=str(error_log_file),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Suppress verbose loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    logger = logging.getLogger("AskYourDoc")
    logger.info("Logging initialized. App log: %s, Error log: %s", app_log_file, error_log_file)
    return logger


logger = setup_logging()
