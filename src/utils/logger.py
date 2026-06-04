import logging
import os
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a module-level logger with a consistent format.
    Writes to stdout and, if logs/ directory exists, to logs/training.log.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — only if logs/ dir exists or can be created
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    try:
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # read-only filesystem — console only

    return logger
