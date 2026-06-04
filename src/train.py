"""
Legacy training script — delegates to src/models/train.py.

This file is kept for backward compatibility (original entry point was `python src/train.py`).
New code should use `python src/models/train.py` or `make train`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import train  # noqa: F401

if __name__ == "__main__":
    train()
