import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_root() -> Path:
    return _PROJECT_ROOT


def get_path(*parts: str) -> Path:
    """Return an absolute path joined from the project root."""
    return _PROJECT_ROOT.joinpath(*parts)


def get_config_path() -> Path:
    return get_path("configs", "model_config.yaml")
