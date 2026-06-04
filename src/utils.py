"""
Legacy utility shim — kept for backward compatibility.

New code should import from src.utils.paths instead:
    from src.utils.paths import get_path
"""
from src.utils.paths import get_path  # noqa: F401
