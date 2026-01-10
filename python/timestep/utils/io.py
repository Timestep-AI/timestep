"""I/O utilities."""

import json
import time
from pathlib import Path
from typing import Any


def now() -> float:
    """Get current timestamp."""
    return time.time()


def clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, x))


def write_json(path: Path, obj: Any) -> None:
    """Write JSON object to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
