"""Hashing utilities."""

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    """Generate stable hash of JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
