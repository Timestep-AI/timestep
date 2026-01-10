"""JSONL file I/O utilities."""

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Read JSONL file, yielding one JSON object per line."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {e}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of JSON objects to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
