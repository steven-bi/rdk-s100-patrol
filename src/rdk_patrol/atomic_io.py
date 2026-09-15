from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_write_text(path: str | Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
    )


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")) + "\n"
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(row)
        handle.flush()
        os.fsync(handle.fileno())

