from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def configure_logging(
    project_root: str | Path,
    *,
    logging_config: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> None:
    root = Path(project_root).resolve()
    path = (
        Path(logging_config).resolve()
        if logging_config is not None
        else root / "configs" / "logging.yaml"
    )
    if not path.is_file():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"logging configuration must be a mapping: {path}")
    handlers = payload.get("handlers")
    if isinstance(handlers, dict):
        for handler in handlers.values():
            if not isinstance(handler, dict) or not handler.get("filename"):
                continue
            filename = Path(str(handler["filename"]))
            if data_dir is not None:
                filename = Path(data_dir).resolve() / "logs" / filename.name
            elif not filename.is_absolute():
                filename = root / filename
            filename.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(filename)
    logging.config.dictConfig(payload)  # type: ignore[arg-type]
