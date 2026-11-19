"""Local structured logs with recursive credential redaction."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from torch.utils.tensorboard import SummaryWriter

_SECRET_FRAGMENTS = (
    "authorization",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
)


def redact_secrets(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_secrets(item) for item in value]
    return value


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def __enter__(self) -> JsonlLogger:
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def log(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        record = redact_secrets({"event": event, **fields})
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __exit__(self, *_: object) -> None:
        self.close()


class CsvLogger:
    """Append rows with one declared, stable field order."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

    def log(self, row: Mapping[str, Any]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.fieldnames, extrasaction="raise"
            )
            if not exists:
                writer.writeheader()
            writer.writerow(redact_secrets(dict(row)))


class TensorBoardLogger:
    def __init__(self, directory: Path) -> None:
        self.writer = SummaryWriter(log_dir=str(directory))

    def scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        self.writer.close()

