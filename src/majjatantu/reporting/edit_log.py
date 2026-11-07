"""Persistent log of every candidate edit (accepted and rejected)."""

from __future__ import annotations

import json
from pathlib import Path

from majjatantu.types import EditLogEntry


class EditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[EditLogEntry] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        self._entries = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(EditLogEntry.model_validate_json(line))

    def append(self, entry: EditLogEntry) -> None:
        self._entries.append(entry)
        with self.path.open("a") as f:
            f.write(entry.model_dump_json() + "\n")

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[EditLogEntry]:
        return list(self._entries)

    def to_json(self, out: str | Path) -> None:
        Path(out).write_text(
            json.dumps([e.model_dump(mode="json") for e in self._entries], indent=2)
        )
