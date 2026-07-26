from __future__ import annotations

import json
from pathlib import Path

from .benchmark import BenchmarkCase
from .models import RetrievalRecord


def load_records(path: str | Path) -> list[RetrievalRecord]:
    return [RetrievalRecord.model_validate(item) for item in _load_jsonl(path)]


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    return [BenchmarkCase.model_validate(item) for item in _load_jsonl(path)]


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Line {line_number} in {path} is not a JSON object")
        records.append(value)
    return records
