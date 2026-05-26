from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[Dict]:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_description(record: Dict, language: str) -> str:
    if language == "zh":
        return str(record.get("zh_description") or record.get("descriptions_zh") or "")
    return str(record.get("en_description") or record.get("descriptions_en") or "")

