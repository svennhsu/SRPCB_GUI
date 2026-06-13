from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.paths import HISTORY_FILE
from inference.detection_result import DetectionResult


@dataclass(slots=True)
class HistoryRecord:
    id: str
    result: DetectionResult
    pinned: bool = False

    @property
    def image_path(self) -> str:
        return self.result.image_path

    @property
    def filename(self) -> str:
        return self.result.filename

    @property
    def timestamp(self) -> str:
        return self.result.timestamp

    @property
    def total_count(self) -> int:
        return self.result.total_count

    @property
    def status(self) -> str:
        return self.result.status

    def to_dict(self) -> dict:
        return {"id": self.id, "pinned": self.pinned, "result": self.result.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryRecord":
        return cls(
            id=data["id"],
            pinned=bool(data.get("pinned", False)),
            result=DetectionResult.from_dict(data["result"]),
        )


class HistoryManager:
    def __init__(self, storage_path: str | Path = HISTORY_FILE, max_unpinned: int = 100) -> None:
        self.storage_path = Path(storage_path)
        self.max_unpinned = max_unpinned
        self._records: list[HistoryRecord] = []
        self.load()

    def add(self, result: DetectionResult) -> HistoryRecord:
        record = HistoryRecord(id=uuid.uuid4().hex, result=result)
        self._records.append(record)
        self._trim_unpinned()
        self.save()
        return record

    def get(self, record_id: str) -> HistoryRecord | None:
        for record in self._records:
            if record.id == record_id:
                return record
        return None

    def remove(self, record_id: str) -> bool:
        before = len(self._records)
        self._records = [record for record in self._records if record.id != record_id]
        changed = len(self._records) != before
        if changed:
            self.save()
        return changed

    def toggle_pin(self, record_id: str) -> bool:
        record = self.get(record_id)
        if record is None:
            return False
        record.pinned = not record.pinned
        self.save()
        return record.pinned

    def clear_unpinned(self) -> int:
        before = len(self._records)
        self._records = [record for record in self._records if record.pinned]
        removed = before - len(self._records)
        if removed:
            self.save()
        return removed

    def records(self) -> list[HistoryRecord]:
        return sorted(self._records, key=lambda rec: (rec.pinned, rec.timestamp), reverse=True)

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            loaded = [HistoryRecord.from_dict(item) for item in data.get("records", [])]
            self._records = [
                record
                for record in loaded
                if Path(record.image_path).exists()
                and (record.result.annotated_image_path is None or Path(record.result.annotated_image_path).exists())
            ]
        except Exception:
            self._records = []

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in self._records]}
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _trim_unpinned(self) -> None:
        unpinned = [record for record in self._records if not record.pinned]
        overflow = len(unpinned) - self.max_unpinned
        if overflow <= 0:
            return
        remove_ids = {record.id for record in unpinned[:overflow]}
        self._records = [record for record in self._records if record.id not in remove_ids]
