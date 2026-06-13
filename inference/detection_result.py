from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Detection:
    box: list[float]
    score: float
    label: int
    class_name: str


@dataclass(slots=True)
class DetectionResult:
    image_path: str
    timestamp: str
    boxes: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    confidence_threshold: float = 0.5
    device: str = "cpu"
    processing_time: float = 0.0
    inference_mode: str = "GPU only"
    pipeline_source: str = ""
    checkpoint_path: str = ""
    effective_image_path: str = ""
    effective_image_size: tuple[int, int] | None = None
    detection_source_label: str = "Original Image"
    vram_strategy: str = "reference_v4"
    vram_strategy_reason: str = ""
    vram_strategy_approximate: bool = False
    fallback_occurred: bool = False
    tile_settings: dict[str, int | float] = field(default_factory=dict)
    cuda_memory_before: str = ""
    cuda_memory_after: str = ""
    status: str = "success"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    annotated_image_path: str | None = None
    sr_image_path: str | None = None
    sr_detection_proxy_path: str | None = None
    detection_source: str = "original"  # "original" or "sr_restored"

    @property
    def filename(self) -> str:
        return Path(self.image_path).name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detections"] = [asdict(det) for det in self.detections]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionResult":
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        payload["detections"] = [Detection(**det) for det in payload.get("detections", [])]
        return cls(**payload)
