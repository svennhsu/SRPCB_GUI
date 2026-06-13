from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CudaMemoryInfo:
    available: bool
    allocated_mib: float | None = None
    reserved_mib: float | None = None
    free_mib: float | None = None
    total_mib: float | None = None
    gpu_name: str | None = None
    error: str | None = None

    def label(self) -> str:
        if not self.available:
            return f"CUDA memory: unavailable{f' ({self.error})' if self.error else ''}"
        parts = [
            f"allocated={self.allocated_mib:.1f} MiB" if self.allocated_mib is not None else None,
            f"reserved={self.reserved_mib:.1f} MiB" if self.reserved_mib is not None else None,
            f"free={self.free_mib:.1f} MiB" if self.free_mib is not None else None,
            f"total={self.total_mib:.1f} MiB" if self.total_mib is not None else None,
        ]
        return "CUDA memory: " + " ".join(p for p in parts if p)


@dataclass(frozen=True)
class DetectionVramPlan:
    requested_mode: str
    effective_mode: str
    strategy_name: str
    reason: str
    approximate: bool
    use_cpu_pass_a_baseline: bool
    tile_size_b: int = 1024
    overlap_b: float = 0.60
    tile_size_c: int = 512
    overlap_c: float = 0.60
    estimated_tile_count_b: int = 0
    estimated_tile_count_c: int = 0
    image_size: tuple[int, int] | None = None
    cuda_before: CudaMemoryInfo | None = None


def image_size(path: str | Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def cuda_memory_info(torch_module, device: str) -> CudaMemoryInfo:
    if torch_module is None or not str(device).startswith("cuda"):
        return CudaMemoryInfo(available=False)
    try:
        if not torch_module.cuda.is_available():
            return CudaMemoryInfo(available=False, error="torch.cuda.is_available() is false")
        allocated = torch_module.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch_module.cuda.memory_reserved(0) / (1024 * 1024)
        free = total = None
        if hasattr(torch_module.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
            free = free_bytes / (1024 * 1024)
            total = total_bytes / (1024 * 1024)
        return CudaMemoryInfo(
            available=True,
            allocated_mib=allocated,
            reserved_mib=reserved,
            free_mib=free,
            total_mib=total,
            gpu_name=torch_module.cuda.get_device_name(0),
        )
    except Exception as exc:
        return CudaMemoryInfo(available=False, error=str(exc))


def estimate_tile_count(width: int, height: int, tile_size: int, overlap: float) -> int:
    if width <= 0 or height <= 0:
        return 0
    stride = max(1, int(tile_size * (1 - overlap)))
    x_count = 1 if width <= tile_size else math.ceil((width - tile_size) / stride) + 1
    y_count = 1 if height <= tile_size else math.ceil((height - tile_size) / stride) + 1
    return x_count * y_count


def _pass_a_gpu_risk(size: tuple[int, int] | None, mem: CudaMemoryInfo) -> tuple[bool, str]:
    if size is None:
        return False, "image size unavailable; running normal GPU only V4"
    width, height = size
    megapixels = (width * height) / 1_000_000
    max_side = max(width, height)
    total_mib = mem.total_mib or 0
    free_mib = mem.free_mib or 0

    # Scale-locked full-frame Pass A can spike memory above the raw tensor.
    # Pass B/C use bounded tiles, so the high-risk part is Pass A.
    if total_mib and total_mib <= 7000 and (megapixels >= 3.0 or max_side >= 2400):
        return True, f"VRAM risk: {width}x{height} ({megapixels:.2f} MP) on {total_mib:.0f} MiB GPU"
    if free_mib and free_mib < 1800 and megapixels >= 1.5:
        return True, f"low free CUDA memory ({free_mib:.0f} MiB) for {width}x{height} Pass A"
    if megapixels >= 5.0 or max_side >= 4096:
        return True, f"large image Pass A risk: {width}x{height} ({megapixels:.2f} MP)"
    return False, f"normal GPU only V4 selected: {width}x{height} ({megapixels:.2f} MP)"


def build_detection_plan(
    *,
    image_path: str | Path,
    requested_mode: str,
    device: str,
    torch_module,
) -> DetectionVramPlan:
    size = image_size(image_path)
    mem = cuda_memory_info(torch_module, device)
    width, height = size or (0, 0)
    count_b = estimate_tile_count(width, height, 1024, 0.60)
    count_c = estimate_tile_count(width, height, 512, 0.60)

    if requested_mode != "CPU + GPU" or not str(device).startswith("cuda"):
        return DetectionVramPlan(
            requested_mode=requested_mode,
            effective_mode=requested_mode,
            strategy_name="reference_v4",
            reason="normal GPU only V4 path",
            approximate=False,
            use_cpu_pass_a_baseline=False,
            estimated_tile_count_b=count_b,
            estimated_tile_count_c=count_c,
            image_size=size,
            cuda_before=mem,
        )

    risky, reason = _pass_a_gpu_risk(size, mem)
    if risky:
        return DetectionVramPlan(
            requested_mode=requested_mode,
            effective_mode="CPU + GPU",
            strategy_name="cpu_pass_a_reference_gpu_tiles",
            reason=reason,
            approximate=False,
            use_cpu_pass_a_baseline=True,
            estimated_tile_count_b=count_b,
            estimated_tile_count_c=count_c,
            image_size=size,
            cuda_before=mem,
        )

    return DetectionVramPlan(
        requested_mode=requested_mode,
        effective_mode="CPU + GPU",
        strategy_name="reference_v4",
        reason=reason,
        approximate=False,
        use_cpu_pass_a_baseline=False,
        estimated_tile_count_b=count_b,
        estimated_tile_count_c=count_c,
        image_size=size,
        cuda_before=mem,
    )


def choose_sr_batch_size(torch_module, device: str, default_batch_size: int = 64) -> tuple[int, str, CudaMemoryInfo]:
    mem = cuda_memory_info(torch_module, device)
    if not str(device).startswith("cuda") or not mem.available:
        return default_batch_size, "CPU or CUDA unavailable; using default SR batch size", mem
    free = mem.free_mib or 0
    if free >= 3500:
        return default_batch_size, f"{free:.0f} MiB free; using default SR batch size", mem
    if free >= 2200:
        return min(default_batch_size, 32), f"{free:.0f} MiB free; reducing SR batch size to 32", mem
    return min(default_batch_size, 16), f"{free:.0f} MiB free; reducing SR batch size to 16", mem


def _model_device(model) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cpu"


def make_cpu_pass_a_baseline(original_baseline, torch_module):
    def _cpu_pass_a_baseline(model, device, img, score_threshold=0.5):
        original_device = _model_device(model)
        logger.warning(
            "CPU + GPU strategy: running V4 Pass A baseline on CPU, then returning model to %s. "
            "Scale lock, thresholds, and NMS are preserved.",
            original_device,
        )
        if str(original_device).startswith("cuda"):
            try:
                torch_module.cuda.empty_cache()
            except Exception as exc:
                logger.debug("CUDA cache cleanup before CPU Pass A failed: %s", exc)
        started = time.perf_counter()
        model.to("cpu")
        try:
            boxes, scores, labels = original_baseline(model, "cpu", img, score_threshold=score_threshold)
        finally:
            model.to(original_device)
            if str(original_device).startswith("cuda"):
                try:
                    torch_module.cuda.empty_cache()
                except Exception as exc:
                    logger.debug("CUDA cache cleanup after CPU Pass A failed: %s", exc)
        logger.info("CPU + GPU strategy: CPU Pass A complete in %.2fs", time.perf_counter() - started)
        return boxes, scores, labels

    return _cpu_pass_a_baseline


@contextmanager
def apply_detection_plan(inference_v4_module, plan: DetectionVramPlan, torch_module) -> Iterator[None]:
    original = inference_v4_module._native_baseline_inference
    if plan.use_cpu_pass_a_baseline:
        inference_v4_module._native_baseline_inference = make_cpu_pass_a_baseline(original, torch_module)
    try:
        yield
    finally:
        inference_v4_module._native_baseline_inference = original
