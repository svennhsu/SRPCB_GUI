from __future__ import annotations

import logging
import sys
import time
import types
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from app.paths import ANNOTATED_CACHE, DETECTION_CHECKPOINT, DETECTION_DIR
from inference.detection_result import Detection, DetectionResult
from inference.vram_strategy import apply_detection_plan, build_detection_plan, cuda_memory_info
from utils.annotation import draw_annotations, save_annotated_image

logger = logging.getLogger(__name__)

PIPELINE_SOURCE = "detection/"

if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

try:
    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        pyplot_stub = types.ModuleType("matplotlib.pyplot")
        pyplot_stub.figure = lambda *args, **kwargs: None
        pyplot_stub.bar = lambda *args, **kwargs: None
        pyplot_stub.xlabel = lambda *args, **kwargs: None
        pyplot_stub.ylabel = lambda *args, **kwargs: None
        pyplot_stub.title = lambda *args, **kwargs: None
        pyplot_stub.savefig = lambda *args, **kwargs: None
        pyplot_stub.close = lambda *args, **kwargs: None
        sys.modules["matplotlib"] = matplotlib_stub
        sys.modules["matplotlib.pyplot"] = pyplot_stub

    import torch
    import torchvision

    from pretrained_model import get_pretrained_fasterrcnn_v4
    from dataset_v4 import CLASS_MAPPING, ID_TO_CLASS
    import inference_v4

    _IMPORT_ERROR = None
except Exception as exc:
    torch = None
    CLASS_MAPPING = {
        "background": 0,
        "capacitor": 1,
        "resistor": 2,
        "ic": 3,
        "connector": 4,
        "led": 5,
        "transistor": 6,
        "diode": 7,
    }
    ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
    _IMPORT_ERROR = str(exc)


CLASS_NAMES = [name for name, _ in sorted(CLASS_MAPPING.items(), key=lambda item: item[1])]


class AOIInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str | Path = DETECTION_CHECKPOINT,
        device_preference: Literal["auto", "cuda", "cpu"] = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_preference = device_preference
        self._device = "cpu"
        self._device_label = "CPU fallback"
        self._model = None
        self._loaded = False
        self._load_error: str | None = _IMPORT_ERROR
        self._warnings: list[str] = []
        self._cuda_block_reason: str | None = None
        self.checkpoint_f1: float | None = None
        self.checkpoint_epoch: int | None = None

        logger.info(
            "AOIInferenceEngine — pipeline: %s  checkpoint: %s",
            PIPELINE_SOURCE, self.checkpoint_path,
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def cuda_block_reason(self) -> str | None:
        return self._cuda_block_reason

    @property
    def pipeline_source(self) -> str:
        return PIPELINE_SOURCE

    @property
    def checkpoint_name(self) -> str:
        return self.checkpoint_path.name

    @property
    def pipeline_info(self) -> str:
        return (
            f"Pipeline: {PIPELINE_SOURCE}\n"
            f"Mode: CPU + GPU by default\n"
            f"Checkpoint: {self.checkpoint_path.name}"
        )

    def _cuda_memory_snapshot(self) -> str:
        return cuda_memory_info(torch, self._device).label()

    def _cuda_block_message(self, detail: str) -> str:
        try:
            gpu = torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else "unknown GPU"
            cap = torch.cuda.get_device_capability(0) if torch is not None and torch.cuda.is_available() else (0, 0)
            cap_str = f" (compute {cap[0]}.{cap[1]})" if cap[0] else ""
        except Exception:
            gpu = "unknown GPU"
            cap_str = ""
        return (
            f"CUDA is blocked: the installed PyTorch cannot run on {gpu}{cap_str}.\n\n"
            f"Detail: {detail}\n\n"
            "Re-install PyTorch with a CUDA build that supports this GPU: "
            "https://pytorch.org/get-started/locally/"
        )

    def _validate_cuda_runtime(self) -> tuple[bool, str | None]:
        if torch is None:
            return False, f"PyTorch import failed: {_IMPORT_ERROR}"
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() is false."

        try:
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
            probe = torch.ones((8, 8), device="cuda")
            _ = float(probe.sum().item())
            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], device="cuda")
            scores = torch.tensor([0.9, 0.8], device="cuda")
            _ = torchvision.ops.nms(boxes, scores, 0.5)
            torch.cuda.synchronize()
            self._device_label = f"CUDA: {name} (compute {capability[0]}.{capability[1]})"
            logger.info("CUDA compatibility OK: %s capability=%s arch_list=%s", name, capability, arch_list)
            return True, None
        except Exception as exc:
            return False, self._cuda_block_message(str(exc))

    def _select_device(self, preference: str | None = None) -> str:
        if torch is None:
            self._warnings.append(f"PyTorch import failed: {_IMPORT_ERROR}")
            return "cpu"

        pref = preference or self.device_preference
        if pref == "cpu":
            self._cuda_block_reason = None
            self._device_label = "CPU fallback"
            return "cpu"

        if pref in ("auto", "cuda"):
            ok, reason = self._validate_cuda_runtime()
            if ok:
                self._cuda_block_reason = None
                return "cuda:0"
            self._cuda_block_reason = reason
            if pref == "cuda" or torch.cuda.is_available():
                raise RuntimeError(reason or "CUDA runtime validation failed.")
            message = "CUDA is not available in this Python environment; using CPU."
            self._warnings.append(message)

        self._device_label = "CPU fallback"
        return "cpu"

    def load(self, device_preference: str | None = None) -> bool:
        if torch is None:
            self._load_error = f"PyTorch/torchvision import failed: {_IMPORT_ERROR}"
            logger.error(self._load_error)
            return False
        if self._loaded:
            return True
        if not self.checkpoint_path.exists():
            self._load_error = f"Checkpoint not found: {self.checkpoint_path}"
            logger.error(self._load_error)
            return False

        try:
            self._device = self._select_device(device_preference)
            logger.info(
                "Loading model from %s on %s",
                self.checkpoint_path, self._device,
            )
            logger.info("Pipeline source: %s", PIPELINE_SOURCE)
            logger.info("Checkpoint: %s", self.checkpoint_path.name)
            model = get_pretrained_fasterrcnn_v4(num_classes=len(CLASS_NAMES))
            ckpt = torch.load(str(self.checkpoint_path), map_location=self._device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.to(self._device)
            model.eval()
            self._model = model
            self._loaded = True
            self._load_error = None
            self.checkpoint_f1 = ckpt.get("f1")
            self.checkpoint_epoch = ckpt.get("epoch")
            logger.info(
                "Model loaded successfully — device=%s pipeline=%s checkpoint=%s F1=%s epoch=%s",
                self._device_label, PIPELINE_SOURCE, self.checkpoint_path.name,
                self.checkpoint_f1, self.checkpoint_epoch,
            )
            return True
        except RuntimeError as exc:
            if self._cuda_block_reason:
                self._load_error = self._cuda_block_reason
                logger.error("CUDA blocked: %s", self._load_error)
            elif self._device.startswith("cuda"):
                self._warnings.append(f"CUDA model load failed: {exc}. Retrying on CPU.")
                logger.exception("CUDA model load failed; retrying CPU")
                self._device = "cpu"
                self._device_label = "CPU fallback"
                try:
                    model = get_pretrained_fasterrcnn_v4(num_classes=len(CLASS_NAMES))
                    ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"], strict=True)
                    model.eval()
                    self._model = model
                    self._loaded = True
                    self._load_error = None
                    self.checkpoint_f1 = ckpt.get("f1")
                    self.checkpoint_epoch = ckpt.get("epoch")
                    return True
                except Exception as cpu_exc:
                    self._load_error = f"Model load failed on CUDA and CPU: {cpu_exc}"
            else:
                self._load_error = f"Model load failed: {exc}"
        except Exception as exc:
            self._load_error = f"Model load failed: {exc}"

        if self._cuda_block_reason:
            logger.error("Failed to load model because CUDA is blocked for this environment.")
        else:
            logger.exception("Failed to load model")
        return False

    THRESHOLD_TILE_PRE = 0.25
    THRESHOLD_TILE_PRE_C = 0.92
    THRESHOLD_TILE_PRE_C_SMD = 0.98
    THRESHOLD_NMS_IOU = 0.40
    THRESHOLD_NMS_IOS = 0.70
    THRESHOLD_CONF_CAP = 0.96
    THRESHOLD_CONF_RES = 0.90
    THRESHOLD_CONF_IC_CON = 0.95
    THRESHOLD_CONF_SMD = 0.55

    def infer(
        self,
        image_path: str | Path,
        confidence_threshold: float = 0.5,
        pass_a_only: bool = True,
        use_vram_strategy: bool = False,
    ) -> DetectionResult:
        """Run V4 inference.

        Default behavior is reference-quality: call AOI_v1/v4/20260519-1400
        without monkey-patching scale_lock, tiling, thresholds, NMS, or merge
        behavior.  CPU + GPU mode may move Pass A to CPU for high-risk CUDA
        images while preserving V4 scale_lock and Pass B/C tile behavior.
        """
        start = time.perf_counter()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = Path(image_path)
        if use_vram_strategy:
            inference_mode = "CPU + GPU"
        elif pass_a_only:
            inference_mode = "Fast Preview"
        else:
            inference_mode = "GPU only"

        try:
            with Image.open(image_path) as image_info:
                input_size = image_info.size
        except OSError as exc:
            logger.warning("Could not read image size for %s: %s", image_path, exc)
            input_size = None
        runtime_warnings: list[str] = []
        plan = build_detection_plan(
            image_path=image_path,
            requested_mode=inference_mode,
            device=self._device,
            torch_module=torch,
        )
        tile_settings = {
            "pass_b_tile_size": plan.tile_size_b,
            "pass_b_overlap": plan.overlap_b,
            "pass_b_estimated_tile_count": plan.estimated_tile_count_b,
            "pass_c_tile_size": plan.tile_size_c,
            "pass_c_overlap": plan.overlap_c,
            "pass_c_estimated_tile_count": plan.estimated_tile_count_c,
        }

        if not self._loaded or self._model is None:
            return DetectionResult(
                image_path=str(image_path),
                timestamp=timestamp,
                confidence_threshold=confidence_threshold,
                device=self.device_label,
                status="error",
                inference_mode=inference_mode,
                pipeline_source=PIPELINE_SOURCE,
                checkpoint_path=str(self.checkpoint_path),
                effective_image_path=str(image_path),
                effective_image_size=input_size,
                vram_strategy=plan.strategy_name,
                vram_strategy_reason=plan.reason,
                vram_strategy_approximate=plan.approximate,
                fallback_occurred=plan.use_cpu_pass_a_baseline,
                tile_settings=tile_settings,
                cuda_memory_before=plan.cuda_before.label() if plan.cuda_before else "",
                cuda_memory_after=self._cuda_memory_snapshot(),
                error=self._load_error or "Model is not loaded.",
                warnings=self.warnings,
            )

        try:
            if self._device.startswith("cuda"):
                ok, reason = self._validate_cuda_runtime()
                if not ok:
                    raise RuntimeError(reason or "CUDA runtime validation failed.")

            logger.info(
                "Infer start — mode=%s pipeline=%s checkpoint=%s input=%s size=%s "
                "effective_input=%s effective_size=%s device=%s",
                inference_mode,
                PIPELINE_SOURCE, self.checkpoint_path.name,
                image_path, input_size, image_path, input_size, self._device_label,
            )
            logger.info("%s", plan.cuda_before.label() if plan.cuda_before else self._cuda_memory_snapshot())
            logger.info(
                "Execution plan — strategy=%s approximate=%s fallback=%s reason=%s",
                plan.strategy_name,
                plan.approximate,
                plan.use_cpu_pass_a_baseline,
                plan.reason,
            )
            logger.info(
                "V4 thresholds — tile_pre=%.2f tile_pre_c=%.2f tile_pre_c_smd=%.2f "
                "nms_iou=%.2f nms_ios=%.2f "
                "conf_cap=%.2f conf_res=%.2f conf_ic_con=%.2f conf_smd=%.2f "
                "class_map=%d classes",
                self.THRESHOLD_TILE_PRE, self.THRESHOLD_TILE_PRE_C,
                self.THRESHOLD_TILE_PRE_C_SMD,
                self.THRESHOLD_NMS_IOU, self.THRESHOLD_NMS_IOS,
                self.THRESHOLD_CONF_CAP, self.THRESHOLD_CONF_RES,
                self.THRESHOLD_CONF_IC_CON, self.THRESHOLD_CONF_SMD,
                len(CLASS_NAMES),
            )
            if hasattr(self._model, "transform"):
                logger.info(
                    "Model transform before inference — min_size=%s max_size=%s",
                    getattr(self._model.transform, "min_size", None),
                    getattr(self._model.transform, "max_size", None),
                )
            logger.info(
                "V4 geometry — Pass B tile=1024 overlap=0.60 estimated_count=%d; "
                "Pass C tile=512 overlap=0.60 estimated_count=%d; "
                "merge_strategy=dense; scale_lock=reference default"
                % (plan.estimated_tile_count_b, plan.estimated_tile_count_c)
            )

            active_plan = plan

            def run_v4_with_plan(run_plan):
                if run_plan.use_cpu_pass_a_baseline:
                    logger.warning(
                        "CPU + GPU active — Pass A baseline will run on CPU with reference scale_lock. "
                        "Pass B/C remain on %s with V4 reference tile/merge settings.",
                        self._device,
                    )
                with apply_detection_plan(inference_v4, run_plan, torch):
                    with torch.inference_mode():
                        return inference_v4.run_v4_inference_native(
                            str(image_path),
                            self._model,
                            self._device,
                            tile_pre_threshold=self.THRESHOLD_TILE_PRE,
                            tile_pre_threshold_c=self.THRESHOLD_TILE_PRE_C,
                            tile_pre_threshold_c_smd=self.THRESHOLD_TILE_PRE_C_SMD,
                            nms_iou_threshold=self.THRESHOLD_NMS_IOU,
                            nms_ios_threshold=self.THRESHOLD_NMS_IOS,
                            conf_cap=self.THRESHOLD_CONF_CAP,
                            conf_res=self.THRESHOLD_CONF_RES,
                            conf_ic_con=self.THRESHOLD_CONF_IC_CON,
                            conf_smd=self.THRESHOLD_CONF_SMD,
                            pass_a_only=pass_a_only,
                        )

            try:
                boxes, scores, labels = run_v4_with_plan(active_plan)
            except RuntimeError as exc:
                if (
                    use_vram_strategy
                    and not active_plan.use_cpu_pass_a_baseline
                    and self._device.startswith("cuda")
                    and "out of memory" in str(exc).lower()
                ):
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cleanup_exc:
                        logger.debug("CUDA cache cleanup after detection OOM failed: %s", cleanup_exc)
                    active_plan = replace(
                        active_plan,
                        strategy_name="cpu_pass_a_reference_gpu_tiles",
                        reason="CUDA OOM during GPU only V4; retrying with CPU Pass A reference baseline",
                        use_cpu_pass_a_baseline=True,
                    )
                    plan = active_plan
                    tile_settings = {
                        "pass_b_tile_size": active_plan.tile_size_b,
                        "pass_b_overlap": active_plan.overlap_b,
                        "pass_b_estimated_tile_count": active_plan.estimated_tile_count_b,
                        "pass_c_tile_size": active_plan.tile_size_c,
                        "pass_c_overlap": active_plan.overlap_c,
                        "pass_c_estimated_tile_count": active_plan.estimated_tile_count_c,
                    }
                    runtime_warnings.append(
                        "GPU only inference exceeded GPU memory; switched to CPU Pass A strategy. "
                        "Reference scale_lock and V4 tile/merge behavior were preserved."
                    )
                    boxes, scores, labels = run_v4_with_plan(active_plan)
                else:
                    raise

            if plan.use_cpu_pass_a_baseline:
                runtime_warnings.append(
                    "CPU + GPU used CPU Pass A baseline with reference scale_lock to avoid GPU OOM. "
                    "This preserves V4 scale lock and tile/merge behavior."
                )

        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                if torch is not None and self._device.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception as cleanup_exc:
                        logger.debug("CUDA cache cleanup after inference failure failed: %s", cleanup_exc)
                message = (
                    "CUDA out of memory during inference.\n\n"
                    "Options:\n"
                    "  • Switch to CPU fallback (Settings → Device → CPU)\n"
                    "  • Use CPU + GPU mode to move Pass A to CPU while preserving V4 behavior\n"
                    "  • If detecting on SR output, use SR-Restored Image (Original Size)\n"
                    "  • Close other GPU applications\n\n"
                    "GPU only V4 parameters were not silently changed."
                )
            elif "no kernel image is available" in message.lower():
                self._cuda_block_reason = self._cuda_block_message(message)
                message = self._cuda_block_reason
            logger.exception("Inference failed")
            return DetectionResult(
                image_path=str(image_path),
                timestamp=timestamp,
                confidence_threshold=confidence_threshold,
                device=self.device_label,
                processing_time=time.perf_counter() - start,
                inference_mode=inference_mode,
                pipeline_source=PIPELINE_SOURCE,
                checkpoint_path=str(self.checkpoint_path),
                effective_image_path=str(image_path),
                effective_image_size=input_size,
                vram_strategy=plan.strategy_name,
                vram_strategy_reason=plan.reason,
                vram_strategy_approximate=plan.approximate,
                fallback_occurred=plan.use_cpu_pass_a_baseline,
                tile_settings=tile_settings,
                cuda_memory_before=plan.cuda_before.label() if plan.cuda_before else "",
                cuda_memory_after=self._cuda_memory_snapshot(),
                status="error",
                error=message,
                warnings=[*self.warnings, *runtime_warnings],
            )
        except Exception as exc:
            logger.exception("Inference failed")
            return DetectionResult(
                image_path=str(image_path),
                timestamp=timestamp,
                confidence_threshold=confidence_threshold,
                device=self.device_label,
                processing_time=time.perf_counter() - start,
                inference_mode=inference_mode,
                pipeline_source=PIPELINE_SOURCE,
                checkpoint_path=str(self.checkpoint_path),
                effective_image_path=str(image_path),
                effective_image_size=input_size,
                vram_strategy=plan.strategy_name,
                vram_strategy_reason=plan.reason,
                vram_strategy_approximate=plan.approximate,
                fallback_occurred=plan.use_cpu_pass_a_baseline,
                tile_settings=tile_settings,
                cuda_memory_before=plan.cuda_before.label() if plan.cuda_before else "",
                cuda_memory_after=self._cuda_memory_snapshot(),
                status="error",
                error=f"Inference failed: {exc}",
                warnings=[*self.warnings, *runtime_warnings],
            )

        detections: list[Detection] = []
        class_counts = {name: 0 for name in CLASS_NAMES if name != "background"}
        filtered_boxes: list[list[float]] = []
        filtered_scores: list[float] = []
        filtered_labels: list[int] = []
        filtered_names: list[str] = []

        for box, score, label in zip(boxes, scores, labels):
            score_float = float(score)
            if score_float < confidence_threshold:
                continue
            label_int = int(label)
            class_name = ID_TO_CLASS.get(label_int, f"class_{label_int}")
            box_list = np.asarray(box, dtype=float).round(2).tolist()
            detections.append(Detection(box=box_list, score=score_float, label=label_int, class_name=class_name))
            filtered_boxes.append(box_list)
            filtered_scores.append(score_float)
            filtered_labels.append(label_int)
            filtered_names.append(class_name)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        warnings = [*self.warnings, *runtime_warnings]
        if not detections:
            warnings.append("No detections met the current confidence threshold.")

        annotated_path = None
        try:
            annotated = draw_annotations(image_path, detections, class_counts, confidence_threshold)
            annotated_name = (
                f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{uuid.uuid4().hex[:8]}.png"
            )
            annotated_path = save_annotated_image(annotated, ANNOTATED_CACHE / annotated_name)
        except Exception as exc:
            warnings.append(f"Could not build annotated preview: {exc}")
            logger.exception("Annotation failed")

        elapsed = time.perf_counter() - start
        logger.info(
            "Infer complete — pipeline=%s checkpoint=%s mode=%s count=%d time=%.2fs device=%s fallback=%s",
            PIPELINE_SOURCE, self.checkpoint_path.name,
            inference_mode, sum(class_counts.values()), elapsed,
            self.device_label, plan.use_cpu_pass_a_baseline,
        )
        return DetectionResult(
            image_path=str(image_path),
            timestamp=timestamp,
            boxes=filtered_boxes,
            labels=filtered_labels,
            scores=filtered_scores,
            class_names=filtered_names,
            detections=detections,
            class_counts=class_counts,
            total_count=sum(class_counts.values()),
            confidence_threshold=confidence_threshold,
            device=self.device_label,
            processing_time=elapsed,
            inference_mode=inference_mode,
            pipeline_source=PIPELINE_SOURCE,
            checkpoint_path=str(self.checkpoint_path),
            effective_image_path=str(image_path),
            effective_image_size=input_size,
            vram_strategy=plan.strategy_name,
            vram_strategy_reason=plan.reason,
            vram_strategy_approximate=plan.approximate,
            fallback_occurred=plan.use_cpu_pass_a_baseline,
            tile_settings=tile_settings,
            cuda_memory_before=plan.cuda_before.label() if plan.cuda_before else "",
            cuda_memory_after=self._cuda_memory_snapshot(),
            status="warning" if warnings else "success",
            warnings=warnings,
            annotated_image_path=annotated_path,
        )
