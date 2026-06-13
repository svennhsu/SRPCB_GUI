from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from app.paths import SR_CACHE, SR_CHECKPOINT, SR_MODEL_PATH
from inference.vram_strategy import choose_sr_batch_size, cuda_memory_info

if TYPE_CHECKING:
    import cv2
    import numpy as np
    import torch

PATCH_SIZE = 32
SCALE = 4
STRIDE = 24
BATCH_SIZE = 64

logger = logging.getLogger(__name__)


def _load_sr_model_module():
    spec = importlib.util.spec_from_file_location("sr_model", str(SR_MODEL_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["sr_model"] = module
    spec.loader.exec_module(module)
    return module.EDSRLite


class SuperResolutionEngine:
    def __init__(self, checkpoint_path: str | Path = SR_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self._model = None
        self._config: dict = {}
        self._loaded = False
        self._device = "cpu"
        self._device_label = "CPU"
        self._load_error: str | None = None

        if not self.checkpoint_path.exists():
            self._load_error = f"SR checkpoint not found: {self.checkpoint_path}"

        logger.info(
            "SuperResolutionEngine — checkpoint: %s  path: %s",
            self.checkpoint_path.name, self.checkpoint_path,
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self, device_preference: str = "auto") -> bool:
        import torch
        if self._loaded:
            return True
        if self._load_error and not self.checkpoint_path.exists():
            return False

        EDSRLite = _load_sr_model_module()

        try:
            self._device = self._select_device(device_preference)
            logger.info("Loading SR model from %s on %s", self.checkpoint_path, self._device)
            ckpt = torch.load(str(self.checkpoint_path), map_location=self._device, weights_only=False)
            config = ckpt["model_config"]
            self._config = dict(config)

            model = EDSRLite(
                scale=config["scale"],
                num_channels=config["num_channels"],
                num_features=config["num_features"],
                num_blocks=config["num_blocks"],
            ).to(self._device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model = model
            self._loaded = True
            self._load_error = None
            logger.info(
                "SR model loaded — device=%s scale=%d channels=%d features=%d blocks=%d",
                self._device_label, config["scale"], config["num_channels"],
                config["num_features"], config["num_blocks"],
            )
            return True
        except Exception as exc:
            self._load_error = f"SR model load failed: {exc}"
            logger.exception("SR model load failed")
            return False

    def run(self, image_path: str | Path, progress_callback: Callable[[int, int], None] | None = None) -> str:
        import cv2
        image_path = Path(image_path)
        if not self._loaded or self._model is None:
            raise RuntimeError(self._load_error or "SR model not loaded")

        start = time.perf_counter()
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        logger.info(
            "SR start — image=%s size=%dx%d device=%s",
            image_path.name, img.shape[1], img.shape[0], self._device_label,
        )
        batch_size = BATCH_SIZE
        try:
            import torch

            batch_size, batch_reason, mem_before = choose_sr_batch_size(torch, self._device, BATCH_SIZE)
            logger.info("SR VRAM plan — batch_size=%d reason=%s", batch_size, batch_reason)
            logger.info("%s", mem_before.label())
        except Exception as exc:
            logger.info("SR VRAM plan unavailable; using batch_size=%d (%s)", batch_size, exc)

        try:
            sr_bgr = self._super_resolve_full_image(self._model, img, progress_callback, batch_size=batch_size)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                try:
                    import torch

                    if str(self._device).startswith("cuda"):
                        torch.cuda.empty_cache()
                except Exception as cleanup_exc:
                    logger.debug("CUDA cache cleanup after SR OOM failed: %s", cleanup_exc)
                raise RuntimeError(
                    "CUDA out of memory during super-resolution. "
                    f"SR used tiled patches with batch_size={batch_size}; retry on CPU or close other GPU applications."
                ) from exc
            raise

        out_dir = SR_CACHE
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{image_path.stem}_SR_{int(time.time())}.png"
        out_path = out_dir / out_name
        ok = cv2.imwrite(str(out_path), sr_bgr)
        if not ok or not out_path.exists():
            raise RuntimeError(f"Failed to write SR output to {out_path}")

        elapsed = time.perf_counter() - start
        file_size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
        logger.info(
            "SR done — output=%s size=%dx%d (%.1f MB) time=%.2fs",
            out_name, sr_bgr.shape[1], sr_bgr.shape[0], file_size_mb, elapsed,
        )
        try:
            import torch

            logger.info("%s", cuda_memory_info(torch, self._device).label())
        except Exception as exc:
            logger.debug("CUDA memory snapshot after SR failed: %s", exc)
        return str(out_path)

    def _select_device(self, preference: str) -> str:
        import torch
        if preference == "cpu":
            self._device_label = "CPU"
            return "cpu"
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                self._device_label = f"CUDA: {name}"
                return "cuda"
            except Exception as exc:
                logger.warning("CUDA was reported available but could not be selected: %s", exc)
        self._device_label = "CPU"
        return "cpu"

    @staticmethod
    def _pad_image(img, patch_size, stride):
        import cv2
        h, w, _ = img.shape
        pad_h = 0 if h >= patch_size else patch_size - h
        pad_w = 0 if w >= patch_size else patch_size - w
        if h >= patch_size:
            rem_h = (h - patch_size) % stride
            if rem_h != 0:
                pad_h = stride - rem_h
        if w >= patch_size:
            rem_w = (w - patch_size) % stride
            if rem_w != 0:
                pad_w = stride - rem_w
        padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
        return padded, pad_h, pad_w

    @staticmethod
    def _super_resolve_full_image(model, img_bgr, progress_callback=None, batch_size: int = BATCH_SIZE):
        import cv2
        import numpy as np
        import torch

        device = next(model.parameters()).device
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb, pad_h, pad_w = SuperResolutionEngine._pad_image(img_rgb, PATCH_SIZE, STRIDE)

        h, w, _ = img_rgb.shape
        out_h = h * SCALE
        out_w = w * SCALE

        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 3), dtype=np.float32)

        y_steps = list(range(0, h - PATCH_SIZE + 1, STRIDE))
        x_steps = list(range(0, w - PATCH_SIZE + 1, STRIDE))
        total_tiles = len(y_steps) * len(x_steps)
        tile_idx = 0
        logger.info(
            "SR tiling — patch=%d stride=%d scale=%d tiles=%d batch_size=%d output=%dx%d",
            PATCH_SIZE, STRIDE, SCALE, total_tiles, batch_size, out_w, out_h,
        )

        patch_batch: list[np.ndarray] = []
        coord_batch: list[tuple[int, int, int, int]] = []

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    patch = img_rgb[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    patch = patch.astype(np.float32) / 255.0
                    patch_batch.append(patch)
                    coord_batch.append((
                        y * SCALE, x * SCALE,
                        (y + PATCH_SIZE) * SCALE, (x + PATCH_SIZE) * SCALE,
                    ))

                    if len(patch_batch) >= batch_size:
                        _flush_sr_batch(model, device, patch_batch, coord_batch,
                                        output, weight)
                        tile_idx += len(patch_batch)
                        patch_batch.clear()
                        coord_batch.clear()
                        if progress_callback:
                            progress_callback(tile_idx, total_tiles)

            if patch_batch:
                _flush_sr_batch(model, device, patch_batch, coord_batch,
                                output, weight)
                tile_idx += len(patch_batch)
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

        output = output / np.maximum(weight, 1e-8)

        if pad_h > 0:
            output = output[:-(pad_h * SCALE), :, :]
        if pad_w > 0:
            output = output[:, :-(pad_w * SCALE), :]

        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr


def _flush_sr_batch(model, device, patches, coords, output, weight):
    import numpy as np
    import torch

    batch = np.stack([np.transpose(p, (2, 0, 1)) for p in patches])
    batch_tensor = torch.from_numpy(batch).to(device)

    sr_batch = model(batch_tensor)
    sr_batch = sr_batch.detach().cpu().numpy()

    for i, (y0, x0, y1, x1) in enumerate(coords):
        sr_patch = np.transpose(sr_batch[i], (1, 2, 0))
        sr_patch = np.clip(sr_patch, 0, 1)
        output[y0:y1, x0:x1] += sr_patch
        weight[y0:y1, x0:x1] += 1.0
