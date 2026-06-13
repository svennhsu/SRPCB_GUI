from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from inference.detection_result import Detection


BOX_COLORS = {
    1: "#FF0000",
    2: "#00FF00",
    3: "#0000FF",
    4: "#FFFF00",
    5: "#FF00FF",
    6: "#33FFFF",
    7: "#FFA500",
}


def draw_annotations(
    image_path: str | Path,
    detections: list[Detection],
    class_counts: dict[str, int],
    confidence_threshold: float,
    show_labels: bool = True,
    show_scores: bool = True,
) -> Image.Image:
    # Disable decompression-bomb guard — SR outputs may be 14000×14000+
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(image_path).convert("RGB")

    # Downscale huge canvases so annotations stay small on disk and fast to load
    MAX_ANNOTATION_DIM = 4096
    w, h = image.size
    ds_scale = 1.0
    if max(w, h) > MAX_ANNOTATION_DIM:
        ds_scale = MAX_ANNOTATION_DIM / max(w, h)
        image = image.resize((int(w * ds_scale), int(h * ds_scale)), Image.LANCZOS)
        w, h = image.size

    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    scale = max(1, width // 1000)
    line_width = max(2, int(scale * 2 * ds_scale))
    font_size = max(8, int(scale * 13 * ds_scale))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size + 2)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    per_class_seen: dict[str, int] = {}
    for det in detections:
        color = BOX_COLORS.get(det.label, "#ffffff")
        box = [float(v) * ds_scale for v in det.box]
        draw.rectangle(box, outline=color, width=line_width)
        if not show_labels:
            continue

        per_class_seen[det.class_name] = per_class_seen.get(det.class_name, 0) + 1
        parts = [f"{det.class_name[:3].upper()}_{per_class_seen[det.class_name]}"]
        if show_scores:
            parts.append(f"{det.score * 100:.0f}%")
        text = " ".join(parts)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = max(0, min(box[0], width - tw - 6))
        ty = max(0, box[1] - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(0, 0, 0, 180))
        draw.text((tx + 3, ty + 2), text, fill=color, font=font)

    non_zero = [(name, count) for name, count in class_counts.items() if count > 0]
    summary_lines = [f"Total: {sum(class_counts.values())}", f"Threshold: {confidence_threshold:.2f}"]
    summary_lines.extend(f"{name}: {count}" for name, count in non_zero[:8])
    line_h = font_size + 5
    panel_w = max(260, int(width * 0.18))
    panel_h = 16 + line_h * (len(summary_lines) + 1)
    draw.rectangle([10, 10, panel_w, panel_h], fill=(0, 0, 0, 185), outline=(255, 255, 255, 180))
    draw.text((18, 17), "PCB AOI Counts", fill="#ffffff", font=title_font)
    y = 22 + line_h
    for line in summary_lines:
        draw.text((18, y), line, fill="#e6edf3", font=font)
        y += line_h

    return image


def save_annotated_image(image: Image.Image, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return str(path)

