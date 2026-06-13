from __future__ import annotations

from inference.detection_result import DetectionResult


def build_operator_report(result: DetectionResult, model_status: str = "Model loaded") -> str:
    lines = [
        "Super Resolution & PCB Component Counting \u2014 Inspection Report",
        "",
        "\u2500\u2500\u2500 Image \u2500\u2500\u2500",
        f"  Filename: {result.filename}",
        f"  Path: {result.image_path}",
        f"  Effective detector input: {result.effective_image_path or result.image_path}",
        f"  Effective detector size: {result.effective_image_size[0]} \u00d7 {result.effective_image_size[1]} px"
        if result.effective_image_size else "  Effective detector size: N/A",
        f"  Processing timestamp: {result.timestamp}",
    ]
    if result.sr_image_path:
        lines.extend([
            "",
            "\u2500\u2500\u2500 Super Resolution \u2500\u2500\u2500",
            f"  SR applied: Yes",
            f"  SR result path: {result.sr_image_path}",
        ])
        if result.sr_detection_proxy_path:
            lines.append("  Note: SR output downsampled to original size for detection.")
    lines.extend([
        "",
        "\u2500\u2500\u2500 Detection \u2500\u2500\u2500",
        f"  Inference mode: {result.inference_mode}",
        f"  Detection source: {result.detection_source_label or result.detection_source}",
        f"  Pipeline source: {result.pipeline_source or 'N/A'}",
        f"  Checkpoint: {result.checkpoint_path or 'N/A'}",
        f"  Confidence threshold: {result.confidence_threshold:.2f}",
        f"  Device: {result.device}",
        f"  Processing time: {result.processing_time:.2f} s",
        "",
        "\u2500\u2500\u2500 VRAM Strategy \u2500\u2500\u2500",
        f"  Execution strategy: {result.vram_strategy or 'N/A'}",
        f"  Strategy reason: {result.vram_strategy_reason or 'N/A'}",
        f"  Approximate fallback: {'Yes' if result.vram_strategy_approximate else 'No'}",
        f"  Fallback occurred: {'Yes' if result.fallback_occurred else 'No'}",
        f"  CUDA before: {result.cuda_memory_before or 'N/A'}",
        f"  CUDA after: {result.cuda_memory_after or 'N/A'}",
    ])
    if result.tile_settings:
        lines.extend([
            "  Tile settings:",
            f"    Pass B: {result.tile_settings.get('pass_b_tile_size')} px, "
            f"overlap {result.tile_settings.get('pass_b_overlap')}, "
            f"~{result.tile_settings.get('pass_b_estimated_tile_count')} tiles",
            f"    Pass C: {result.tile_settings.get('pass_c_tile_size')} px, "
            f"overlap {result.tile_settings.get('pass_c_overlap')}, "
            f"~{result.tile_settings.get('pass_c_estimated_tile_count')} tiles",
        ])
    lines.extend([
        "",
        "\u2500\u2500\u2500 Results \u2500\u2500\u2500",
        f"  Total detected components: {result.total_count}",
        f"  Model status: {model_status}",
        "",
        "  Component counts:",
    ])
    non_zero = False
    for class_name, count in sorted(result.class_counts.items()):
        if count > 0:
            non_zero = True
            lines.append(f"    {class_name.capitalize()}: {count}")
    if not non_zero:
        lines.append("    No components detected at the current threshold.")

    if result.warnings:
        lines.extend([
            "",
            "\u2500\u2500\u2500 Notes \u2500\u2500\u2500",
        ])
        lines.extend(f"  {warning}" for warning in result.warnings)
    if result.error:
        lines.extend(["", "  Error:", f"    {result.error}"])
    return "\n".join(lines)


def detection_table_rows(result: DetectionResult) -> list[list[str]]:
    rows = [["Index", "Class", "Confidence", "Bounding Box"]]
    for idx, det in enumerate(result.detections, start=1):
        box = ", ".join(f"{value:.1f}" for value in det.box)
        rows.append([str(idx), det.class_name.capitalize(), f"{det.score:.3f}", box])
    return rows
