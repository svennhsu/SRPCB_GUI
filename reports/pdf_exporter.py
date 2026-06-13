from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from inference.detection_result import DetectionResult
from reports.report_builder import detection_table_rows


def _table(data, widths):
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def export_pdf(result: DetectionResult, output_path: str | Path, include_annotated_image: bool = True) -> None:
    output_path = Path(output_path)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "AOITitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#12355b"),
        spaceAfter=8,
    )
    story = [
        Paragraph("Super Resolution & PCB Component Counting", title),
        Paragraph("Automated Optical Inspection Report", styles["Normal"]),
        Spacer(1, 6 * mm),
    ]

    info = [
        ["Image filename", result.filename],
        ["Image path", result.image_path],
        ["Processing timestamp", result.timestamp],
        ["Processing time", f"{result.processing_time:.2f} s"],
        ["Inference mode", result.inference_mode],
        ["Detection source", result.detection_source_label or result.detection_source],
        ["Effective detector input", result.effective_image_path or result.image_path],
        [
            "Effective detector size",
            f"{result.effective_image_size[0]} × {result.effective_image_size[1]} px"
            if result.effective_image_size else "N/A",
        ],
        ["Pipeline source", result.pipeline_source or "N/A"],
        ["Checkpoint", result.checkpoint_path or "N/A"],
        ["Execution strategy", result.vram_strategy or "N/A"],
        ["Execution strategy reason", result.vram_strategy_reason or "N/A"],
        ["Approximate fallback", "Yes" if result.vram_strategy_approximate else "No"],
        ["Fallback occurred", "Yes" if result.fallback_occurred else "No"],
        ["CUDA before inference", result.cuda_memory_before or "N/A"],
        ["CUDA after inference", result.cuda_memory_after or "N/A"],
        ["SR applied", "Yes" if result.sr_image_path else "No"],
        ["Device used", result.device],
        ["Confidence threshold", f"{result.confidence_threshold:.2f}"],
        ["Total detected components", str(result.total_count)],
    ]
    if result.sr_image_path:
        info.insert(6, ["SR result path", result.sr_image_path])
    if result.sr_detection_proxy_path:
        info.insert(7, ["Detection proxy path", result.sr_detection_proxy_path])
        info.insert(8, ["Detection proxy note",
                         "SR output downsampled to original dimensions before detection to fit GPU memory."])
    story.append(_table(info, [45 * mm, 128 * mm]))
    story.append(Spacer(1, 6 * mm))

    counts = [["Component Class", "Count"]]
    for class_name, count in sorted(result.class_counts.items()):
        if count > 0:
            counts.append([class_name.capitalize(), str(count)])
    if len(counts) == 1:
        counts.append(["No detections", "0"])
    counts.append(["TOTAL", str(result.total_count)])
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(_table(counts, [85 * mm, 35 * mm]))

    if result.detections:
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Detection Details", styles["Heading2"]))
        story.append(_table(detection_table_rows(result), [14 * mm, 35 * mm, 25 * mm, 90 * mm]))

    # Use detection proxy for PDF (full 4× SR output is too large for PDF embed)
    sr_pdf_path = result.sr_detection_proxy_path or result.sr_image_path
    if include_annotated_image and sr_pdf_path and Path(sr_pdf_path).exists():
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Super-Resolved Image (Original Size)", styles["Heading2"]))
        story.append(Image(sr_pdf_path, width=170 * mm, height=110 * mm, kind="proportional"))

    if include_annotated_image and result.annotated_image_path and Path(result.annotated_image_path).exists():
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Annotated Image", styles["Heading2"]))
        story.append(Image(result.annotated_image_path, width=170 * mm, height=110 * mm, kind="proportional"))

    if result.warnings or result.error:
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph("Warnings / Errors", styles["Heading2"]))
        for message in result.warnings:
            story.append(Paragraph(message, styles["Normal"]))
        if result.error:
            story.append(Paragraph(result.error, styles["Normal"]))

    doc.build(story)
