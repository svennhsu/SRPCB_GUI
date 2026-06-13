"""
V4 Production Inference Utility
================================
Run this script on ANY standalone image. 
Generates a high-gloss audit infographic with annotations placed below the board,
precluding overlap, alongside bounding-box confidence scores.

Usage:
    python v4/predict_online_v4.py
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Absolute folder mapping for absolute isolation
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.append(curr_dir)

from inference_v4 import load_best_v4, run_v4_inference_native
from dataset_v4 import CLASS_MAPPING, ID_TO_CLASS

def draw_online_report_v4(image_path, boxes, labels, scores, output_path):
    """
    Renders a premium audit canvas with:
    1. Top: Full Board with [ID + % Confidence] dynamic tags
    2. Bottom: High-Gloss Analytics Footer with quantification metrics (Zero Overlap!)
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Dynamic font & border scaling based on image dimension
    scale = max(1, int(width / 1000))
    line_width = max(2, scale * 2)
    font_size = max(12, scale * 14)
    
    try:
        # Attempt TrueType loading for ultra-crisp industrial output
        font = ImageFont.truetype("arial.ttf", font_size)
        header_font = ImageFont.truetype("arial.ttf", font_size + 6)
    except:
        font = ImageFont.load_default()
        header_font = font

    # High Contrast Palette
    colors = {
        1: "#FF0000", 2: "#00FF00", 3: "#0000FF", 4: "#FFFF00",
        5: "#FF00FF", 6: "#33FFFF", 7: "#FFA500"
    }
    
    # Step 1: Overlay Detections on Board using specified font-scaling
    canvas = img.copy()
    draw_ctx = ImageDraw.Draw(canvas)
    
    # Prepare header title for prediction strip
    banner_h = int(font_size * 2.5)
    
    # Tally setup for local indices
    instance_counts = {c: 0 for c in CLASS_MAPPING.values()}
    
    for box, label, score in zip(boxes, labels, scores):
        instance_counts[label] += 1
        cls_name = ID_TO_CLASS.get(label, "unk")
        color = colors.get(label, "#FFFFFF")
        
        # Draw precise rectangle
        draw_ctx.rectangle(list(box), outline=color, width=line_width)
        
        # Format dynamic tag: "CAP_1 | 98%"
        tag_text = f"{cls_name[:3].upper()}_{instance_counts[label]} | {int(score * 100)}%"
        
        # Position text neatly above box with clamp to prevent falling off top of frame
        txt_y = max(0, box[1] - font_size - 2)
        
        # Draw subtle text backdrop for max legibility on dense backgrounds
        bbox_txt = draw_ctx.textbbox((box[0], txt_y), tag_text, font=font)
        draw_ctx.rectangle(bbox_txt, fill="#000000")
        draw_ctx.text((box[0], txt_y), tag_text, fill=color, font=font)

    # Step 2: Synthesize Expanded Canvas for High-Gloss Footer
    footer_h = max(300, int(font_size * 20))
    total_height = height + footer_h
    
    report = Image.new("RGB", (width, total_height), "#111111")
    
    # Paste board at absolute top
    report.paste(canvas, (0, 0))
    
    # Draw dividing spacer
    report_ctx = ImageDraw.Draw(report)
    report_ctx.rectangle([0, height, width, height+5], fill="#FF9900")
    
    # Step 3: Construct Audit Tables in the newly generated footer deadzone
    y_base = height + int(footer_h * 0.2)
    x_margin = int(width * 0.05)
    
    report_ctx.text((x_margin, y_base), "AI COMPONENT QUANTIFICATION REPORT", fill="#FF9900", font=header_font)
    
    y_head = y_base + int(font_size * 2.5)
    c1 = x_margin
    c2 = x_margin + max(200, int(width * 0.3))
    
    report_ctx.text((c1, y_head), "Component Class", fill="#AAAAAA", font=font)
    report_ctx.text((c2, y_head), "Quantity Detected", fill="#AAAAAA", font=font)
    
    cur_y = y_head + int(font_size * 1.8)
    report_ctx.line([(x_margin, cur_y - 5), (x_margin + max(350, int(width * 0.5)), cur_y - 5)], fill="#444444", width=1)
    
    # Populate tally from Class Mapping
    for cat_name, cat_id in CLASS_MAPPING.items():
        if cat_name == "background": continue
        
        num_detected = instance_counts.get(cat_id, 0)
        
        # White text for found items, Dim grey for missing to increase contrast scan speed
        t_col = "#FFFFFF" if num_detected > 0 else "#555555"
        
        report_ctx.text((c1, cur_y), cat_name.capitalize(), fill=t_col, font=font)
        report_ctx.text((c2, cur_y), str(num_detected), fill=t_col, font=font)
        
        cur_y += int(font_size * 1.3)

    # Draw attribution watermark
    report_ctx.text((width - int(width*0.25), total_height - int(font_size * 2)), "V4 AOI Engine Live", fill="#333333", font=font)

    # Output stream
    report.save(output_path, quality=95, optimize=True)
    print(f"--- REPORT EXPORTED SUCCESSFULLY: {output_path} ---")


def main():
    # ===============================================================
    # RUN-TIME SETTINGS
    # ===============================================================
    # --> 📂 Target Image Input 📂 <--
    # Enter path to ANY standalone image file here
    INPUT_PATH = r"D:\Jones\AOI_v1\Olympus_C-960.jpg" 
    
    # ── Confidence Thresholds ───────────────────────────────────────
    # TILE_PRE_THRESHOLD  – per-tile pre-filter before the merge for Pass A and B.
    #   Lower = cast wider net; weak single-tile hits still get culled
    #   by the class-agnostic NMS + final thresholds below.
    TILE_PRE_THRESHOLD = 0.25

    # TILE_PRE_THRESHOLD_C – pre-filter applied to Pass C tiles (2:1 scale)
    #   for large/rare parts (ICs, Connectors, Transistors, Diodes, LEDs).
    TILE_PRE_THRESHOLD_C = 0.92

    # TILE_PRE_THRESHOLD_C_SMD – aggressive strict pre-filter applied to Pass C
    #   tiles (2:1 scale) for SMD Capacitors and Resistors. Set to 0.98 to
    #   completely cull text characters and empty pad false positives.
    TILE_PRE_THRESHOLD_C_SMD = 0.98

    # CONFIDENCE_IC_CON   – final threshold applied to ICs and Connectors
    #   after NMS. Keeps large/rare parts highly sensitive.
    #CONFIDENCE_IC_CON = 0.50
    CONFIDENCE_IC_CON = 0.95

    # CONFIDENCE_CAP      – final threshold applied to Capacitors (1) after NMS.
    #   Set to 0.70 to achieve the perfect 5% drop on top of tile filtering.
    CONFIDENCE_CAP = 0.96
    #CONFIDENCE_CAP = 0.60

    # CONFIDENCE_RES      – final threshold applied to Resistors (2) after NMS.
    #   Set to 0.82 to achieve the perfect 6.6% drop on top of tile filtering.
    CONFIDENCE_RES = 0.9

    # CONFIDENCE_SMD      – final threshold applied to remaining small SMD components
    #   (LEDs, Transistors, Diodes).
    CONFIDENCE_SMD = 0.55

    # NMS_IOU_THRESHOLD   – IoU ceiling for class-agnostic suppression.
    #   0.40 is a safe default.
    NMS_IOU_THRESHOLD = 0.40

    # NMS_IOS_THRESHOLD   – IoS (Intersection over Smaller area) ceiling.
    #   0.70 suppresses heavily contained/nested boxes.
    NMS_IOS_THRESHOLD = 0.70
    # ===============================================================
    
    # Resolve path automatically relative to root if manual path fails
    if not os.path.exists(INPUT_PATH):
        fn = os.path.basename(INPUT_PATH)
        test_fallback = os.path.join(root_dir, fn)
        if os.path.exists(test_fallback):
            INPUT_PATH = test_fallback
        else:
            print(f"\n[FATAL] Image file not found: {INPUT_PATH}")
            print("Please edit the 'INPUT_PATH' variable inside 'v4/predict_online_v4.py' to your target image.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nInitializing V4 Infrastructure | Device: {device}")
    
    # Load weights from isolation container
    model_container = os.path.join(curr_dir, "..", "checkpoints_v4")
    net = load_best_v4(model_container, device)
    
    if net is None:
        print("[FATAL] No checkpoint repository detected. Ensure model has been trained at least 1 epoch!")
        return

    print(f"Injecting pipeline into: {os.path.basename(INPUT_PATH)}...")
    
    # Execute multi-scale tiled inference with containment-aware NMS merge
    b, s, l = run_v4_inference_native(
        INPUT_PATH, net, device,
        tile_pre_threshold=TILE_PRE_THRESHOLD,
        tile_pre_threshold_c=TILE_PRE_THRESHOLD_C,
        tile_pre_threshold_c_smd=TILE_PRE_THRESHOLD_C_SMD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        nms_ios_threshold=NMS_IOS_THRESHOLD,
        conf_cap=CONFIDENCE_CAP,
        conf_res=CONFIDENCE_RES,
        conf_ic_con=CONFIDENCE_IC_CON,
        conf_smd=CONFIDENCE_SMD,
    )
    
    print(f"Discovery Complete! Found {len(b)} actionable components.")
    
    out_name = os.path.splitext(os.path.basename(INPUT_PATH))[0] + "_V4_Report.jpg"
    final_output = os.path.join(root_dir, out_name)
    
    # Dispatch to the new non-occluding visual compositor
    draw_online_report_v4(INPUT_PATH, b, l, s, final_output)
    
    print("\nMISSION SUCCESS")
    print(f"Open this file to view report: {final_output}\n")

if __name__ == "__main__":
    main()
