import os
import logging
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
from torchvision.ops import batched_nms
import matplotlib.pyplot as plt

from pretrained_model import get_pretrained_fasterrcnn_v4
from dataset_v4 import CLASS_MAPPING, classify_annotation

logger = logging.getLogger(__name__)
ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

def calculate_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    tp = 0
    matched_gt = set()
    for p_idx, p_box in enumerate(pred_boxes):
        best_iou = iou_threshold
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt: continue
            if pred_labels[p_idx] != gt_labels[g_idx]: continue
            xA = max(p_box[0], g_box[0])
            yA = max(p_box[1], g_box[1])
            xB = min(p_box[2], g_box[2])
            yB = min(p_box[3], g_box[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            boxP_area = (p_box[2]-p_box[0])*(p_box[3]-p_box[1])
            boxG_area = (g_box[2]-g_box[0])*(g_box[3]-g_box[1])
            union = boxP_area + boxG_area - inter
            iou = inter / (union + 1e-6)
            if iou >= best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

def get_ground_truth(xml_path):
    gt_boxes = []
    gt_labels = []
    if not os.path.exists(xml_path):
        return np.array([]), np.array([])
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_id = classify_annotation(name) if name else 0
            if class_id == 0: continue
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            if xmax > xmin and ymax > ymin:
                gt_boxes.append([xmin, ymin, xmax, ymax])
                gt_labels.append(class_id)
    except Exception as exc:
        logger.warning("Could not parse XML annotations from %s: %s", xml_path, exc)
    return np.array(gt_boxes), np.array(gt_labels)

def draw_comparison(image_path, pred_boxes, pred_labels, pred_scores, output_path):
    xml_path = os.path.splitext(image_path)[0] + ".xml"
    gt_boxes, gt_labels = get_ground_truth(xml_path)

    prec, rec, f1 = calculate_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels)
    
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    scale = max(1, int(width / 800))
    line_width = max(2, scale * 2)
    font_size = max(12, scale * 14)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        header_font = ImageFont.truetype("arial.ttf", font_size + 6)
    except:
        font = ImageFont.load_default()
        header_font = font

    colors = {
        1: "#FF0000", 2: "#00FF00", 3: "#0000FF", 4: "#FFFF00",
        5: "#FF00FF", 6: "#33FFFF", 7: "#FFA500"
    }
    
    def annotate(base_img, boxes, labels, is_pred=True):
        canvas = base_img.copy()
        draw = ImageDraw.Draw(canvas)
        title = "PREDICTION" if is_pred else "GROUND TRUTH"
        banner_h = int(font_size * 2.5)
        draw.rectangle([0, 0, width, banner_h], fill="#000000")
        draw.text((20, int(banner_h*0.2)), title, fill="#00FF00" if is_pred else "#00CCFF", font=header_font)
        
        counts = {c: 0 for c in CLASS_MAPPING.values()}
        for box, label in zip(boxes, labels):
            cls_name = ID_TO_CLASS.get(label, "unk")
            counts[label] += 1
            color = colors.get(label, "#FFFFFF")
            draw.rectangle(list(box), outline=color, width=line_width)
            prefix = cls_name[:3]
            draw.text((box[0], max(banner_h, box[1] - font_size - 2)), f"{prefix}_{counts[label]}", fill=color, font=font)
        return canvas

    left_img = annotate(img, pred_boxes, pred_labels, True)
    right_img = annotate(img, gt_boxes, gt_labels, False)

    combined_width = width * 2
    footer_h = max(350, int(font_size * 25))
    total_height = height + footer_h
    
    report = Image.new("RGB", (combined_width, total_height), "#111111")
    report.paste(left_img, (0, 0))
    report.paste(right_img, (width, 0))
    
    draw = ImageDraw.Draw(report)
    
    x_metrics = int(width * 0.1)
    x_table = int(width + (width * 0.1))
    y_base = height + int(footer_h * 0.15)
    
    draw.text((x_metrics, y_base), "ANALYTICAL EVALUATION", fill="#FF9900", font=header_font)
    y_vals = y_base + int(font_size * 2.5)
    line_sp = int(font_size * 1.8)
    
    draw.text((x_metrics, y_vals), f"Precision (Accuracy):  {prec*100:.1f}%", fill="#FFFFFF", font=font)
    draw.text((x_metrics, y_vals + line_sp), f"Recall (Completeness): {rec*100:.1f}%", fill="#FFFFFF", font=font)
    draw.text((x_metrics, y_vals + line_sp*2), f"F1 Confidence Score:  {f1*100:.1f}%", fill="#FFFFFF", font=font)
    
    draw.text((x_table, y_base), "COMPONENT QUANTIFICATION AUDIT", fill="#FF9900", font=header_font)
    y_table_head = y_base + int(font_size * 2.5)
    
    c1 = x_table
    c2 = x_table + int(width * 0.3)
    c3 = x_table + int(width * 0.5)
    
    draw.text((c1, y_table_head), "Category", fill="#AAAAAA", font=font)
    draw.text((c2, y_table_head), "Predicted", fill="#AAAAAA", font=font)
    draw.text((c3, y_table_head), "Dataset GT", fill="#AAAAAA", font=font)
    
    cur_y = y_table_head + int(font_size * 1.5)
    draw.line([(x_table, cur_y - 5), (x_table + int(width*0.7), cur_y - 5)], fill="#444444", width=1)
    
    for cat_name, cat_id in CLASS_MAPPING.items():
        if cat_name == "background": continue
        p_count = int(np.sum(pred_labels == cat_id))
        g_count = int(np.sum(gt_labels == cat_id))
        
        fill_col = "#FFFFFF" if p_count == g_count else "#BBBBBB"
        draw.text((c1, cur_y), cat_name.capitalize(), fill=fill_col, font=font)
        draw.text((c2, cur_y), str(p_count), fill=fill_col, font=font)
        draw.text((c3, cur_y), str(g_count), fill=fill_col, font=font)
        cur_y += int(font_size * 1.3)
        
    report.save(output_path, quality=95)
    logger.info("High-fidelity comparison report saved: %s", output_path)

def draw_predictions(image_path, boxes, labels, scores, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    colors = {
        1: "#FF3333", # capacitor -> red
        2: "#33FF33", # resistor -> green
        3: "#3333FF", # ic -> blue
        4: "#FFFF33", # connector -> yellow
        5: "#FF33FF", # led -> pink
        6: "#33FFFF", # transistor -> cyan
        7: "#FFA500"  # diode -> orange
    }

    counts = {category: 0 for category in CLASS_MAPPING.keys() if category != "background"}

    width, height = img.size
    scale = max(1, int(width / 800))
    line_width = max(2, scale * 2)
    font_size = max(12, scale * 14)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        cls_name = ID_TO_CLASS.get(label, "unknown")
        if cls_name in counts:
            counts[cls_name] += 1
            instance_num = counts[cls_name]
        else:
            instance_num = 1

        color = colors.get(label, "#FFFFFF")
        draw.rectangle(list(box), outline=color, width=line_width)
        
        prefix = cls_name[:3] if cls_name != "unknown" else "unk"
        label_text = f"{prefix}_{instance_num}"
        draw.text((box[0], max(0, box[1] - font_size - 2)), label_text, fill=color, font=font)

    box_x, box_y = int(10 * scale), int(10 * scale)
    padding = int(10 * scale)
    text_y_spacing = font_size + int(4 * scale)
    box_width = int(250 * scale)
    box_height = padding * 2 + text_y_spacing * (len(counts) + 1)
    
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill="#000000", outline="#FFFFFF", width=max(1, scale))
    draw.text((box_x + padding, box_y + padding), "Component Counts:", fill="#FFFFFF", font=font)
    
    y_pos = box_y + padding + text_y_spacing
    for cls_name, count in counts.items():
        draw.text((box_x + padding, y_pos), f"  {cls_name.capitalize()}: {count}", fill="#FFFFFF", font=font)
        y_pos += text_y_spacing

    img.save(output_path)
    logger.info("Annotated image saved: %s counts=%s", output_path, counts)

    plt.figure(figsize=(10, 6))
    categories = list(counts.keys())
    values = list(counts.values())
    
    non_zero = [(cat, val) for cat, val in zip(categories, values) if val > 0]
    if non_zero:
        categories, values = zip(*non_zero)
        
    bar_colors = [colors.get(CLASS_MAPPING.get(cat, 0), '#333333') for cat in categories]
    plt.bar(categories, values, color=bar_colors)
    plt.title("Detected PCB Components")
    plt.xlabel("Component Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    base_path, _ = os.path.splitext(output_path)
    chart_path = f"{base_path}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    logger.info("Component count chart saved: %s", chart_path)

def _generate_tiles(img_w, img_h, tile_size, overlap):
    """
    Yields (x1, y1, x2, y2) tile rectangles covering the full image with
    the given tile_size and fractional overlap. Edge tiles are always included
    even if they are smaller than tile_size (the model's transform handles it).
    """
    stride = max(1, int(tile_size * (1 - overlap)))
    seen = set()

    xs = list(range(0, max(1, img_w - tile_size), stride))
    ys = list(range(0, max(1, img_h - tile_size), stride))

    # Always anchor a tile flush to the right / bottom edge
    if img_w > tile_size and (not xs or xs[-1] + tile_size < img_w):
        xs.append(img_w - tile_size)
    if img_h > tile_size and (not ys or ys[-1] + tile_size < img_h):
        ys.append(img_h - tile_size)

    if not xs:
        xs = [0]
    if not ys:
        ys = [0]

    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            yield x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)


def _infer_tile(model, device, tile_img, pre_threshold=0.25, scale_lock=True):
    """
    Run a single forward pass on one tile (PIL Image).
    - torchvision's GeneralizedRCNN.transform handles internal rescaling.
    - postprocess() maps boxes back to the tile's native pixel space.
    - Scores are clamped to [0, 1] defensively before any comparison.
    Returns boxes in tile pixel space.
    """
    # Match the original engine's critical behavior: lock torchvision's
    # transform to the incoming tile/native image size so it cannot silently
    # downscale small PCB parts before they reach the detector.
    restore_transform = False
    if scale_lock and hasattr(model, 'transform'):
        orig_min = model.transform.min_size
        orig_max = model.transform.max_size
        w, h = tile_img.size
        model.transform.min_size = (min(w, h),)
        model.transform.max_size = max(w, h)
        restore_transform = True

    inp = F.to_tensor(tile_img).unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            out = model(inp)[0]
    finally:
        if restore_transform:
            model.transform.min_size = orig_min
            model.transform.max_size = orig_max
    s = out['scores'].cpu().clamp(0.0, 1.0)   # Guard: never exceed 1.0
    b = out['boxes'].cpu()
    l = out['labels'].cpu()
    mask = s > pre_threshold
    return b[mask], s[mask], l[mask]


def _native_baseline_inference(model, device, img, score_threshold=0.5):
    """
    Original-engine equivalent full-frame pass.
    This is intentionally simple: scale-locked forward pass, score cutoff, then
    class-aware NMS. Tiled passes are additive and must not degrade this anchor.
    """
    b, s, l = _infer_tile(model, device, img, pre_threshold=score_threshold)
    if len(b) == 0:
        return b, s, l
    keep = batched_nms(b, s, l, 0.5)
    return b[keep], s[keep], l[keep]


def _box_ios_against_many(box, boxes):
    if len(boxes) == 0:
        return 0.0
    lt = torch.maximum(box[:2].unsqueeze(0), boxes[:, :2])
    rb = torch.minimum(box[2:].unsqueeze(0), boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_many = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    smaller = torch.minimum(area_box, area_many)
    return float((inter / (smaller + 1e-6)).max().item())


def _pair_overlap(box_a, box_b):
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    iou = inter / (union + 1e-6)
    ios = inter / (min(area_a, area_b) + 1e-6)
    return iou, ios


def _dedupe_dense_predictions(boxes, scores, labels):
    """
    Production visual dedupe: keep dense SMD coverage, but aggressively remove
    duplicate/fractured IC and connector boxes caused by tile boundaries.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    keep_boxes, keep_scores, keep_labels = [], [], []
    order = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)

    for idx in order:
        box = [float(v) for v in boxes[idx]]
        score = float(scores[idx])
        label = int(labels[idx])
        duplicate = False

        for kept_box, kept_label in zip(keep_boxes, keep_labels):
            iou, ios = _pair_overlap(box, kept_box)

            if label in [3, 4] and kept_label in [3, 4]:
                if iou > 0.12 or ios > 0.42:
                    duplicate = True
                    break
            elif label == kept_label:
                if label in [1, 2]:
                    if iou > 0.45 or ios > 0.78:
                        duplicate = True
                        break
                else:
                    if iou > 0.30 or ios > 0.65:
                        duplicate = True
                        break

        if not duplicate:
            keep_boxes.append(box)
            keep_scores.append(score)
            keep_labels.append(label)

    return keep_boxes, keep_scores, keep_labels


def custom_pcb_nms(boxes, scores, labels, sources, iou_threshold=0.40, ios_threshold=0.70):
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long)

    # Sort boxes by score descending
    order = torch.argsort(scores, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        box_i = boxes[i].unsqueeze(0)  # [1, 4]
        other_indices = order[1:]
        other_boxes = boxes[other_indices]  # [N-1, 4]

        # Calculate Intersections
        lt = torch.max(box_i[:, :2], other_boxes[:, :2])
        rb = torch.min(box_i[:, 2:], other_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]  # [N-1]

        # Calculate Areas
        area_i = (box_i[0, 2] - box_i[0, 0]) * (box_i[0, 3] - box_i[0, 1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        # IoU (Intersection over Union)
        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)

        # IoS (Intersection over Smaller area)
        smaller_area = torch.minimum(area_i, area_others)
        ios = inter / (smaller_area + 1e-6)

        # Build suppression mask
        suppress_mask = torch.zeros(other_indices.numel(), dtype=torch.bool)
        
        cancelled_i = False
        for idx in range(other_indices.numel()):
            j = other_indices[idx].item()
            
            lbl_i = labels[i].item()
            lbl_j = labels[j].item()
            
            # ── High-Precision Double Bounding Box Suppression (Large-Large) ──────────────────
            is_duplicate = False
            if lbl_i in [3, 4] and lbl_j in [3, 4]:
                if iou[idx] > 0.15 or ios[idx] > 0.45:
                    is_duplicate = True
            
            # ── Physical PCB Nesting Guard (Large-Small) ──────────────────────────────────────
            # A small component (Capacitor, Resistor, LED, Diode, Transistor) cannot touch
            # or overlap with a large structural component (IC, Connector). 
            # If they overlap, we suppress the small one if:
            # - LED, Transistor, Diode (5, 6, 7) overlapping by ios > 0.08 (highly sensitive to pins/leads)
            # - Capacitor, Resistor (1, 2) genuinely nested/overlapping by ios > 0.45
            suppress_small_j = False
            suppress_small_i = False
            if (lbl_i in [3, 4] and lbl_j in [1, 2, 5, 6, 7]) or (lbl_j in [3, 4] and lbl_i in [1, 2, 5, 6, 7]):
                small_lbl = lbl_j if lbl_j in [1, 2, 5, 6, 7] else lbl_i
                thresh = 0.08 if small_lbl in [5, 6, 7] else 0.45
                if ios[idx] > thresh:
                    if lbl_j in [1, 2, 5, 6, 7]:
                        suppress_small_j = True
                    else:
                        suppress_small_i = True
            
            is_collision = False
            
            # ── Suppression Execution ────────────────────────────────────────────────────────
            if suppress_small_i:
                # Small component i is nested/touching large component j, suppress i
                keep.pop()
                cancelled_i = True
                break
            elif suppress_small_j or is_collision or iou[idx] > iou_threshold or ios[idx] > ios_threshold:
                # Suppress the lower-scoring candidate j
                suppress_mask[idx] = True
            elif is_duplicate:
                # GLOBAL CONTEXT PROTECTION RULE:
                if sources[j] == 0 and labels[j] in [3, 4] and sources[i] == 1 and labels[i] != labels[j]:
                    keep.pop()
                    cancelled_i = True
                    break
                else:
                    suppress_mask[idx] = True

        if cancelled_i:
            # We discarded the current box i, so the rest of the order remains active except box i
            order = order[1:]
        else:
            # Keep only boxes that are NOT suppressed
            order = order[1:][~suppress_mask]

    return torch.tensor(keep, dtype=torch.long)


def run_v4_inference_native(
    img_path,
    model,
    device,
    tile_pre_threshold=0.25,
    tile_pre_threshold_c=0.92,
    tile_pre_threshold_c_smd=0.98,
    tile_pre_threshold_c_connector=0.995,
    nms_iou_threshold=0.40,
    nms_ios_threshold=0.70,
    conf_cap=0.96,
    conf_res=0.90,
    conf_ic_con=0.95,
    conf_smd=0.55,
    pass_a_only=False,
    merge_strategy="dense",
):
    """
    Run the production-aligned V4 high-fidelity multi-scale tiled inference directly
    on an image file path.
    """
    img = Image.open(img_path).convert("RGB")
    return run_v4_inference_pil(
        img,
        model,
        device,
        tile_pre_threshold=tile_pre_threshold,
        tile_pre_threshold_c=tile_pre_threshold_c,
        tile_pre_threshold_c_smd=tile_pre_threshold_c_smd,
        tile_pre_threshold_c_connector=tile_pre_threshold_c_connector,
        nms_iou_threshold=nms_iou_threshold,
        nms_ios_threshold=nms_ios_threshold,
        conf_cap=conf_cap,
        conf_res=conf_res,
        conf_ic_con=conf_ic_con,
        conf_smd=conf_smd,
        pass_a_only=pass_a_only,
        merge_strategy=merge_strategy,
    )


def run_v4_inference_pil(
    img,
    model,
    device,
    tile_pre_threshold=0.25,
    tile_pre_threshold_c=0.92,
    tile_pre_threshold_c_smd=0.98,
    tile_pre_threshold_c_connector=0.995,
    nms_iou_threshold=0.40,
    nms_ios_threshold=0.70,
    conf_cap=0.96,
    conf_res=0.90,
    conf_ic_con=0.95,
    conf_smd=0.55,
    pass_a_only=False,
    merge_strategy="dense",
):
    """
    Multi-Scale Tiled Inference with Context Protection, PCB Physical Constraints, and Strict Zoom Filter.
    merge_strategy:
      - "dense": default production inspection mode. Pass C remains a true
        zoom pass, preserving small-component coverage while duplicate large
        boxes are culled.
      - "benchmark": protected Pass A baseline plus conservative tile rescues.
    Takes a PIL Image and runs the full scale-adaptive post-processing pipeline.
    """
    model.eval()
    W, H = img.size

    all_boxes, all_scores, all_labels, all_sources = [], [], [], []

    # ------------------------------------------------------------------
    # PASS A: Original-engine full image baseline.
    # This is protected: later tiled passes may add missing small detections,
    # but they do not suppress or relabel these full-context predictions.
    # ------------------------------------------------------------------
    baseline_boxes, baseline_scores, baseline_labels = _native_baseline_inference(
        model, device, img, score_threshold=0.5
    )
    if pass_a_only:
        return baseline_boxes.numpy(), baseline_scores.numpy(), baseline_labels.numpy()

    # ------------------------------------------------------------------
    # PASS B: 1024-px sliding tiles at native 1:1 scale (Source = 1)
    # ------------------------------------------------------------------
    if W > 1024 or H > 1024:
        tile_count_b = 0
        for tx1, ty1, tx2, ty2 in _generate_tiles(W, H, 1024, 0.60):
            crop = img.crop((tx1, ty1, tx2, ty2))
            b, s, l = _infer_tile(model, device, crop, pre_threshold=tile_pre_threshold)
            if len(b):
                b = b.clone()
                b[:, 0] += tx1;  b[:, 2] += tx1
                b[:, 1] += ty1;  b[:, 3] += ty1
                all_boxes.append(b)
                all_scores.append(s)
                all_labels.append(l)
                all_sources.append(torch.ones(len(b), dtype=torch.int32))
            tile_count_b += 1
        logger.info("Pass B tiling: %d 1024-px tiles processed", tile_count_b)

    # ------------------------------------------------------------------
    # DYNAMIC SCALE GUARD FOR PASS C (ZOOM TILING)
    # ------------------------------------------------------------------
    # Measure the size of high-confidence true SMD parts (Capacitors/Resistors) 
    # in the native Pass A + B. If true SMDs are already large (median >= 28.0px) 
    # or the image is a small crop/low-res, we bypass Pass C to prevent 2.0x 
    # magnification scale-confusion. Keeps Pass C active for high-density boards.
    # ------------------------------------------------------------------
    run_pass_c = False
    if W > 512 or H > 512:
        run_pass_c = True
        smd_sizes_ab = []
        if all_boxes:
            temp_boxes = torch.cat(all_boxes)
            temp_scores = torch.cat(all_scores)
            temp_labels = torch.cat(all_labels)
            for idx in range(len(temp_boxes)):
                box = temp_boxes[idx]
                lbl = temp_labels[idx].item()
                scr = temp_scores[idx].item()
                if lbl in [1, 2] and scr > 0.75:  # High-confidence true SMDs only
                    w_smd = (box[2] - box[0]).item()
                    h_smd = (box[3] - box[1]).item()
                    smd_sizes_ab.append(max(w_smd, h_smd))

        if len(smd_sizes_ab) >= 5:
            median_smd = float(np.median(smd_sizes_ab))
            if median_smd >= 28.0:
                run_pass_c = False
                logger.info("Pass C bypassed: native SMD median %.1fpx >= 28px", median_smd)
        else:
            if max(W, H) < 1500:
                run_pass_c = False
                logger.info("Pass C bypassed: few SMDs and small image")

    # ------------------------------------------------------------------
    # PASS C: 512-px zoom-in sliding tiles (Source = 2)
    # Uses strict tile_pre_threshold_c to eliminate empty-pad hallucinations.
    # ------------------------------------------------------------------
    if run_pass_c:
        tile_count_c = 0
        for tx1, ty1, tx2, ty2 in _generate_tiles(W, H, 512, 0.60):
            crop = img.crop((tx1, ty1, tx2, ty2))
            b, s, l = _infer_tile(
                model,
                device,
                crop,
                pre_threshold=0.25,
                scale_lock=(merge_strategy != "dense"),
            )
            if len(b):
                # Hybrid Pass C class-specific filtering to eliminate empty pads & text early
                mask = torch.zeros(len(b), dtype=torch.bool)
                for idx in range(len(b)):
                    lbl = l[idx].item()
                    scr = s[idx].item()
                    if lbl in [1, 2]: # SMD Capacitor / Resistor
                        if scr > tile_pre_threshold_c_smd:
                            mask[idx] = True
                    elif lbl == 4: # Connectors are the main Pass C false-positive source
                        if scr > tile_pre_threshold_c_connector:
                            mask[idx] = True
                    else: # IC, Connector, Transistor, Diode, LED
                        if scr > tile_pre_threshold_c:
                            mask[idx] = True
                b, s, l = b[mask], s[mask], l[mask]
                if len(b):
                    b = b.clone()
                    b[:, 0] += tx1;  b[:, 2] += tx1
                    b[:, 1] += ty1;  b[:, 3] += ty1
                    all_boxes.append(b)
                    all_scores.append(s)
                    all_labels.append(l)
                    all_sources.append(torch.ones(len(b), dtype=torch.int32) * 2)
            tile_count_c += 1
        logger.info("Pass C tiling: %d 512-px tiles processed", tile_count_c)

    if not all_boxes:
        return baseline_boxes.numpy(), baseline_scores.numpy(), baseline_labels.numpy()

    all_boxes  = torch.cat(all_boxes)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    all_sources = torch.cat(all_sources)

    # Clamp boxes inside boundaries
    all_boxes[:, 0].clamp_(min=0, max=W)
    all_boxes[:, 1].clamp_(min=0, max=H)
    all_boxes[:, 2].clamp_(min=0, max=W)
    all_boxes[:, 3].clamp_(min=0, max=H)

    # Drop zero-area/degenerate boxes
    valid = (all_boxes[:, 2] > all_boxes[:, 0]) & (all_boxes[:, 3] > all_boxes[:, 1])
    all_boxes, all_scores, all_labels, all_sources = (
        all_boxes[valid],
        all_scores[valid],
        all_labels[valid],
        all_sources[valid],
    )

    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # ------------------------------------------------------------------
    # CONTAINMENT-AWARE PCB NMS WITH GLOBAL CONTEXT PROTECTION
    # ------------------------------------------------------------------
    keep = custom_pcb_nms(
        all_boxes,
        all_scores,
        all_labels,
        all_sources,
        iou_threshold=nms_iou_threshold,
        ios_threshold=nms_ios_threshold,
    )
    
    # Save a copy of pre-NMS raw Pass A boxes for boundary error correction
    raw_pass_a_boxes = []
    raw_pass_a_labels = []
    for idx in range(len(baseline_boxes)):
        if baseline_labels[idx].item() in [3, 4]:
            raw_pass_a_boxes.append(baseline_boxes[idx])
            raw_pass_a_labels.append(baseline_labels[idx].item())
            
    all_boxes, all_scores, all_labels = all_boxes[keep], all_scores[keep], all_labels[keep]

    for i in range(len(all_boxes)):
        lbl_i = all_labels[i].item()
        if lbl_i in [3, 4]:
            box_i = all_boxes[i]
            x1_i, y1_i, x2_i, y2_i = box_i[0].item(), box_i[1].item(), box_i[2].item(), box_i[3].item()
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            
            for box_a, lbl_a in zip(raw_pass_a_boxes, raw_pass_a_labels):
                if lbl_a in [3, 4]:
                    x1_a, y1_a, x2_a, y2_a = box_a[0].item(), box_a[1].item(), box_a[2].item(), box_a[3].item()
                    
                    # Compute intersection
                    xi1 = max(x1_i, x1_a)
                    yi1 = max(y1_i, y1_a)
                    xi2 = min(x2_i, x2_a)
                    yi2 = min(y2_i, y2_a)
                    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                    
                    if inter_area > 0:
                        area_a = (x2_a - x1_a) * (y2_a - y1_a)
                        union_area = area_i + area_a - inter_area
                        iou = inter_area / union_area
                        ios = inter_area / min(area_i, area_a)
                        
                        # If they are duplicates of the same physical part
                        if iou > 0.35 or ios > 0.50:
                            if area_a > 1.05 * area_i:
                                # Recover complete coordinates safely
                                all_boxes[i][0] = min(x1_i, x1_a)
                                all_boxes[i][1] = min(y1_i, y1_a)
                                all_boxes[i][2] = max(x2_i, x2_a)
                                all_boxes[i][3] = max(y2_i, y2_a)
                                break

    smd_sizes = []
    for idx in range(len(all_boxes)):
        box = all_boxes[idx]
        lbl = all_labels[idx].item()
        if lbl in [1, 2]:  # Capacitor or Resistor SMD
            w_smd = (box[2] - box[0]).item()
            h_smd = (box[3] - box[1]).item()
            smd_sizes.append(max(w_smd, h_smd))

    if len(smd_sizes) >= 5:
        median_smd = float(np.median(smd_sizes))
        size_thresh = 7.0 * median_smd
        area_thresh = 50.0 * (median_smd ** 2)
    else:
        size_thresh = 0.05 * max(W, H)
        area_thresh = 0.0025 * (W * H)

    final_boxes, final_scores, final_labels = [], [], []
    for idx in range(len(all_boxes)):
        box = all_boxes[idx]
        score = all_scores[idx].item()
        label = all_labels[idx].item()

        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h

        # Class specific final threshold filter
        if label == 1:
            conf_thresh = conf_cap
        elif label == 2:
            conf_thresh = conf_res
        elif label in [3, 4]:
            # Filter out small misclassified ICs (area < 1150 px^2, e.g. text/pad noise)
            if label == 3 and area < 1150:
                continue
            conf_thresh = conf_ic_con
        else:
            conf_thresh = conf_smd

        if score > conf_thresh:
            final_boxes.append(box.tolist())
            final_scores.append(score)
            final_labels.append(label)

    # ── Post-Expansion Duplicate Box Culling ───────────────────────────
    # If coordinate expansion of local tile passes creates multiple overlapping boxes
    # for the same component, we cull any lower-confidence duplicates (IoU > 0.85).
    final_boxes_clean = []
    final_scores_clean = []
    final_labels_clean = []
    
    indices_sorted = sorted(range(len(final_boxes)), key=lambda k: final_scores[k], reverse=True)
    for idx in indices_sorted:
        box_idx = final_boxes[idx]
        lbl_idx = final_labels[idx]
        score_idx = final_scores[idx]
        
        dup = False
        for kept_box, kept_lbl in final_boxes_clean:
            xi1 = max(box_idx[0], kept_box[0])
            yi1 = max(box_idx[1], kept_box[1])
            xi2 = min(box_idx[2], kept_box[2])
            yi2 = min(box_idx[3], kept_box[3])
            inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
            
            if inter > 0:
                area_idx = (box_idx[2] - box_idx[0]) * (box_idx[3] - box_idx[1])
                area_kept = (kept_box[2] - kept_box[0]) * (kept_box[3] - kept_box[1])
                iou = inter / (area_idx + area_kept - inter)
                if iou > 0.85:
                    dup = True
                    break
                    
        if not dup:
            final_boxes_clean.append((box_idx, lbl_idx))
            final_scores_clean.append(score_idx)
            final_labels_clean.append(lbl_idx)
            
    final_boxes = [box for box, lbl in final_boxes_clean]
    final_scores = final_scores_clean
    final_labels = final_labels_clean

    if len(final_boxes) == 0:
        return baseline_boxes.numpy(), baseline_scores.numpy(), baseline_labels.numpy()

    if merge_strategy == "dense":
        dense_boxes = final_boxes
        dense_scores = final_scores
        dense_labels = final_labels
        dense_boxes, dense_scores, dense_labels = _dedupe_dense_predictions(
            dense_boxes, dense_scores, dense_labels
        )
        return np.array(dense_boxes), np.array(dense_scores), np.array(dense_labels)

    # ------------------------------------------------------------------
    # PASS A PROTECTION + ADDITIVE TILE RESCUE
    # ------------------------------------------------------------------
    # Keep the original-engine baseline intact, then admit only high-confidence
    # tiled detections that are not duplicates/fragmentary overlaps of a
    # baseline component. This prevents tiling from making easy boards worse
    # while still allowing B/C to recover missed small parts.
    merged_boxes = baseline_boxes.tolist()
    merged_scores = baseline_scores.tolist()
    merged_labels = baseline_labels.tolist()

    baseline_ref = baseline_boxes.cpu()
    tile_order = sorted(range(len(final_boxes)), key=lambda k: final_scores[k], reverse=True)
    admitted_tile_boxes = []

    for idx in tile_order:
        box = torch.as_tensor(final_boxes[idx], dtype=torch.float32)
        score = float(final_scores[idx])
        label = int(final_labels[idx])

        if label in [1, 2]:
            extra_thresh = 0.92
        elif label in [3, 4]:
            extra_thresh = 0.80
        else:
            extra_thresh = 0.90

        if score < extra_thresh:
            continue
        if _box_ios_against_many(box, baseline_ref) > 0.30:
            continue
        if admitted_tile_boxes and _box_ios_against_many(box, torch.stack(admitted_tile_boxes)) > 0.70:
            continue

        admitted_tile_boxes.append(box)
        merged_boxes.append(final_boxes[idx])
        merged_scores.append(score)
        merged_labels.append(label)

    if len(merged_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.array(merged_boxes), np.array(merged_scores), np.array(merged_labels)


def load_best_v4(model_dir, device='cuda'):
    path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(path):
        logger.error("Missing V4 checkpoint: %s", path)
        return None
    
    m = get_pretrained_fasterrcnn_v4(num_classes=8)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(device)
    return m
