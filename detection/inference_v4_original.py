"""
V4 Inference Engine - Optimized Native Execution
================================================
This replicates the EXACT logic used in the high-success original scripts, 
bypassing sliding windows entirely by directly feeding the high-resolution
frame into the Fully Convolutional layers, effectively detecting 100% scale.
"""

import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
from torchvision.ops import batched_nms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import sys
# Support relative import from outside dir just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pretrained_model import get_pretrained_fasterrcnn_v4
from dataset_v4 import CLASS_MAPPING, classify_annotation

# Inverse class mapping to get label names from class IDs
ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

# Import evaluation utilities from root source folder
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

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
    except Exception as e:
        print(f"Warning parsing XML: {e}")
    return np.array(gt_boxes), np.array(gt_labels)

def draw_comparison(image_path, pred_boxes, pred_labels, pred_scores, output_path):
    # Auto-resolve XML path
    xml_path = os.path.splitext(image_path)[0] + ".xml"
    gt_boxes, gt_labels = get_ground_truth(xml_path)
    
    # 1. Calculate Science Metrics
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
        # Draw Title Header
        title = "PREDICTION (AI MODEL)" if is_pred else "GROUND TRUTH (DATASET)"
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

    # 2. Create annotated sub-images
    left_img = annotate(img, pred_boxes, pred_labels, True)
    right_img = annotate(img, gt_boxes, gt_labels, False)
    
    # 3. Build Presentation Canvas
    combined_width = width * 2
    footer_h = max(350, int(font_size * 25))
    total_height = height + footer_h
    
    # High-gloss dark background
    report = Image.new("RGB", (combined_width, total_height), "#111111")
    report.paste(left_img, (0, 0))
    report.paste(right_img, (width, 0))
    
    draw = ImageDraw.Draw(report)
    
    # 4. Draw Analytical Footer
    x_metrics = int(width * 0.1)
    x_table = int(width + (width * 0.1))
    y_base = height + int(footer_h * 0.15)
    
    # Performance Card
    draw.text((x_metrics, y_base), "ANALYTICAL EVALUATION", fill="#FF9900", font=header_font)
    y_vals = y_base + int(font_size * 2.5)
    line_sp = int(font_size * 1.8)
    
    draw.text((x_metrics, y_vals), f"Precision (Accuracy):  {prec*100:.1f}%", fill="#FFFFFF", font=font)
    draw.text((x_metrics, y_vals + line_sp), f"Recall (Completeness): {rec*100:.1f}%", fill="#FFFFFF", font=font)
    draw.text((x_metrics, y_vals + line_sp*2), f"F1 Confidence Score:  {f1*100:.1f}%", fill="#FFFFFF", font=font)
    
    # Count Audit Table
    draw.text((x_table, y_base), "COMPONENT QUANTIFICATION AUDIT", fill="#FF9900", font=header_font)
    y_table_head = y_base + int(font_size * 2.5)
    
    # Draw vertical table columns manually for consistent presentation
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
        
        # Highlight rows that match perfectly in bright color
        fill_col = "#FFFFFF" if p_count == g_count else "#BBBBBB"
        draw.text((c1, cur_y), cat_name.capitalize(), fill=fill_col, font=font)
        draw.text((c2, cur_y), str(p_count), fill=fill_col, font=font)
        draw.text((c3, cur_y), str(g_count), fill=fill_col, font=font)
        cur_y += int(font_size * 1.3)
        
    report.save(output_path, quality=95)
    print(f"--- HIGH FIDELITY REPORT SAVED: {output_path} ---")

def draw_predictions(image_path, boxes, labels, scores, output_path):
    """
    Draws predicted bounding boxes and counts on the PCB image and saves it.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Class coloring map for high contrast visualization
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

    # Dynamic scaling based on image width
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
        
        # Text label above box
        prefix = cls_name[:3] if cls_name != "unknown" else "unk"
        label_text = f"{prefix}_{instance_num}"
        draw.text((box[0], max(0, box[1] - font_size - 2)), label_text, fill=color, font=font)

    # Draw summary count box at top-left
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
    print(f"\nSaved annotated image with counts to: {output_path}")
    print("Detected component counts:")
    for cls_name, count in counts.items():
        print(f"  {cls_name.capitalize()}: {count}")

    # Generate and save bar chart
    plt.figure(figsize=(10, 6))
    categories = list(counts.keys())
    values = list(counts.values())
    
    # Filter out categories with 0 count for a cleaner chart
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
    print(f"Saved component count chart to: {chart_path}")

def run_v4_inference_native(img_path, model, device, score_threshold=0.4):
    """
    Feeds the FULL resolution PCB image directly into the CNN layers, 
    by dynamically matching the transform filter size to native dimension.
    Ensures 0 component-slicing and 1:1 pixel detection fidelity.
    """
    model.eval()
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # CRITICAL DYNAMIC INJECTION
    # Adjust model filter to native dimensions temporarily to prevent squashing
    if hasattr(model, 'transform'):
        orig_min = model.transform.min_size
        orig_max = model.transform.max_size
        model.transform.min_size = (min(w, h),)
        model.transform.max_size = max(w, h)
    
    # Format input as [0,1] tensor, matching our V4 training exactly
    inp = F.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp)[0]
        
    # Revert scale locks immediately after just to preserve state
    if hasattr(model, 'transform'):
        model.transform.min_size = orig_min
        model.transform.max_size = orig_max

    b = out['boxes'].cpu()
    s = out['scores'].cpu()
    l = out['labels'].cpu()
    
    mask = s > score_threshold
    b, s, l = b[mask], s[mask], l[mask]
    
    # Final tight NMS clamp down duplicate overlap
    # Final NMS clamp. Raised to 0.5 to allow high-density neighbor clusters on dense boards!
    keep = batched_nms(b, s, l, 0.5)
    
    return b[keep].numpy(), s[keep].numpy(), l[keep].numpy()


def load_best_v4(model_dir, device='cuda'):
    path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(path):
        print("Missing V4 Checkpoint!")
        return None
    
    m = get_pretrained_fasterrcnn_v4(num_classes=8)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(device)
    print(f"V4 Model Loaded Successfully (Checkpoint F1: {ckpt.get('f1', 'N/A')})")
    return m

if __name__ == "__main__":
    # Direct local CLI tester
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chk_dir = os.path.join(root, "checkpoints_v4")
    
    net = load_best_v4(chk_dir, dev)
    if net:
        # Hardcoded Zybo path for quick smoke test
        test_f = os.path.join(root, "component_counting_pcb_wacv_2019", "Zybo", "Zybo.jpg")
        if os.path.exists(test_f):
            print("Running Smoke Test...")
            boxes, sc, lab = run_v4_inference_native(test_f, net, dev)
            print(f"Smoke Test Found {len(boxes)} objects. Execution OK.")
        else:
            print("Specify an image path to test.")
