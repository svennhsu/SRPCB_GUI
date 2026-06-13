import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import numpy as np
import random

def crop_robust_patches(image_path, xml_path, output_dir, patch_size=1024, min_keep_ratio=0.3):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except: return 0

    objects = []
    for obj in root.findall('object'):
        nm = obj.find('name').text.strip()
        # Filter out text metadata regions
        if "text" in nm.lower(): continue
        bnd = obj.find('bndbox')
        if not bnd: continue
        x1, y1, x2, y2 = float(bnd.find('xmin').text), float(bnd.find('ymin').text), float(bnd.find('xmax').text), float(bnd.find('ymax').text)
        objects.append({'name': nm, 'box': (x1, y1, x2, y2), 'area': (x2-x1)*(y2-y1)})

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    count = 0

    overlap = 256
    stride = patch_size - overlap

    y_steps = list(range(0, max(1, height - patch_size + 1), stride))
    x_steps = list(range(0, max(1, width - patch_size + 1), stride))
    
    for y in y_steps:
        for x in x_steps:
            
            j_x = max(0, min(max(0, width - patch_size), x + random.randint(-20, 20)))
            j_y = max(0, min(max(0, height - patch_size), y + random.randint(-20, 20)))

            process_patch(img, j_x, j_y, patch_size, patch_size, objects, f"{base_name}_n_{j_x}_{j_y}", output_dir, min_keep_ratio)
            count += 1

            if random.random() > 0.5:
                large_sz = int(patch_size * 1.2)
                l_x = max(0, min(max(0, width - large_sz), j_x))
                l_y = max(0, min(max(0, height - large_sz), j_y))
                process_patch(img, l_x, l_y, large_sz, patch_size, objects, f"{base_name}_zout_{l_x}_{l_y}", output_dir, min_keep_ratio)
                count += 1
                
    print(f"Created {count} intelligent patches from {base_name}")
    return count

def process_patch(full_img, cx, cy, crop_w, target_w, all_objs, pid, out_dir, min_keep_ratio):
    patch = full_img.crop((cx, cy, cx+crop_w, cy+crop_w))

    scale = float(target_w) / float(crop_w)
    if target_w != crop_w:
        patch = patch.resize((target_w, target_w), Image.Resampling.LANCZOS)

    if random.random() > 0.7:
        enhancer = ImageEnhance.Brightness(patch)
        patch = enhancer.enhance(random.uniform(0.7, 1.3))

    contained = []
    for obj in all_objs:
        x1, y1, x2, y2 = obj['box']
        ix1, iy1, ix2, iy2 = max(cx, x1), max(cy, y1), min(cx+crop_w, x2), min(cy+crop_w, y2)
        
        if ix2 > ix1 and iy2 > iy1:
            iarea = (ix2-ix1)*(iy2-iy1)
            if (iarea / obj['area']) >= min_keep_ratio:
                lx1 = (ix1 - cx) * scale
                ly1 = (iy1 - cy) * scale
                lx2 = (ix2 - cx) * scale
                ly2 = (iy2 - cy) * scale
                contained.append({'name': obj['name'], 'box': (lx1, ly1, lx2, ly2)})

    if len(contained) == 0 and random.random() > 0.1:
        return

    patch.save(os.path.join(out_dir, 'images', f"{pid}.jpg"))

    rt = ET.Element("annotation")
    ET.SubElement(rt, "filename").text = f"{pid}.jpg"
    sz = ET.SubElement(rt, "size")
    ET.SubElement(sz, "width").text = str(target_w)
    ET.SubElement(sz, "height").text = str(target_w)
    
    for c in contained:
        ob = ET.SubElement(rt, "object")
        ET.SubElement(ob, "name").text = c['name']
        bd = ET.SubElement(ob, "bndbox")
        ET.SubElement(bd, "xmin").text = str(int(c['box'][0]))
        ET.SubElement(bd, "ymin").text = str(int(c['box'][1]))
        ET.SubElement(bd, "xmax").text = str(int(c['box'][2]))
        ET.SubElement(bd, "ymax").text = str(int(c['box'][3]))
    
    ET.ElementTree(rt).write(os.path.join(out_dir, 'annotations', f"{pid}.xml"))

if __name__ == "__main__":
    src = r"component_counting_pcb_wacv_2019"
    dest = r"processed_patches_v4"

    curr = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(curr)
    
    in_dir = os.path.join(root, src)
    out_dir = os.path.join(root, dest)
    
    print(f"Generating Intelligent Multi-Scale Patches...\nSource: {in_dir}\nDest: {out_dir}")
    
    total = 0
    for d in [x for x in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, x))]:
        sub = os.path.join(in_dir, d)
        img_p = os.path.join(sub, f"{d}.jpg")
        xml_p = os.path.join(sub, f"{d}.xml")
        if os.path.exists(img_p) and os.path.exists(xml_p):
            total += crop_robust_patches(img_p, xml_p, out_dir)
    print(f"\nFinished. Created {total} high-variability training samples.")
