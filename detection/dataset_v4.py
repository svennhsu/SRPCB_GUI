import os
import logging
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

CLASS_MAPPING = {
    "background": 0, "capacitor": 1, "resistor": 2, "ic": 3, 
    "connector": 4, "led": 5, "transistor": 6, "diode": 7
}
ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

LABEL_TO_CLASS = {
    "capacitor": 1, 
    "resistor": 2, 
    "ic": 3, 
    "connector": 4, "header": 4, "pins": 4, # Keep pure connector pins
    "led": 5, 
    "transistor": 6, 
    "diode": 7
}

def classify_annotation(name_raw):
    name = name_raw.strip().strip('"').strip("'").lower()
    if "text" in name:
        return 0
    
    first_word = name.split()[0] if name else ""
    if first_word in LABEL_TO_CLASS:
        return LABEL_TO_CLASS[first_word]
    for keyword, cid in LABEL_TO_CLASS.items():
        if keyword in name and cid > 0: return cid
    return 0

def get_train_aug():
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=10))

def get_val_aug():
    if not HAS_ALBUMENTATIONS: return None
    return A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=10))

class PCBDatasetV4(Dataset):
    def __init__(self, patches_dir, is_training=True):
        self.images_dir = os.path.join(patches_dir, 'images')
        self.ann_dir = os.path.join(patches_dir, 'annotations')
        self.is_training = is_training
        self.img_names = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        self.aug = get_train_aug() if is_training else get_val_aug()

    def __len__(self): return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_p = os.path.join(self.images_dir, name)
        img = Image.open(img_p).convert("RGB")
        
        xml_p = os.path.join(self.ann_dir, os.path.splitext(name)[0] + ".xml")
        boxes, labels = [], []
        
        if os.path.exists(xml_p):
            try:
                root = ET.parse(xml_p).getroot()
                for obj in root.findall('object'):
                    txt = obj.find('name').text
                    cid = classify_annotation(txt) if txt else 0
                    if cid > 0:
                        box = obj.find('bndbox')
                        x1 = float(box.find('xmin').text)
                        y1 = float(box.find('ymin').text)
                        x2 = float(box.find('xmax').text)
                        y2 = float(box.find('ymax').text)
                        if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cid)
            except ET.ParseError as exc:
                logger.warning("Skipping malformed annotation %s: %s", xml_p, exc)

        if len(boxes) == 0:
             boxes = np.zeros((0, 4))
             labels = []

        image_np = np.array(img)
        
        if self.aug:
            t = self.aug(image=image_np, bboxes=boxes, labels=labels)
            img_tensor = t['image'].float() / 255.0
            res_boxes = t['bboxes']
            res_labels = t['labels']
        else:
            import torchvision.transforms.functional as F
            img_tensor = F.to_tensor(img)
            res_boxes = boxes
            res_labels = labels

        if len(res_boxes) > 0:
            final_boxes = torch.as_tensor(res_boxes, dtype=torch.float32)
            final_labels = torch.as_tensor(res_labels, dtype=torch.int64)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_labels = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, {"boxes": final_boxes, "labels": final_labels, "image_id": torch.tensor([idx])}
