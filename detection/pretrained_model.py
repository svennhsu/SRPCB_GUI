import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


PCB_ANCHOR_SIZES = ((8, 16), (32, 48), (64, 128), (256, 384), (512, 768))
PCB_ANCHOR_RATIOS = ((0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),) * 5


def get_pretrained_fasterrcnn_v4(num_classes, freeze_backbone=False):
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        min_size=1024,
        max_size=1024,
        box_detections_per_img=800,
        rpn_post_nms_top_n_test=1000,
    )

    state_dict = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict(progress=True)
    model.load_state_dict(state_dict)

    pcb_anchor_generator = AnchorGenerator(
        sizes=PCB_ANCHOR_SIZES,
        aspect_ratios=PCB_ANCHOR_RATIOS,
    )
    model.rpn.anchor_generator = pcb_anchor_generator

    num_anchors_per_loc = pcb_anchor_generator.num_anchors_per_location()[0]
    in_channels = model.rpn.head.cls_logits.in_channels
    model.rpn.head.cls_logits = nn.Conv2d(in_channels, num_anchors_per_loc, kernel_size=1)
    model.rpn.head.bbox_pred = nn.Conv2d(in_channels, num_anchors_per_loc * 4, kernel_size=1)
    nn.init.normal_(model.rpn.head.cls_logits.weight, std=0.01)
    nn.init.constant_(model.rpn.head.cls_logits.bias, 0)
    nn.init.normal_(model.rpn.head.bbox_pred.weight, std=0.01)
    nn.init.constant_(model.rpn.head.bbox_pred.bias, 0)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model

def unfreeze_backbone_v4(model):
    for param in model.backbone.parameters():
        param.requires_grad = True


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
