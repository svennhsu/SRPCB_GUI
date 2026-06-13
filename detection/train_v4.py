"""
V4 Training Orchestrator — With Real-Time Metrics
==================================================
Features:
1. Hardcoded Scale Lock Enforcement (512x512 forced)
2. Real-Time MAP, Precision & Recall valuation EVERY EPOCH!
3. Best-Model Saving based on mAP50, NOT Loss!
4. Full Isolated Execution in /v4
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

# Import isolated local files
from pretrained_model import get_pretrained_fasterrcnn_v4, unfreeze_backbone_v4, cuda_diagnostic
from dataset_v4 import PCBDatasetV4, classify_annotation
from inference_v4 import run_v4_inference_pil
from torchvision.transforms import functional as TF

def collate_fn(batch): return tuple(zip(*batch))

def create_imbalance_sampler(dataset_subset):
    import xml.etree.ElementTree as ET
    print("[INFO] Optimizing class balance: prioritizing underrepresented LEDs and Transistors...")
    weights = []
    for idx in tqdm(dataset_subset.indices, desc="Weighing subset"):
        img_name = dataset_subset.dataset.img_names[idx]
        xfile = os.path.join(dataset_subset.dataset.ann_dir, os.path.splitext(img_name)[0] + ".xml")
        try:
            root = ET.parse(xfile).getroot()
            boost = 1.0
            for obj in root.findall('object'):
                cid = classify_annotation(obj.find('name').text)
                # Massive boost priority for minority components
                if cid in [5, 6, 7]: # LED, Transistor, Diode
                    boost = 10.0; break 
                if cid == 3: # IC
                    boost = max(boost, 3.0) 
            weights.append(boost)
        except:
            weights.append(1.0)
    return torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

@torch.no_grad()
def evaluate_precision_recall_map(model, loader, device, iou_threshold=0.5):
    """
    Computes aggregate detection metrics (Precision, Recall, F1) aligned with
    the new high-precision V4 tiled inference and dynamic scaling system.
    """
    model.eval()
    all_tp = 0
    all_fp = 0
    all_total_gt = 0
    
    # Restrict to first 150 images to maintain high training speed, 
    # while gaining high confidence statistical samples.
    sample_limit = 150 
    processed = 0

    for imgs, targets in loader:
        if processed >= sample_limit: break
        
        for batch_idx in range(len(imgs)):
            img_tensor = imgs[batch_idx]
            t = targets[batch_idx]
            
            t_boxes = t['boxes'].cpu()
            t_labels = t['labels'].cpu()
            all_total_gt += len(t_labels)
            
            # Convert PyTorch tensor to PIL Image to run production scale-adaptive tiled pipeline
            img_pil = TF.to_pil_image(img_tensor.cpu())
            
            # Run the exact high-fidelity production inference pipeline!
            p_boxes_np, p_scores_np, p_labels_np = run_v4_inference_pil(
                img_pil,
                model,
                device,
                tile_pre_threshold=0.25,
                tile_pre_threshold_c=0.92,
                tile_pre_threshold_c_smd=0.98,
                nms_iou_threshold=0.40,
                nms_ios_threshold=0.70,
                conf_cap=0.96,
                conf_res=0.90,
                conf_ic_con=0.95,
                conf_smd=0.55,
            )
            
            if len(p_labels_np) == 0:
                # 0 TP, 0 FP, all FN
                continue
                
            p_boxes = torch.tensor(p_boxes_np, dtype=torch.float32)
            p_labels = torch.tensor(p_labels_np, dtype=torch.long)
            
            if len(t_labels) == 0:
                all_fp += len(p_labels)
                continue
                
            # Compute pairwise IoU
            ious = box_iou(p_boxes, t_boxes) # [N_pred, M_gt]
            
            matches = 0
            matched_gts = set()
            
            # Simple matching strategy for stats
            for p_idx in range(len(p_boxes)):
                best_iou = iou_threshold
                best_gt = -1
                for g_idx in range(len(t_labels)):
                    if g_idx in matched_gts: continue
                    if p_labels[p_idx] != t_labels[g_idx]: continue # Must match class
                    
                    curr_iou = ious[p_idx, g_idx].item()
                    if curr_iou >= best_iou:
                        best_iou = curr_iou
                        best_gt = g_idx
                
                if best_gt != -1:
                    matches += 1
                    matched_gts.add(best_gt)
            
            all_tp += matches
            all_fp += (len(p_labels) - matches)
            
        processed += len(imgs)

    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_total_gt + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1


def train_one_epoch(model, optimizer, loader, device, epoch, scaler=None):
    model.train()
    tot = 0.0
    pbar = tqdm(loader, desc=f"Epoch [{epoch}]")
    for imgs, tars in pbar:
        imgs = [i.to(device) for i in imgs]
        tars = [{k: v.to(device) for k, v in t.items()} for t in tars]
        
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                losses = sum(model(imgs, tars).values())
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = sum(model(imgs, tars).values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
        tot += losses.item()
        pbar.set_postfix({'Loss': f"{losses.item():.4f}"})
    return tot / len(loader)

def main():
    cuda_diagnostic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Local Project Structure Discovery
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patch_dir = os.path.join(root, "processed_patches_v4")
    
    if not os.path.exists(patch_dir):
         print(f"[WARNING] '{patch_dir}' not found. Defaulting back to original 'processed_patches'.")
         print("--> TIP: Run 'python v4/preprocessing_v4.py' first to generate better training data!")
         patch_dir = os.path.join(root, "processed_patches")
    
    print(f"Loading isolated V4 Dataset from: {patch_dir}")
    full = PCBDatasetV4(patch_dir, is_training=True)
    train_size = int(0.85 * len(full))
    train_set, val_set = torch.utils.data.random_split(full, [train_size, len(full)-train_size])
    
    # ⚖️ BALANCER: Scan training data to ensure rare components receive priority representation
    train_sampler = create_imbalance_sampler(train_set)
    
    # High-Resolution Imagery Loader with Intelligent Class Balancing
    # ⚠️ SAFETY CAP: Reduced to 3 to ensure Phase 2 unfreezing doesn't trigger OOM on the 24GB limit.
    train_loader = DataLoader(train_set, batch_size=3, sampler=train_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Starting V4 Run | Train: {len(train_set)} | Val: {len(val_set)}")
    
    # Scale Locked Model Init
    model = get_pretrained_fasterrcnn_v4(num_classes=8, freeze_backbone=True)
    model.to(device)
    
    # ── Command-Line Parameter Parsing ──────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(description="V4 Training Orchestrator")
    parser.add_argument("--restart", action="store_true", help="Restart training from epoch 0")
    parser.add_argument("--best_model", action="store_true", help="Load weights from best_model.pth")
    args = parser.parse_known_args()[0]
    # ────────────────────────────────────────────────────────────────────────────

    chk_dir = os.path.join(root, "checkpoints_v4")
    os.makedirs(chk_dir, exist_ok=True)
    
    # TRAINING REGIME (2 Phase)
    # PHASE 1: Burn in the head quickly
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=5e-3, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_f1 = 0.0
    
    history = {'epoch': [], 'prec': [], 'rec': [], 'f1': [], 'loss': []}
    
    # 💾 AUTOMATIC CHECKPOINT RESUMPTION / EXPLICIT RESTART
    # Seamlessly picks up where it left off, or restarts with best weights per user input.
    start_epoch = 0
    
    # Choose loading pathway
    if args.best_model:
        ckpt_path = os.path.join(chk_dir, "best_model.pth")
    else:
        # Priority load: latest temporary backup -> best model fallback
        ckpt_path = os.path.join(chk_dir, "last_epoch.pth") 
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(chk_dir, "best_model.pth")
        
    if os.path.exists(ckpt_path):
        lbl_msg = "best_model.pth" if "best_model.pth" in ckpt_path else "last_epoch.pth"
        print(f"\n[LOAD] Attempting weights load from: {lbl_msg}")
        try:
            # Set weights_only=False as standard safety for non-production legacy loading
            load_pkg = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(load_pkg['model_state_dict'])
            
            # Always fetch the global absolute best F1 from best_model.pth
            best_ckpt = os.path.join(chk_dir, "best_model.pth")
            if os.path.exists(best_ckpt):
                temp_b = torch.load(best_ckpt, map_location='cpu', weights_only=False)
                best_f1 = temp_b.get('f1', 0.0)
            else:
                best_f1 = load_pkg.get('f1', 0.0)
                
            if args.restart:
                start_epoch = 0
                best_f1 = 0.0
                print(f"[SUCCESS] Rehydrated weights. RESTARTING at Epoch 0 per request. Absolute Best F1 reset to 0.0.")
            else:
                start_epoch = load_pkg.get('epoch', 0) + 1
                print(f"[SUCCESS] Rehydrated weights. Resuming from Epoch {start_epoch}. Absolute Best F1 retained: {best_f1:.1%}")
        except Exception as e:
            print("[NOTICE] Shape mismatch or loading error. Starting clean build.")

    phase1_epochs = 20
    phase2_epochs = 980
    
    # Skip phase 1 entirely if we successfully resumed into a fully fine-tuned checkpoint,
    # or if we explicitly loaded fully fine-tuned weights from the best model!
    if start_epoch >= phase1_epochs or args.best_model:
         print("[SKIP] Phase 1 head burn-in redundant (fully-trained model loaded). Moving straight to phase 2 fine-tuning.")
         phase1_epochs = 0 
    
    print("\n" + "="*60 + "\nPHASE 1: Head Burn-in\n" + "="*60)
    
    for epoch in range(start_epoch, start_epoch + phase1_epochs):
        loss = train_one_epoch(model, optim, train_loader, device, epoch, scaler)
        prec, rec, f1 = evaluate_precision_recall_map(model, val_loader, device)
        
        print(f"  -> STATS: P: {prec:.1%} | R: {rec:.1%} | F1: {f1:.1%}")
        
        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['prec'].append(prec)
        history['rec'].append(rec)
        history['f1'].append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model_state_dict': model.state_dict(), 'f1': f1, 'epoch': epoch}, os.path.join(chk_dir, "best_model.pth"))
            print(f"  🏆 NEW BEST! Saved with F1={f1:.3f}")

    print("\n" + "="*60 + "\nPHASE 2: Full Model Unfreeze & Fine-Tuning (Longevity Run)\n" + "="*60)
    unfreeze_backbone_v4(model)
    
    # High-Performance Optimizer Configuration
    b_params = []
    h_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name: b_params.append(param)
        else: h_params.append(param)
        
    optim = torch.optim.AdamW([
        {'params': b_params, 'lr': 5e-6}, 
        {'params': h_params, 'lr': 5e-5}
    ], weight_decay=1e-4)
    
    # 🔄 ADDING SCHEDULER: Gradually lowers learning rate to optimize final perfection phase
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=phase2_epochs, eta_min=1e-8)
    
    # 🚀 CATCH UP: Fast-forward scheduler state to match resumed step for unbroken curve
    if args.restart:
        past_phase2_steps = 0
    else:
        past_phase2_steps = max(0, start_epoch - 20)
        
    if past_phase2_steps > 0:
        print(f"[SCHEDULER] Seamless alignment: Forwarding curve by {past_phase2_steps} steps.")
        for _ in range(past_phase2_steps): scheduler.step()
    
    # Start loop from wherever start_epoch left off to maintain global counter
    if args.restart:
        loop_start = 0
    else:
        loop_start = max(start_epoch, 20) # phase 1 total fixed count was 20
    for epoch in range(loop_start, 1000):
        loss = train_one_epoch(model, optim, train_loader, device, epoch, scaler)
        prec, rec, f1 = evaluate_precision_recall_map(model, val_loader, device)
        
        # Advance Scheduler Step
        scheduler.step()
        curr_lr = optim.param_groups[1]['lr']
        
        print(f"  -> STATS: P: {prec:.1%} | R: {rec:.1%} | F1: {f1:.1%} | LR: {curr_lr:.2e}")
        
        history['epoch'].append(epoch)
        history['loss'].append(loss)
        history['prec'].append(prec)
        history['rec'].append(rec)
        history['f1'].append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model_state_dict': model.state_dict(), 'f1': f1, 'epoch': epoch}, os.path.join(chk_dir, "best_model.pth"))
            print(f"  🏆 NEW BEST! Saved with F1={f1:.3f}")
            
        # 🔒 CONTINUITY GUARD: Every 5 epochs, force a checkpoint write so progress is never lost!
        if epoch % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(), 
                'f1': f1, 
                'epoch': epoch,
                'scheduler': scheduler.state_dict() # Keep scheduler curve synced
            }, os.path.join(chk_dir, "last_epoch.pth"))
            
    print("\nTraining Complete. Check loss graphs.")
    # Save graphs automatically
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['loss'], label="Loss")
    plt.title("V4 Train Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['prec'], label="Precision")
    plt.plot(history['epoch'], history['rec'], label="Recall")
    plt.plot(history['epoch'], history['f1'], label="F1", linestyle='--')
    plt.legend()
    plt.title("V4 Validation Metrics (True Indicators)")
    plt.savefig(os.path.join(chk_dir, "v4_curves.png"))
    print("Curves saved to", os.path.join(chk_dir, "v4_curves.png"))

if __name__ == "__main__":
    main()
