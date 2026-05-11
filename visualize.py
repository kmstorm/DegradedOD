"""
Comprehensive Evaluation + Visualization: Baseline G-DINO vs MODD v3
- Both use threshold=0.5 for fair comparison
- MODD: <0.1 discard, 0.1-0.5 refine via memory, >0.5 keep directly
- Finds best samples where MODD improves over baseline
- Computes per-image mAP and picks top-10 improvement samples
"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, ".")
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from collections import defaultdict

from src.models.gdino_wrapper import GroundingDINOWrapper
from src.models.memory_bank import MemoryBank
from src.models.modd_detector import MODDConfig, MODDDetector
from src.data.exdark_dataset import ExDarkDataset, EXDARK_TO_COCO, parse_exdark_annotation

device = "cuda"
model_id = "IDEA-Research/grounding-dino-base"
checkpoint_path = "outputs/checkpoints_v3/best_phase_b.ckpt"
memory_bank_path = "outputs/memory_bank_final"
OUT_DIR = "/home/bao.km/.gemini/antigravity/brain/f4770134-66b8-4752-a2ef-9741be3f9e95/scratch"
os.makedirs(OUT_DIR, exist_ok=True)

coco_names = sorted(set(EXDARK_TO_COCO.values()))
prompt = " . ".join(coco_names) + " ."

# ── Color palette per class ──
PALETTE = [
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#FABEBE",
    "#008080", "#E6BEFF",
]
CLASS_COLORS = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(coco_names)}

SCORE_THRESH = 0.5  # Final display threshold for BOTH baseline & MODD

# ── IoU helpers ──
def iou_single(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def compute_image_metrics(dets, gts, iou_thresh=0.5):
    """Compute TP, FP, FN for one image."""
    tp, fp = 0, 0
    gt_matched = set()
    for d in dets:
        best_iou, best_gi = 0, -1
        for gi, g in enumerate(gts):
            if gi in gt_matched: continue
            if d["class"] != g["class"]: continue
            iou = iou_single(d["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thresh:
            tp += 1; gt_matched.add(best_gi)
        else:
            fp += 1
    fn = len(gts) - len(gt_matched)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1}


def draw_boxes(img, detections, title=""):
    """Draw boxes on image. detections: list of {bbox, class, score(optional)}"""
    draw = ImageDraw.Draw(img)
    # Title bar
    draw.rectangle([0, 0, img.width, 32], fill="black")
    draw.text((8, 8), title, fill="white")
    
    for d in detections:
        box = d["bbox"]
        label = d["class"]
        score = d.get("score", None)
        color = CLASS_COLORS.get(label, "#FFFFFF")
        draw.rectangle(box, outline=color, width=3)
        text = f"{label}" if score is None else f"{label} {score:.2f}"
        tw = len(text) * 7
        draw.rectangle([box[0], max(0,box[1]-16), box[0]+tw, max(0,box[1])], fill=color)
        draw.text((box[0]+2, max(0,box[1]-15)), text, fill="black")
    return img


def result_to_dets(result, thresh=0.5):
    """Convert DetectionResult to list of dicts, filtering by threshold."""
    dets = []
    for i in range(len(result)):
        s = result.scores[i].item()
        if s < thresh: continue
        dets.append({
            "bbox": result.boxes[i].tolist(),
            "class": result.labels[i],
            "score": s,
        })
    return dets


# ══════════════════════════════════════════════════════════════
# Load models
# ══════════════════════════════════════════════════════════════
print("Loading Baseline G-DINO...")
baseline = GroundingDINOWrapper(model_id=model_id, device=device, dtype=torch.float16, box_threshold=0.1)

print("Loading Memory Bank...")
mb = MemoryBank.load(memory_bank_path)

print("Loading MODD v3...")
config = MODDConfig(model_id=model_id, dtype="float16", low_conf_threshold=0.1, high_conf_threshold=0.5)
detector = MODDDetector(config)
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
detector.load_state_dict(ckpt["model_state_dict"], strict=False)
detector = detector.to(device).eval()
print(f"  Loaded checkpoint: epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('best_metric', '?'):.4f}")


# ══════════════════════════════════════════════════════════════
# Evaluate on ExDark val set
# ══════════════════════════════════════════════════════════════
dataset = ExDarkDataset("data/exdark", split="val", val_ratio=0.2)
print(f"\nEvaluating on {len(dataset)} val images...")

results_per_image = []

for idx in range(len(dataset)):
    sample = dataset[idx]
    img = sample["image"]
    gt_boxes = sample["gt_boxes"]
    gt_labels = sample["gt_labels"]
    image_id = sample["image_id"]
    
    # GT as dicts
    gt_dets = []
    for i in range(gt_boxes.shape[0]):
        gt_dets.append({"bbox": gt_boxes[i].tolist(), "class": gt_labels[i]})
    
    if len(gt_dets) == 0:
        continue
    
    with torch.no_grad():
        # Baseline: extract at low threshold, then filter at 0.5
        base_result = baseline.extract_features(img, prompt, box_threshold=0.1).detection
        base_dets = result_to_dets(base_result, thresh=SCORE_THRESH)
        
        # MODD: detect() handles thresholds internally (0.1 discard, 0.1-0.5 refine, >0.5 keep)
        modd_result = detector.detect(img, prompt, mb)
        modd_dets = result_to_dets(modd_result, thresh=SCORE_THRESH)
    
    base_metrics = compute_image_metrics(base_dets, gt_dets)
    modd_metrics = compute_image_metrics(modd_dets, gt_dets)
    
    # Track improvement
    f1_gain = modd_metrics["f1"] - base_metrics["f1"]
    tp_gain = modd_metrics["tp"] - base_metrics["tp"]
    fn_reduction = base_metrics["fn"] - modd_metrics["fn"]
    
    results_per_image.append({
        "idx": idx, "image_id": image_id,
        "base": base_metrics, "modd": modd_metrics,
        "base_dets": base_dets, "modd_dets": modd_dets, "gt_dets": gt_dets,
        "f1_gain": f1_gain, "tp_gain": tp_gain, "fn_reduction": fn_reduction,
        "n_base": len(base_dets), "n_modd": len(modd_dets), "n_gt": len(gt_dets),
    })
    
    if (idx+1) % 200 == 0:
        print(f"  [{idx+1}/{len(dataset)}] processed")

print(f"  Evaluated {len(results_per_image)} images")

# ══════════════════════════════════════════════════════════════
# Aggregate metrics
# ══════════════════════════════════════════════════════════════
def aggregate(results, key):
    tp = sum(r[key]["tp"] for r in results)
    fp = sum(r[key]["fp"] for r in results)
    fn = sum(r[key]["fn"] for r in results)
    prec = tp / (tp+fp) if (tp+fp)>0 else 0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": prec, "Recall": rec, "F1": f1}

base_agg = aggregate(results_per_image, "base")
modd_agg = aggregate(results_per_image, "modd")

print("\n" + "="*60)
print(f"RESULTS @ threshold={SCORE_THRESH} ({len(results_per_image)} images)")
print("="*60)
print(f"{'Metric':<15} {'Baseline':>12} {'MODD v3':>12} {'Δ':>10}")
print("-"*50)
for k in ["TP", "FP", "FN", "Precision", "Recall", "F1"]:
    b, m = base_agg[k], modd_agg[k]
    if isinstance(b, float):
        print(f"{k:<15} {b:>12.4f} {m:>12.4f} {m-b:>+10.4f}")
    else:
        print(f"{k:<15} {b:>12d} {m:>12d} {m-b:>+10d}")
print("="*60)

# Save metrics to JSON
metrics_out = {
    "threshold": SCORE_THRESH,
    "num_images": len(results_per_image),
    "baseline": base_agg,
    "modd_v3": modd_agg,
}
with open(f"{OUT_DIR}/metrics_comparison.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

# ══════════════════════════════════════════════════════════════
# Find top-10 best samples (MODD improves most over baseline)
# ══════════════════════════════════════════════════════════════
# Sort by: 1) tp_gain (MODD found more GT objects), 2) f1_gain
results_per_image.sort(key=lambda r: (r["tp_gain"], r["f1_gain"]), reverse=True)

print(f"\n{'='*60}")
print(f"TOP-10 MODD IMPROVEMENTS")
print(f"{'='*60}")

top10 = results_per_image[:10]
for rank, r in enumerate(top10):
    print(f"\n[#{rank+1}] {r['image_id']}")
    print(f"  GT={r['n_gt']} | Base={r['n_base']} (TP={r['base']['tp']},FP={r['base']['fp']},FN={r['base']['fn']}) | "
          f"MODD={r['n_modd']} (TP={r['modd']['tp']},FP={r['modd']['fp']},FN={r['modd']['fn']})")
    print(f"  TP_gain={r['tp_gain']:+d}, F1_gain={r['f1_gain']:+.3f}")
    
    # Generate side-by-side image
    img_path = Path("data/exdark/images") / r["image_id"]
    img = Image.open(str(img_path)).convert("RGB")
    w, h = img.size
    
    img_gt = draw_boxes(img.copy(), r["gt_dets"], f"GT ({r['n_gt']})")
    img_base = draw_boxes(img.copy(), r["base_dets"], f"Baseline ({r['n_base']})")
    img_modd = draw_boxes(img.copy(), r["modd_dets"], f"MODD ({r['n_modd']})")
    
    combined = Image.new("RGB", (w * 3, h))
    combined.paste(img_gt, (0, 0))
    combined.paste(img_base, (w, 0))
    combined.paste(img_modd, (w * 2, 0))
    
    out_path = f"{OUT_DIR}/top{rank+1:02d}_{r['image_id'].replace('/', '_')}"
    combined.save(out_path)
    print(f"  -> Saved: {out_path}")

print(f"\nDone! All outputs in {OUT_DIR}")
