#!/usr/bin/env python3
"""
Ring-Based Buffer Evaluation — CLIP vs Pixel-Level Ground Truth

Evaluates CLIP classification predictions against pixel-level ground truth
rasterized from QGIS vector shapefiles.

Ground truth (.npz) contains per-pixel binary masks:
  vegetation — greenAtta or greenDeta polygons
  building — building polygons
  road — remaining pixels (not veg, not building)

Pipeline:
  SAM .npz → segment masks (N × H × W bool)
  CLIP .npy → class label per segment (list of strings)
  Combined → per-pixel class layer (H × W uint8)

Metrics per ring per class:
  MAE — |pred% − gt%|
  Precision — TP / (TP + FP)
  Recall — TP / (TP + FN)
  F1-score — harmonic mean of precision and recall
  IoU — TP / (TP + FP + FN)
  R² — area estimation quality (correlation)
  MBE — mean bias error (systematic over/underestimation)

Output:
  outputs/<experiment>/
  ├── per_tree_metrics_pixelgt.csv     ← per tree × ring × class
  ├── summary_pixelgt.csv              ← mean ± std across trees
  ├── final_summary_pixelgt.csv        ← aggregated by ring + class
  └── visualizations/
      └── {devEUI}_ring_eval.png       ← 5-panel figure per tree

Usage:
  python evaluate_rings.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import config

# RING MASKS
def meters_to_pixels(radius_m: float) -> float:
    """Convert metres to pixels using config scale."""
    return radius_m / config.METERS_PER_PIXEL

def create_ring_mask(inner_m: float, outer_m: float) -> np.ndarray:
    """Create boolean annulus mask centred at image centre."""
    Y, X = np.ogrid[:config.IMAGE_HEIGHT, :config.IMAGE_WIDTH]
    dist_px = np.sqrt((X - config.CENTER_COL) ** 2 + (Y - config.CENTER_ROW) ** 2)
    inner_px = meters_to_pixels(inner_m)
    outer_px = meters_to_pixels(outer_m)
    if inner_px <= 0:
        return dist_px <= outer_px
    return (dist_px > inner_px) & (dist_px <= outer_px)

# BUILD PREDICTION LAYER
def build_prediction_layer(masks_3d: np.ndarray, predictions: list) -> np.ndarray:
    """Combine SAM masks + CLIP predictions into per-pixel class layer.
    
    Class indices:
      0 — unclassified (background, excluded from metrics)
      1 — Vegetation
      2 — Building
      3 — Road
    """
    cls_idx = {"Vegetation": 1, "Building": 2, "Road": 3}
    cls_layer = np.zeros((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=np.uint8)
    for i in range(masks_3d.shape[0]):
        idx = cls_idx.get(predictions[i], 0)
        if idx > 0:
            cls_layer[masks_3d[i]] = idx
    return cls_layer

# PIXEL COUNTING
def pred_ring_pixels(cls_layer: np.ndarray, ring_masks: dict) -> dict:
    """Count predicted pixels per class per ring."""
    cls_idx = {"Vegetation": 1, "Building": 2, "Road": 3}
    results = {}
    for ring in config.RINGS:
        sfx = ring["suffix"]
        rm = ring_masks[sfx]
        total_px = int(np.sum(rm))
        res = {"total_pixels": total_px}
        for cls_name, idx in cls_idx.items():
            px = int(np.sum((cls_layer == idx) & rm))
            pct = round(px / total_px * 100, 3) if total_px > 0 else 0.0
            res[cls_name] = {"pixels": px, "percentage": pct}
        results[sfx] = res
    return results

def gt_ring_pixels(gt_npz: dict, ring_masks: dict) -> dict:
    """Count ground truth pixels per class per ring from rasterized .npz."""
    class_maps = {
        "Vegetation": gt_npz[config.GT_NPZ_KEYS["Vegetation"]].astype(bool),
        "Building": gt_npz[config.GT_NPZ_KEYS["Building"]].astype(bool),
        "Road": gt_npz[config.GT_NPZ_KEYS["Road"]].astype(bool),
    }
    results = {}
    for ring in config.RINGS:
        sfx = ring["suffix"]
        rm = ring_masks[sfx]
        total_px = int(np.sum(rm))
        res = {"total_pixels": total_px}
        for cls_name, px_map in class_maps.items():
            px = int(np.sum(px_map & rm))
            pct = round(px / total_px * 100, 3) if total_px > 0 else 0.0
            res[cls_name] = {"pixels": px, "percentage": pct}
        results[sfx] = res
    return results


# METRICS
def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, ring_mask: np.ndarray) -> dict:
    """Compute evaluation metrics from binary pixel masks inside a ring."""
    pred_in = pred_bin & ring_mask
    gt_in = gt_bin & ring_mask
    total = int(np.sum(ring_mask))

    tp = int(np.sum(pred_in & gt_in))
    fp = int(np.sum(pred_in & ~gt_in))
    fn = int(np.sum(~pred_in & gt_in))

    pred_px = int(np.sum(pred_in))
    gt_px = int(np.sum(gt_in))

    pred_pct = round(pred_px / total * 100, 3) if total > 0 else 0.0
    gt_pct = round(gt_px / total * 100, 3) if total > 0 else 0.0
    mae = round(abs(pred_pct - gt_pct), 3)

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    iou = round(tp / (tp + fp + fn), 4) if (tp + fp + fn) > 0 else 0.0

    return {
        "pred_pct": pred_pct, "gt_pct": gt_pct,
        "pred_px": pred_px, "gt_px": gt_px,
        "tp": tp, "fp": fp, "fn": fn,
        "mae": mae, "precision": precision, "recall": recall, "f1": f1, "iou": iou,
    }

def compute_r2_mbe(df: pd.DataFrame, ring_suffix: str, class_name: str) -> dict:
    """Compute R² and MBE for a specific ring and class across all trees."""
    pred_col = f"{class_name}_{ring_suffix}_pred_pct"
    gt_col = f"{class_name}_{ring_suffix}_gt_pct"
    
    pred = df[pred_col].values
    gt = df[gt_col].values
    
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mbe = np.mean(pred - gt)
    
    return {"r2": round(r2, 4), "mbe": round(mbe, 3)}

# VISUALIZATION
def save_visualization(image_np: np.ndarray, cls_layer: np.ndarray, gt_npz: dict,
                       ring_masks: dict, pred_rings: dict, gt_rings: dict,
                       dev_eui: str, tree_name: str, n_masks: int,
                       output_path: str, masks_3d: np.ndarray):
    """Save 5-panel evaluation figure for one tree."""
    h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    fig, axes = plt.subplots(1, 5, figsize=(35, 7))

    def add_rings(ax):
        for ring in config.RINGS:
            outer_px = meters_to_pixels(ring["outer_m"])
            ax.add_patch(plt.Circle((config.CENTER_COL, config.CENTER_ROW), outer_px,
                                   color=config.RING_CIRCLE_COLOR, fill=False,
                                   linewidth=config.RING_LINEWIDTH, label=f"{ring['outer_m']}m"))
        ax.plot(config.CENTER_COL, config.CENTER_ROW, "r+", markersize=12, markeredgewidth=2)

    # Panel 1: Original image
    axes[0].imshow(image_np)
    add_rings(axes[0])
    axes[0].set_title(f"Original\n{tree_name[:35]}", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].axis("off")

    # Panel 2: SAM segmentation
    seg_overlay = np.zeros((h, w, 4), dtype=np.float32)
    np.random.seed(42)
    for seg_mask in masks_3d[:50]:
        color = np.random.rand(4)
        color[3] = 0.4
        seg_overlay[seg_mask] = color
    axes[1].imshow(image_np)
    axes[1].imshow(seg_overlay)
    add_rings(axes[1])
    axes[1].set_title(f"SAM\n{len(masks_3d)} segments", fontsize=10)
    axes[1].axis("off")

    # Panel 3: CLIP classification
    cls_idx_map = {"Vegetation": 1, "Building": 2, "Road": 3}
    clip_overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls_name, idx in cls_idx_map.items():
        color = config.CLIP_CLASS_COLORS_RGBA[cls_name]
        clip_overlay[cls_layer == idx] = color
    axes[2].imshow(image_np)
    axes[2].imshow(clip_overlay)
    add_rings(axes[2])
    axes[2].set_title(f"CLIP prediction\n{n_masks} segments", fontsize=9)
    axes[2].axis("off")

    # Panel 4: Pixel GT
    gt_overlay = np.zeros((h, w, 4), dtype=np.float32)
    for cls_name, key in config.GT_NPZ_KEYS.items():
        gt_mask = gt_npz[key].astype(bool)
        color = config.GT_CLASS_COLORS_RGBA[cls_name]
        gt_overlay[gt_mask] = color
    axes[3].imshow(image_np)
    axes[3].imshow(gt_overlay)
    add_rings(axes[3])
    axes[3].set_title("Pixel GT\n(rasterized)", fontsize=9)
    axes[3].axis("off")

    # Panel 5: Bar chart
    ring_suffixes = [r["suffix"] for r in config.RINGS]
    x = np.arange(len(ring_suffixes))
    bar_w = 0.12
    cls_bar_colors = {"Vegetation": "green", "Building": "orangered", "Road": "gold"}
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    for ci, cls in enumerate(config.CLASS_NAMES):
        pred_vals = [pred_rings[sfx].get(cls, {}).get("percentage", 0.0) for sfx in ring_suffixes]
        gt_vals = [gt_rings[sfx].get(cls, {}).get("percentage", 0.0) for sfx in ring_suffixes]
        col = cls_bar_colors[cls]
        axes[4].bar(x + offsets[ci * 2] * bar_w, pred_vals, bar_w,
                   color=col, alpha=0.9, label=f"{cls} (CLIP)")
        axes[4].bar(x + offsets[ci * 2 + 1] * bar_w, gt_vals, bar_w,
                   color=col, alpha=0.4, label=f"{cls} (GT)", hatch="//")

    axes[4].set_xticks(x)
    axes[4].set_xticklabels(ring_suffixes)
    axes[4].set_ylabel("percentage (%)")
    axes[4].set_ylim(0, 115)
    axes[4].set_title("CLIP vs GT per ring", fontsize=9)
    axes[4].legend(fontsize=7, ncol=2)
    axes[4].grid(axis="y", alpha=0.3)

    plt.suptitle(f"{tree_name} | devEUI: {dev_eui}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

# MAIN
def main():
    print("=" * 70)
    print("RING-BASED BUFFER EVALUATION — pixel-level ground truth")
    print("=" * 70)
    print(f"segments dir     : {config.SEGMENTS_DIR}")
    print(f"classifications  : {config.CLASSIFICATIONS_DIR}")
    print(f"pixel GT dir     : {config.GROUND_TRUTH}")
    print(f"images dir       : {config.IMAGES_DIR}")
    print(f"m/px (DPI {config.EVAL_DPI}): {config.METERS_PER_PIXEL:.5f}")
    print("=" * 70)

    viz_dir = os.path.join(config.OUTPUT_BASE_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print("\npre-computing ring masks...")
    ring_masks = {}
    for ring in config.RINGS:
        mask = create_ring_mask(ring["inner_m"], ring["outer_m"])
        ring_masks[ring["suffix"]] = mask
        print(f"  {ring['suffix']}: outer={meters_to_pixels(ring['outer_m']):.1f} px "
              f"pixels={int(np.sum(mask))} theory={ring['theory_m2']:.2f} m²")

    clf_pattern = os.path.join(config.CLASSIFICATIONS_DIR, f"*{config.EVAL_SUFFIX}_classification.npy")
    clf_files = sorted(glob.glob(clf_pattern))
    print(f"\nfound {len(clf_files)} classification files")

    results = []
    evaluated = 0
    skipped = 0

    for clf_path in clf_files:
        fname = os.path.basename(clf_path)
        dev_eui = fname.replace(f"{config.EVAL_SUFFIX}_classification.npy", "")

        try:
            clf_data = np.load(clf_path, allow_pickle=True).item()
            predictions = list(clf_data["predictions"])
        except Exception as e:
            print(f"  ERROR clf {fname}: {e}")
            skipped += 1
            continue

        seg_path = os.path.join(config.SEGMENTS_DIR, f"{dev_eui}{config.EVAL_SUFFIX}_segments.npz")
        if not os.path.exists(seg_path):
            print(f"  SKIP {dev_eui}: segments not found")
            skipped += 1
            continue

        try:
            seg_data = np.load(seg_path)
            masks_3d = seg_data["segmentations"]
            n_masks = masks_3d.shape[0]
            if n_masks != len(predictions):
                print(f"  SKIP {dev_eui}: mask/pred mismatch ({n_masks} vs {len(predictions)})")
                skipped += 1
                continue
        except MemoryError as e:
            print(f"  SKIP {dev_eui}: memory error")
            skipped += 1
            continue
        except Exception as e:
            print(f"  ERROR seg {dev_eui}: {e}")
            skipped += 1
            continue

        gt_path = os.path.join(config.GROUND_TRUTH, f"{dev_eui}{config.EVAL_SUFFIX}_gt.npz")
        if not os.path.exists(gt_path):
            print(f"  SKIP {dev_eui}: pixel GT not found")
            skipped += 1
            continue

        try:
            gt_npz = dict(np.load(gt_path))
        except Exception as e:
            print(f"  ERROR gt {dev_eui}: {e}")
            skipped += 1
            continue

        img_path = os.path.join(config.IMAGES_DIR, f"{dev_eui}{config.EVAL_SUFFIX}.png")
        if not os.path.exists(img_path):
            print(f"  SKIP {dev_eui}: image not found")
            skipped += 1
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))

        cls_layer = build_prediction_layer(masks_3d, predictions)
        pred_bin = {
            "Vegetation": cls_layer == 1,
            "Building": cls_layer == 2,
            "Road": cls_layer == 3,
        }
        gt_bin = {cls: gt_npz[config.GT_NPZ_KEYS[cls]].astype(bool) for cls in config.CLASS_NAMES}

        pred_rings = pred_ring_pixels(cls_layer, ring_masks)
        gt_rings = gt_ring_pixels(gt_npz, ring_masks)

        tree_row = {"dev_eui": dev_eui}
        for ring in config.RINGS:
            sfx = ring["suffix"]
            rm = ring_masks[sfx]
            for cls in config.CLASS_NAMES:
                m = compute_metrics(pred_bin[cls], gt_bin[cls], rm)
                base = f"{cls.lower()}_{sfx}"
                for key, val in m.items():
                    tree_row[f"{base}_{key}"] = val
            total_px = int(np.sum(rm))
            uncovered_px = int(np.sum((cls_layer == 0) & rm))
            uncovered_pct = round(uncovered_px / total_px * 100, 3) if total_px > 0 else 0.0
            tree_row[f"uncovered_{sfx}_px"] = uncovered_px
            tree_row[f"uncovered_{sfx}_pct"] = uncovered_pct

        results.append(tree_row)
        evaluated += 1

        viz_path = os.path.join(viz_dir, f"{dev_eui}_ring_eval.png")
        save_visualization(image_np, cls_layer, gt_npz, ring_masks, pred_rings, gt_rings,
                          dev_eui, dev_eui, n_masks, viz_path, masks_3d)

        veg_mae = tree_row.get("vegetation_2m5_mae", 0)
        print(f"  [{evaluated:>3}] {dev_eui:<40} veg_mae_2m5={veg_mae:.1f}% masks={n_masks}")

    print(f"\nevaluated: {evaluated} | skipped: {skipped}")

    if not results:
        print("no results — check paths in config.py")
        return

    df = pd.DataFrame(results)

    per_tree_path = os.path.join(config.OUTPUT_BASE_DIR, config.PER_TREE_METRICS_FILE)
    df.to_csv(per_tree_path, index=False)
    print(f"saved: {per_tree_path}")

    num_df = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "metric": num_df.columns,
        "mean": num_df.mean().round(4).values,
        "std": num_df.std().round(4).values,
        "min": num_df.min().round(4).values,
        "max": num_df.max().round(4).values,
    })
    summary_path = os.path.join(config.OUTPUT_BASE_DIR, config.SUMMARY_FILE)
    summary.to_csv(summary_path, index=False)
    print(f"saved: {summary_path}")

    final_rows = []
    for ring in config.RINGS:
        sfx = ring["suffix"]
        for cls in config.CLASS_NAMES:
            base = f"{cls.lower()}_{sfx}"
            cols_needed = [f"{base}_mae", f"{base}_precision", f"{base}_recall", f"{base}_f1", f"{base}_iou"]
            if not all(c in num_df.columns for c in cols_needed):
                continue
            
            r2_mbe = compute_r2_mbe(df, sfx, cls.lower())
            
            final_rows.append({
                "ring": sfx, "class": cls,
                "mean_mae": round(num_df[f"{base}_mae"].mean(), 3),
                "std_mae": round(num_df[f"{base}_mae"].std(), 3),
                "mean_precision": round(num_df[f"{base}_precision"].mean(), 3),
                "mean_recall": round(num_df[f"{base}_recall"].mean(), 3),
                "mean_f1": round(num_df[f"{base}_f1"].mean(), 3),
                "mean_iou": round(num_df[f"{base}_iou"].mean(), 3),
                "r2": r2_mbe["r2"], "mbe": r2_mbe["mbe"],
            })

    final_path = os.path.join(config.OUTPUT_BASE_DIR, config.FINAL_SUMMARY_FILE)
    pd.DataFrame(final_rows).to_csv(final_path, index=False)
    print(f"saved: {final_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — mean metrics per ring × class")
    print("=" * 70)
    print(f"{'ring':<6} {'class':<12} {'MAE%':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'IoU':>6}")
    print("-" * 55)
    for r in final_rows:
        print(f"{r['ring']:<6} {r['class']:<12} {r['mean_mae']:>6.2f} {r['mean_precision']:>6.3f} "
              f"{r['mean_recall']:>6.3f} {r['mean_f1']:>6.3f} {r['mean_iou']:>6.3f}")

    mae_cols = [c for c in num_df.columns if c.endswith("_mae")]
    print(f"\noverall mean MAE: {num_df[mae_cols].values.mean():.2f}%")
    print(f"visualizations: {viz_dir}")
    print("done!")
    
    print("\n" + "=" * 70)
    print("R² AND MBE — per ring × class (area estimation quality)")
    print("=" * 70)
    print(f"{'ring':<6} {'class':<12} {'R²':>8} {'MBE (pp)':>10} {'interpretation':<25}")
    print("-" * 65)
    
    for ring in config.RINGS:
        sfx = ring["suffix"]
        for cls in config.CLASS_NAMES:
            r2_mbe = compute_r2_mbe(df, sfx, cls.lower())
            
            if cls == "Building" and r2_mbe["r2"] < 0.05:
                interp = "negligible (rare)"
            elif r2_mbe["r2"] > 0.7:
                interp = "strong"
            elif r2_mbe["r2"] > 0.4:
                interp = "moderate"
            elif r2_mbe["r2"] > 0.2:
                interp = "weak"
            else:
                interp = "none"
            
            bias = "over" if r2_mbe["mbe"] > 0 else "under" if r2_mbe["mbe"] < 0 else "zero"
            print(f"{sfx:<6} {cls:<12} {r2_mbe['r2']:>8.3f} {r2_mbe['mbe']:>10.2f} "
                  f"{bias}estimate {abs(r2_mbe['mbe']):.1f}pp ({interp})")

if __name__ == "__main__":
    main()
