#!/usr/bin/env python3
"""
Stage 1 Visualization — Classification Results

Creates PNG visualizations from saved classification results.

Two modes:
  --approach {name}
    Visualize one approach. Produces 3-panel figure per image:
      original | class overlay | confidence heatmap

  --compare
    Compare all four approaches side-by-side. Shows raw SAM segments
    plus all approaches for each image.

Usage:
  # Visualize one approach
  python visualize_samples.py --approach zero
  python visualize_samples.py --approach highlight

  # Compare all approaches
  python visualize_samples.py --compare
  python visualize_samples.py --compare --max_images 5
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pandas as pd
import config
import utils

# HELPERS
def build_class_overlay(masks: list, predictions: list, image_shape: tuple) -> np.ndarray:
    """Build RGBA overlay with segments colored by predicted class."""
    overlay = np.zeros((*image_shape, 4), dtype=np.float32)
    areas = [int(np.sum(m)) for m in masks]
    order = np.argsort(areas)[::-1]
    for i in order:
        cls = predictions[i]
        if cls in config.CLASS_COLORS:
            overlay[masks[i]] = config.CLASS_COLORS[cls]
    return overlay

def build_confidence_heatmap(masks: list, confidences: list, image_shape: tuple) -> np.ndarray:
    """Build per-pixel confidence map."""
    heatmap = np.zeros(image_shape, dtype=np.float32)
    areas = [int(np.sum(m)) for m in masks]
    order = np.argsort(areas)[::-1]
    for i in order:
        heatmap[masks[i]] = confidences[i]
    return heatmap

def build_segmentation_overlay(masks: list, image_shape: tuple, alpha: float = 0.6,
                               seed: int = 42) -> np.ndarray:
    """Build SAM segmentation overlay with random colors per segment."""
    overlay = np.zeros((*image_shape, 4), dtype=np.float32)
    areas = [int(np.sum(m)) for m in masks]
    order = np.argsort(areas)[::-1]
    rng = np.random.default_rng(seed)
    for i in order:
        colour = rng.random(3)
        overlay[masks[i]] = [*colour, alpha]
    return overlay

def find_image_path(base_name: str) -> str:
    """Resolve path to original aerial image."""
    candidate = os.path.join(config.IMAGES_DIR, f"{base_name}.png")
    return candidate if os.path.exists(candidate) else ""

def find_segments_path(base_name: str) -> str:
    """Resolve path to SAM segments file."""
    candidate = os.path.join(config.SEGMENTS_DIR, f"{base_name}_segments.npz")
    return candidate if os.path.exists(candidate) else ""

def get_processed_base_names(approach: str, aggregation: str = "single") -> list:
    """List all base names classified under given approach."""
    suffix = "" if aggregation == "single" else f"_{aggregation}"
    cls_dir = os.path.join(config.OUTPUT_BASE_DIR, f"{approach}{suffix}", "classifications")
    if not os.path.exists(cls_dir):
        return []
    files = sorted(glob.glob(os.path.join(cls_dir, "*_classification.npy")))
    return [os.path.basename(f).replace("_classification.npy", "") for f in files]

def legend_patches():
    """Build matplotlib legend patches for target classes."""
    return [mpatches.Patch(color=config.CLASS_COLORS[cls][:3], label=cls,
                          alpha=config.CLASS_COLORS[cls][3])
           for cls in config.CLASS_NAMES]

# SINGLE APPROACH
def visualize_single_approach(approach: str, aggregation: str = "single", max_images: int = None):
    """Create per-image visualizations for one approach."""
    print("=" * 70)
    print(f"VISUALIZE: {approach.upper()} / {aggregation.upper()}")
    print("=" * 70)

    if approach not in config.APPROACHES:
        print(f"ERROR: unknown approach '{approach}'")
        return

    viz_dir = os.path.join(config.OUTPUT_BASE_DIR, "visualizations", f"{approach}_{aggregation}")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"output: {viz_dir}")

    base_names = get_processed_base_names(approach, aggregation)
    if not base_names:
        print(f"ERROR: no results for {approach}/{aggregation}")
        print(f"run exp01_clip_classification.py --approach {approach} first")
        return

    if max_images:
        base_names = base_names[:max_images]

    print(f"visualizing {len(base_names)} images\n")

    suffix = "" if aggregation == "single" else f"_{aggregation}"
    cls_dir = os.path.join(config.OUTPUT_BASE_DIR, f"{approach}{suffix}", "classifications")

    for idx, base_name in enumerate(base_names, 1):
        print(f"[{idx}/{len(base_names)}] {base_name}")

        img_path = find_image_path(base_name)
        seg_path = find_segments_path(base_name)
        cls_path = os.path.join(cls_dir, f"{base_name}_classification.npy")

        if not img_path or not seg_path:
            print(f"  skip: missing image or segments")
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        masks = utils.load_segments_from_npz(seg_path)
        predictions, confidences, _ = utils.load_classification_results(cls_path)

        class_overlay = build_class_overlay(masks, predictions, image_np.shape[:2])
        confidence_heatmap = build_confidence_heatmap(masks, confidences, image_np.shape[:2])

        counts = {cls: sum(1 for p in predictions if p == cls) for cls in config.CLASS_NAMES}
        mean_conf = np.mean(confidences) if confidences else 0.0

        fig, axes = plt.subplots(1, 3, figsize=config.VIZ_FIGSIZE)

        axes[0].imshow(image_np)
        axes[0].set_title("Original", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(image_np)
        axes[1].imshow(class_overlay)
        axes[1].set_title(f"Classification\nV:{counts['Vegetation']} B:{counts['Building']} R:{counts['Road']}",
                         fontsize=12, fontweight="bold")
        axes[1].axis("off")
        axes[1].legend(handles=legend_patches(), loc="lower right", fontsize=9)

        im = axes[2].imshow(confidence_heatmap, cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title(f"Confidence\nMean: {mean_conf:.3f}", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"{base_name} — {approach}/{aggregation}",
                    fontsize=13, fontweight="bold", y=1.02)

        out_path = os.path.join(viz_dir, f"{base_name}_viz.png")
        plt.savefig(out_path, dpi=config.VIZ_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")

    print(f"\nDONE — {len(base_names)} images saved to {viz_dir}")

# COMPARISON
def _render_approach_panel(ax, image_np, masks, cls_path, title):
    """Render one classification panel."""
    preds, confs, _ = utils.load_classification_results(cls_path)
    overlay = build_class_overlay(masks, preds, image_np.shape[:2])
    counts = {cls: sum(1 for p in preds if p == cls) for cls in config.CLASS_NAMES}
    mean_conf = float(np.mean(confs)) if confs else 0.0

    ax.imshow(image_np)
    ax.imshow(overlay)
    ax.set_title(f"{title}\nconf {mean_conf:.2f} V:{counts['Vegetation']} B:{counts['Building']} R:{counts['Road']}",
                fontsize=11, fontweight="bold")
    ax.axis("off")

    return {"mean_confidence": round(mean_conf, 4), "vegetation_count": counts["Vegetation"],
            "building_count": counts["Building"], "road_count": counts["Road"]}

def _aggregation_modes_available() -> list:
    """Detect available aggregation modes."""
    approaches = list(config.APPROACHES.keys())
    modes = []
    if all(get_processed_base_names(a, "single") for a in approaches):
        modes.append("single")
    if all(get_processed_base_names(a, "average") for a in approaches):
        modes.append("average")
    return modes

def _common_images_across(aggregation: str) -> set:
    """Return images classified under ALL approaches."""
    approaches = list(config.APPROACHES.keys())
    sets = [set(get_processed_base_names(a, aggregation)) for a in approaches]
    return set.intersection(*sets) if all(sets) else set()

def visualize_comparison(max_images: int = None):
    """Create side-by-side comparison for all approaches."""
    print("=" * 70)
    print("VISUALIZE COMPARISON — ALL APPROACHES")
    print("=" * 70)

    approaches = list(config.APPROACHES.keys())
    modes = _aggregation_modes_available()

    if not modes:
        print("ERROR: no classification results found")
        print("run exp01_clip_classification.py for each approach first")
        return

    print(f"aggregation modes: {modes}")

    image_sets = []
    for mode in modes:
        image_sets.append(_common_images_across(mode))

    common = sorted(set.intersection(*image_sets)) if image_sets else []
    if not common:
        print("ERROR: no images classified under all approaches")
        return

    if max_images:
        common = common[:max_images]

    print(f"common images: {len(common)}")

    viz_dir = os.path.join(config.OUTPUT_BASE_DIR, "visualizations", "comparison")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"output: {viz_dir}\n")

    comparison_stats = []

    for idx, base_name in enumerate(common, 1):
        print(f"[{idx}/{len(common)}] {base_name}")

        img_path = find_image_path(base_name)
        seg_path = find_segments_path(base_name)

        if not img_path or not seg_path:
            print(f"  skip: missing image or segments")
            continue

        image_np = np.array(Image.open(img_path).convert("RGB"))
        masks = utils.load_segments_from_npz(seg_path)

        n_rows = len(modes)
        fig, axes = plt.subplots(n_rows, 6, figsize=(30, 6 * n_rows), squeeze=False)
        row_stats = {"image": base_name}

        for row_idx, mode in enumerate(modes):
            row = axes[row_idx]

            if row_idx == 0:
                row[0].imshow(image_np)
                row[0].set_title(f"Original\n{len(masks)} segments", fontsize=12, fontweight="bold")
                row[0].axis("off")

                seg_overlay = build_segmentation_overlay(masks, image_np.shape[:2])
                row[1].imshow(image_np)
                row[1].imshow(seg_overlay)
                row[1].set_title(f"SAM\n{len(masks)} segments", fontsize=12, fontweight="bold")
                row[1].axis("off")
            else:
                row[0].axis("off")
                row[1].axis("off")

            row[0].text(-0.1, 0.5, mode.upper(), transform=row[0].transAxes,
                       fontsize=14, fontweight="bold", rotation=90, va="center", ha="right")

            for col, approach in enumerate(approaches, start=2):
                cls_path = os.path.join(config.OUTPUT_BASE_DIR, f"{approach}_{mode}",
                                       "classifications", f"{base_name}_classification.npy")
                panel_stats = _render_approach_panel(row[col], image_np, masks, cls_path,
                                                     f"{approach}/{mode}")
                key_prefix = f"{approach}_{mode}"
                row_stats[f"{key_prefix}_conf"] = panel_stats["mean_confidence"]
                row_stats[f"{key_prefix}_veg"] = panel_stats["vegetation_count"]
                row_stats[f"{key_prefix}_bldg"] = panel_stats["building_count"]
                row_stats[f"{key_prefix}_road"] = panel_stats["road_count"]

        fig.legend(handles=legend_patches(), loc="lower center", ncol=3, fontsize=11,
                  bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"{base_name} — Stage 1 Comparison", fontsize=14, fontweight="bold", y=1.00)

        out_path = os.path.join(viz_dir, f"{base_name}_comparison.png")
        plt.savefig(out_path, dpi=config.VIZ_DPI, bbox_inches="tight")
        plt.close(fig)

        comparison_stats.append(row_stats)
        print(f"  saved: {out_path}")

    if comparison_stats:
        summary_df = pd.DataFrame(comparison_stats)
        summary_path = os.path.join(config.OUTPUT_BASE_DIR, config.COMPARISON_SUMMARY_FILE)
        summary_df.to_csv(summary_path, index=False)
        print(f"\ncomparison CSV: {summary_path}")

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    agg_rows = []
    for mode in modes:
        for approach in approaches:
            metrics_csv = os.path.join(config.OUTPUT_BASE_DIR, f"{approach}_{mode}",
                                      "metrics", config.PER_IMAGE_METRICS_FILE)
            if not os.path.exists(metrics_csv):
                continue
            df = pd.read_csv(metrics_csv)
            agg_rows.append({
                "approach": approach,
                "aggregation": mode,
                "n_images": len(df),
                "mean_confidence": round(df["mean_confidence_overall"].mean(), 4),
                "std_confidence": round(df["mean_confidence_overall"].std(), 4),
                "low_conf_pct": round(df["low_confidence_percentage"].mean(), 2),
                "spatial_cons": round(df["spatial_consistency_score"].mean(), 4),
                "veg_cov%": round(df["vegetation_coverage_percent"].mean(), 2),
                "bldg_cov%": round(df["building_coverage_percent"].mean(), 2),
                "road_cov%": round(df["road_coverage_percent"].mean(), 2),
                "time_per_img": round(df["processing_time_seconds"].mean(), 2),
            })

    if agg_rows:
        agg_df = pd.DataFrame(agg_rows)
        print("\n" + agg_df.to_string(index=False))
        agg_path = os.path.join(config.OUTPUT_BASE_DIR, "stage1_aggregate_summary.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"\naggregate summary: {agg_path}")

    print(f"\nDONE — {len(common)} comparison images saved")

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 classification visualization",
        epilog="examples:\n"
               "  python visualize_samples.py --approach zero\n"
               "  python visualize_samples.py --compare\n"
               "  python visualize_samples.py --compare --max_images 5"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--approach", choices=list(config.APPROACHES.keys()),
                     help="visualize one approach")
    mode.add_argument("--compare", action="store_true",
                     help="compare all approaches side-by-side")
    parser.add_argument("--aggregation", choices=["single", "average"], default="single",
                       help="aggregation mode (default: single)")
    parser.add_argument("--max_images", type=int, default=None,
                       help="maximum images to visualize")
    args = parser.parse_args()

    if args.compare:
        visualize_comparison(max_images=args.max_images)
    else:
        visualize_single_approach(args.approach, args.aggregation, max_images=args.max_images)

if __name__ == "__main__":
    main()
