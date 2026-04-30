#!/usr/bin/env python3
"""
Generate visualizations from SAM segmentation results.
Creates sample visualizations for individual experiments and cross-experiment comparisons.
Output: visualizations/ subdirectory in each experiment folder + comparisons/ folder
"""

import os
import glob
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import config
import utils

EXPERIMENTS = [
    "exp01_single_default",
    "exp02_single_finetuned",
    "exp03_multiscale_default",
    "exp04_multiscale_finetuned",
    "exp05_multiscale_225_150_dpi",
    "exp06_multiscale_300_225_dpi",
]

EXPERIMENT_LABELS = {
    "exp01_single_default": "E1: Single-Scale (Default)",
    "exp02_single_finetuned": "E2: Single-Scale (Fine-tuned)",
    "exp03_multiscale_default": "E3: Multi-Scale (Default)",
    "exp04_multiscale_finetuned": "E4: Multi-Scale (Fine-tuned)",
    "exp05_multiscale_225_150_dpi": "E5: 225+150 DPI",
    "exp06_multiscale_300_225_dpi": "E6: 300+225 DPI",
}


def load_segments_npz(npz_path: str) -> dict:
    """Load SAM segments from .npz file."""
    data = np.load(npz_path, allow_pickle=False)
    return {
        "masks": data["segmentations"],
        "areas": data["areas"],
        "predicted_ious": data["predicted_ious"],
        "stability_scores": data.get("stability_scores", np.ones(len(data["segmentations"]))),
    }


def visualize_single_experiment(image: np.ndarray, segments_data: dict,
                                exp_name: str, save_path: str, max_masks: int = 50):
    """Create single visualization with original + overlay + metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Segmentation overlay
    overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    masks = segments_data["masks"]
    areas = segments_data["areas"]
    n_masks_total = len(masks)

    sorted_indices = np.argsort(areas)[::-1][:max_masks]
    np.random.seed(42)
    for idx in sorted_indices:
        mask = masks[idx]
        color = np.random.random(3)
        overlay[mask] = [*color, 0.5]

    axes[1].imshow(image)
    axes[1].imshow(overlay)
    axes[1].set_title(f"SAM Segmentation\n{len(sorted_indices)}/{n_masks_total} masks shown",
                      fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Metrics
    mean_iou = np.mean(segments_data["predicted_ious"])
    mean_stability = np.mean(segments_data["stability_scores"])
    coverage = np.sum(areas) / (image.shape[0] * image.shape[1]) * 100

    stats_text = f"Metrics:\n"
    stats_text += f"Total masks: {n_masks_total}\n"
    stats_text += f"Mean IoU: {mean_iou:.3f}\n"
    stats_text += f"Mean stability: {mean_stability:.3f}\n"
    stats_text += f"Coverage: {coverage:.1f}%\n"
    stats_text += f"Mean area: {np.mean(areas):.0f} px²"

    axes[2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                 family="monospace")
    axes[2].set_title(f"{EXPERIMENT_LABELS[exp_name]}\nMetrics", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_comparison_grid(image: np.ndarray, exp_segments: dict,
                              dev_eui: str, save_path: str, max_masks: int = 40):
    """Create comparison grid showing all 6 experiments + original."""
    n_exps = len(exp_segments)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Panel [0,0]: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("ORIGINAL", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panels [0,1-3] and [1,0-2]: Experiments
    all_exps = sorted(exp_segments.keys())
    for panel_idx, exp_name in enumerate(all_exps):
        if panel_idx < 3:
            ax = axes[0, panel_idx + 1]
        else:
            ax = axes[1, panel_idx - 3]

        if exp_name not in exp_segments or exp_segments[exp_name] is None:
            ax.text(0.5, 0.5, "NOT FOUND", ha="center", va="center", fontsize=10)
            ax.set_title(EXPERIMENT_LABELS[exp_name], fontsize=10)
            ax.axis("off")
            continue

        try:
            segments_data = exp_segments[exp_name]
            overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)
            masks = segments_data["masks"]
            areas = segments_data["areas"]
            n_masks = len(masks)

            sorted_indices = np.argsort(areas)[::-1][:max_masks]
            np.random.seed(42)
            for idx in sorted_indices:
                mask = masks[idx]
                color = np.random.random(3)
                overlay[mask] = [*color, 0.5]

            ax.imshow(image)
            ax.imshow(overlay)

            mean_iou = np.mean(segments_data["predicted_ious"])
            title = f"{EXPERIMENT_LABELS[exp_name]}\n"
            title += f"{len(sorted_indices)}/{n_masks} | IoU: {mean_iou:.2f}"
            ax.set_title(title, fontsize=10, fontweight="bold")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha="center", va="center", fontsize=9)
            ax.set_title(EXPERIMENT_LABELS[exp_name], fontsize=10)

        ax.axis("off")

    # Panel [1,3]: Summary
    axes[1, 3].axis("off")
    summary_text = f"Tree ID: {dev_eui}\n\n"
    summary_text += "Comparison Summary:\n"
    for exp_name in all_exps:
        if exp_name in exp_segments and exp_segments[exp_name] is not None:
            n_masks = len(exp_segments[exp_name]["masks"])
            summary_text += f"{exp_name[:4]}: {n_masks} masks\n"
    axes[1, 3].text(0.1, 0.5, summary_text, fontsize=9, verticalalignment="center",
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
    axes[1, 3].set_title("Summary", fontsize=11, fontweight="bold")

    plt.suptitle(f"Cross-Experiment Comparison: {dev_eui}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_experiment(experiment_name: str, num_samples: int):
    """Generate visualizations for a single experiment."""
    print(f"\n{'='*70}")
    print(f"Visualizing: {EXPERIMENT_LABELS[experiment_name]}")
    print(f"{'='*70}")

    segments_dir = os.path.join(config.OUTPUT_BASE_DIR, experiment_name, "segments")
    viz_dir = os.path.join(config.OUTPUT_BASE_DIR, experiment_name, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    if not os.path.exists(segments_dir):
        print(f"ERROR: segments directory not found: {segments_dir}")
        return 0

    segment_files = sorted(glob.glob(os.path.join(segments_dir, "*_segments.npz")))

    if not segment_files:
        print(f"No segment files found in {segments_dir}")
        return 0

    if len(segment_files) > num_samples:
        segment_files = random.sample(segment_files, num_samples)

    print(f"Creating visualizations for {len(segment_files)} images...")

    successful = 0
    for seg_path in tqdm(segment_files, desc="Processing"):
        try:
            basename = os.path.basename(seg_path)
            dev_eui = basename.replace("_segments.npz", "")
            image_path = os.path.join(config.IMAGES_DIR, f"{dev_eui}.png")

            if not os.path.exists(image_path):
                continue

            image = np.array(Image.open(image_path).convert("RGB"))
            segments_data = load_segments_npz(seg_path)

            viz_path = os.path.join(viz_dir, f"viz_{dev_eui}.png")
            visualize_single_experiment(image, segments_data, experiment_name, viz_path)
            successful += 1

        except Exception as e:
            continue

    print(f"✓ Created {successful} visualizations")
    print(f"Output: {viz_dir}")
    return successful


def create_comparison_grids(num_comparisons: int):
    """Create cross-experiment comparison grids."""
    print(f"\n{'='*70}")
    print("Creating Cross-Experiment Comparison Grids")
    print(f"{'='*70}")

    comparison_dir = os.path.join(config.OUTPUT_BASE_DIR, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    common_images = None
    for exp in EXPERIMENTS:
        segments_dir = os.path.join(config.OUTPUT_BASE_DIR, exp, "segments")
        if not os.path.exists(segments_dir):
            continue

        exp_images = set([
            os.path.basename(f).replace("_segments.npz", "")
            for f in glob.glob(os.path.join(segments_dir, "*_segments.npz"))
        ])

        if common_images is None:
            common_images = exp_images
        else:
            common_images = common_images.intersection(exp_images)

    if not common_images:
        print("No common images found across all experiments")
        return 0

    common_images = sorted(list(common_images))
    if len(common_images) > num_comparisons:
        common_images = random.sample(common_images, num_comparisons)

    print(f"Creating {len(common_images)} comparison grids...")

    successful = 0
    for dev_eui in tqdm(common_images, desc="Processing"):
        try:
            image_path = os.path.join(config.IMAGES_DIR, f"{dev_eui}.png")
            if not os.path.exists(image_path):
                continue

            image = np.array(Image.open(image_path).convert("RGB"))

            exp_segments = {}
            for exp in EXPERIMENTS:
                seg_path = os.path.join(
                    config.OUTPUT_BASE_DIR, exp, "segments", f"{dev_eui}_segments.npz"
                )
                if os.path.exists(seg_path):
                    exp_segments[exp] = load_segments_npz(seg_path)
                else:
                    exp_segments[exp] = None

            comparison_path = os.path.join(comparison_dir, f"comparison_{dev_eui}.png")
            visualize_comparison_grid(image, exp_segments, dev_eui, comparison_path)
            successful += 1

        except Exception as e:
            continue

    print(f"✓ Created {successful} comparison grids")
    print(f"Output: {comparison_dir}")
    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations from SAM segmentation results"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to visualize (e.g., exp01_single_default)",
    )
    parser.add_argument(
        "--all_experiments",
        action="store_true",
        help="Generate visualizations for all experiments",
    )
    parser.add_argument(
        "--comparisons",
        action="store_true",
        help="Create cross-experiment comparison grids",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample images per experiment (default: 5)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SAM SEGMENTATION VISUALIZATION GENERATOR")
    print("=" * 70)
    print(f"Images directory: {config.IMAGES_DIR}")
    print(f"Output base directory: {config.OUTPUT_BASE_DIR}")
    print("=" * 70)

    if not os.path.exists(config.IMAGES_DIR):
        print(f"ERROR: Images directory not found: {config.IMAGES_DIR}")
        return

    total_viz = 0

    if args.all_experiments:
        print(f"\nGenerating visualizations for all experiments ({len(EXPERIMENTS)} total)...\n")
        for exp in EXPERIMENTS:
            total_viz += visualize_experiment(exp, args.num_samples)

    elif args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS)}")
            return
        total_viz += visualize_experiment(args.experiment, args.num_samples)

    else:
        print("Please specify --experiment NAME or --all_experiments")
        return

    if args.comparisons:
        total_viz += create_comparison_grids(args.num_samples)

    print("\n" + "=" * 70)
    print(f"COMPLETE: Generated {total_viz} visualizations")
    print("=" * 70)


if __name__ == "__main__":
    main()
