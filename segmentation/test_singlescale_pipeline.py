#!/usr/bin/env python3
"""
Test single-scale segmentation pipeline (exp01 & exp02).
Validates SAM segmentation on 3 sample 225 DPI images before running full experiments.
Output: sam_test_pipeline/single_scale/ with segments and visualizations
"""

import sam_device_patch
import os
import glob
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import config
import utils

N_TEST_IMAGES = 3


def visualize_single_scale(image: np.ndarray, masks: list,
                           title: str, save_path: str):
    """Save side-by-side visualization of original and segmented image."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    overlay = np.zeros((*image.shape[:2], 4))
    for m in sorted(masks, key=lambda x: x["area"], reverse=True):
        overlay[m["segmentation"]] = [*np.random.random(3), 0.6]

    axes[1].imshow(image)
    axes[1].imshow(overlay)
    axes[1].set_title(f"{title} — {len(masks)} segments", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def test_single_scale(mask_generator, results: dict, start_time: float):
    """Test single-scale segmentation on N_TEST_IMAGES 225 DPI images."""
    print("\n" + "=" * 70)
    print("STAGE 1: SINGLE-SCALE SEGMENTATION (225 DPI)")
    print("=" * 70)

    pattern = os.path.join(config.IMAGES_DIR, config.SINGLE_SCALE_PATTERN)
    all_imgs = sorted(glob.glob(pattern))[:N_TEST_IMAGES]

    if not all_imgs:
        print(f"ERROR: no images found matching {pattern}")
        results["single_scale"] = "FAIL"
        return

    output_dirs = {
        "segments": os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "single_scale", "segments"),
        "viz": os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "single_scale", "visualizations"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    for img_path in all_imgs:
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        print(f"\n  {filename}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))

            with utils.Timer() as t:
                masks = mask_generator.generate(image)

            metrics = utils.calculate_segment_metrics(masks)
            print(f"    segments: {metrics['segment_count']}")
            print(f"    coverage: {metrics['pixel_coverage_percent']:.1f}%")
            print(f"    confidence: {metrics['mean_confidence']:.3f}")
            print(f"    time: {t.elapsed:.2f}s")

            seg_path = os.path.join(output_dirs["segments"], f"{base}_segments.npz")
            utils.save_segments_npz(masks, seg_path)
            print(f"    saved: {seg_path}")

            viz_path = os.path.join(output_dirs["viz"], f"test_{filename}")
            visualize_single_scale(image, masks, "single-scale", viz_path)
            print(f"    viz: {viz_path}")

            utils.clear_gpu_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            results["single_scale"] = "FAIL"
            return

    results["single_scale"] = "PASS"
    print(f"\n  single-scale: PASS")


def print_summary(results: dict, summary_path: str, start_time: float):
    """Print and save test summary."""
    elapsed = time.time() - start_time

    lines = [
        "=" * 70,
        "SINGLE-SCALE PIPELINE TEST SUMMARY",
        "=" * 70,
        f"total time: {elapsed:.1f}s",
        "",
        f"single-scale segmentation (225 DPI): {results.get('single_scale', 'NOT RUN')}",
        "",
    ]

    if results.get("single_scale") == "PASS":
        lines.append("PASSED — safe to run exp01 and exp02")
    else:
        lines.append("FAILED — fix errors above before running experiments")

    lines.append("=" * 70)
    output = "\n".join(lines)

    print("\n" + output)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"summary saved: {summary_path}")


def main():
    start_time = time.time()

    print("=" * 70)
    print("SINGLE-SCALE PIPELINE TEST")
    print("=" * 70)
    print(f"images dir: {config.IMAGES_DIR}")
    print(f"SAM checkpoint: {config.SAM_CHECKPOINT}")
    print(f"test images: {N_TEST_IMAGES}")
    print("=" * 70)

    if not os.path.exists(config.SAM_CHECKPOINT):
        print(f"ERROR: SAM checkpoint not found: {config.SAM_CHECKPOINT}")
        return

    if not os.path.exists(config.IMAGES_DIR):
        print(f"ERROR: images directory not found: {config.IMAGES_DIR}")
        return

    print(f"\nloading SAM model: {config.SAM_MODEL_TYPE}")
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[config.SAM_MODEL_TYPE](
        checkpoint=config.SAM_CHECKPOINT
    ).to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.EXP1_PARAMS["points_per_side"],
        pred_iou_thresh=config.EXP1_PARAMS["pred_iou_thresh"],
        stability_score_thresh=config.EXP1_PARAMS["stability_score_thresh"],
        min_mask_region_area=config.EXP1_PARAMS["min_mask_region_area"],
    )
    print(f"SAM loaded on {device}\n")

    results = {}
    test_single_scale(mask_generator, results, start_time)

    summary_path = os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "test_single_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    print_summary(results, summary_path, start_time)


if __name__ == "__main__":
    main()
