#!/usr/bin/env python3
"""
Test multi-scale segmentation pipeline (exp03-exp06).
Validates PGW parsing, multi-scale fusion on 3 sample image groups before full experiments.
Output: sam_test_pipeline/multiscale/ with segments and visualizations
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

N_TEST_GROUPS = 3

SUFFIX_TO_SCALE = {
    "_1.png": "high",
    "_2.png": "medium",
    "_3.png": "low",
}

REQUIRED_SCALES = ["high", "medium", "low"]


def visualize_multiscale(images_dict: dict, masks_by_scale: dict,
                         fused_masks: list, group_id: str, save_path: str):
    """Save visualization of all scales and fused result."""
    scales = ["high", "medium", "low"]
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for col, scale in enumerate(scales):
        if scale not in images_dict:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        img = images_dict[scale]
        masks = masks_by_scale.get(scale, [])

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{scale} scale", fontsize=10)
        axes[0, col].axis("off")

        overlay = np.zeros((*img.shape[:2], 4))
        for m in masks:
            overlay[m["segmentation"]] = [*np.random.random(3), 0.6]

        axes[1, col].imshow(img)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{len(masks)} segments", fontsize=10)
        axes[1, col].axis("off")

    med = images_dict["medium"]
    axes[0, 3].imshow(med)
    axes[0, 3].set_title("fused (225 DPI)", fontsize=10, fontweight="bold")
    axes[0, 3].axis("off")

    fused_overlay = np.zeros((*med.shape[:2], 4))
    for m in fused_masks:
        fused_overlay[m["segmentation"]] = [*np.random.random(3), 0.6]

    axes[1, 3].imshow(med)
    axes[1, 3].imshow(fused_overlay)
    axes[1, 3].set_title(f"{len(fused_masks)} fused", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle(f"multi-scale test — group {group_id}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def test_multiscale(mask_generator, results: dict, start_time: float):
    """Test multi-scale segmentation and fusion on N_TEST_GROUPS image groups."""
    print("\n" + "=" * 70)
    print("STAGE 2: MULTI-SCALE SEGMENTATION (300 + 225 + 150 DPI)")
    print("=" * 70)

    all_images = []
    for pattern in config.MULTISCALE_PATTERNS.values():
        all_images.extend(glob.glob(os.path.join(config.IMAGES_DIR, pattern)))
    all_images = sorted(set(all_images))

    image_groups = utils.group_images_by_prefix(
        [os.path.basename(f) for f in all_images]
    )
    test_groups = sorted(image_groups.keys())[:N_TEST_GROUPS]

    if not test_groups:
        print("ERROR: no image groups found")
        results["multiscale"] = "FAIL"
        return

    output_dirs = {
        "segments": os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "multiscale", "segments"),
        "viz": os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "multiscale", "visualizations"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    for group_id in test_groups:
        print(f"\n  group: {group_id}")

        try:
            images_dict = {}
            pgw_dict = {}

            for img_file in image_groups[group_id]:
                img_path = os.path.join(config.IMAGES_DIR, img_file)
                scale_name = next(
                    (v for k, v in SUFFIX_TO_SCALE.items() if img_file.endswith(k)),
                    None
                )
                if scale_name is None:
                    continue

                images_dict[scale_name] = np.array(Image.open(img_path).convert("RGB"))

                pgw_path = img_path.replace(".png", ".pgw")
                if not os.path.exists(pgw_path):
                    raise FileNotFoundError(f"missing PGW: {pgw_path}")
                pgw_dict[scale_name] = utils.parse_pgw(pgw_path)

            missing = [s for s in REQUIRED_SCALES if s not in images_dict]
            if missing:
                print(f"    WARNING: missing scales {missing} — skipping")
                continue

            masks_by_scale = {}
            total_raw = 0

            with utils.Timer() as segment_timer:
                for scale_name in REQUIRED_SCALES:
                    raw_masks = mask_generator.generate(images_dict[scale_name])
                    masks_by_scale[scale_name] = raw_masks
                    total_raw += len(raw_masks)

            print(f"    segmented {total_raw} masks in {segment_timer.elapsed:.2f}s")

            target_shape = images_dict["medium"].shape[:2]

            with utils.Timer() as fusion_timer:
                fused_masks = utils.fuse_multiscale_masks(
                    masks_by_scale=masks_by_scale,
                    pgw_by_scale=pgw_dict,
                    target_scale="medium",
                    target_shape=target_shape,
                    min_pixels=config.EXP4_PARAMS["fusion_min_pixels"],
                    merge_threshold=config.EXP4_PARAMS["fusion_merge_threshold"],
                )

            print(f"    fused to {len(fused_masks)} masks in {fusion_timer.elapsed:.2f}s")

            metrics = utils.calculate_segment_metrics(fused_masks)
            print(f"    coverage: {metrics['pixel_coverage_percent']:.1f}%")
            print(f"    confidence: {metrics['mean_confidence']:.3f}")

            seg_path = os.path.join(output_dirs["segments"], f"{group_id}_2_segments.npz")
            utils.save_segments_npz(fused_masks, seg_path)
            print(f"    saved: {seg_path}")

            viz_path = os.path.join(output_dirs["viz"], f"test_group_{group_id}.png")
            visualize_multiscale(images_dict, masks_by_scale, fused_masks, group_id, viz_path)
            print(f"    viz: {viz_path}")

            utils.clear_gpu_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            results["multiscale"] = "FAIL"
            return

    results["multiscale"] = "PASS"
    print(f"\n  multiscale: PASS")


def print_summary(results: dict, summary_path: str, start_time: float):
    """Print and save test summary."""
    elapsed = time.time() - start_time

    lines = [
        "=" * 70,
        "MULTI-SCALE PIPELINE TEST SUMMARY",
        "=" * 70,
        f"total time: {elapsed:.1f}s",
        "",
        f"multi-scale fusion (300+225+150 DPI): {results.get('multiscale', 'NOT RUN')}",
        "",
    ]

    if results.get("multiscale") == "PASS":
        lines.append("PASSED — safe to run exp03, exp04, exp05, exp06")
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
    print("MULTI-SCALE PIPELINE TEST")
    print("=" * 70)
    print(f"images dir: {config.IMAGES_DIR}")
    print(f"SAM checkpoint: {config.SAM_CHECKPOINT}")
    print(f"test groups: {N_TEST_GROUPS}")
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
        points_per_side=config.EXP4_PARAMS["points_per_side"],
        pred_iou_thresh=config.EXP4_PARAMS["pred_iou_thresh"],
        stability_score_thresh=config.EXP4_PARAMS["stability_score_thresh"],
        min_mask_region_area=config.EXP4_PARAMS["min_mask_region_area"],
    )
    print(f"SAM loaded on {device}\n")

    results = {}
    test_multiscale(mask_generator, results, start_time)

    summary_path = os.path.join(config.OUTPUT_BASE_DIR, "sam_test_pipeline", "test_multiscale_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    print_summary(results, summary_path, start_time)


if __name__ == "__main__":
    main()
