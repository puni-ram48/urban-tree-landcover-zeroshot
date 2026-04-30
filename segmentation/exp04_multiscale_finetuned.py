#!/usr/bin/env python3
"""
Experiment 04: Multi-scale segmentation with fine-tuned SAM parameters.
Fine-tuned params: higher points_per_side, lower thresholds, minimum region area.
Processes 300+225+150 DPI images, reprojects to 225 DPI via PGW, fuses masks.
Best-performing configuration; provides input for CLIP classification.
Input: {devEUI}_1.png (300), {devEUI}_2.png (225), {devEUI}_3.png (150)
Output: segments/.npz, metrics/.csv, terminal_log.txt
"""

import sam_device_patch
import os
import glob
import time
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import config
import utils

# Scale suffix -> scale name mapping
SUFFIX_TO_SCALE = {
    "_1.png": "high",    # 300 DPI
    "_2.png": "medium",  # 225 DPI  <-- fusion reference
    "_3.png": "low",     # 150 DPI
}

REQUIRED_SCALES = ["high", "medium", "low"]


# Processing
def process_dataset(images_dir: str, output_dirs: dict,
                    logger, sam_generator) -> list:
    """Run multi-scale SAM segmentation, fuse masks to 225 DPI reference."""
    all_images = []
    for pattern in config.MULTISCALE_PATTERNS.values():
        all_images.extend(glob.glob(os.path.join(images_dir, pattern)))
    all_images = sorted(set(all_images))

    if not all_images:
        logger.warning(f"no images found in: {images_dir}")
        return []

    image_groups = utils.group_images_by_prefix(
        [os.path.basename(f) for f in all_images]
    )

    logger.info(f"found {len(all_images)} images in {len(image_groups)} groups")
    logger.info("=" * 70)

    checkpoint_path = os.path.join(output_dirs["base"], "checkpoint.npy")
    checkpoint = utils.load_checkpoint(checkpoint_path)
    processed_groups = set(checkpoint.get("processed_groups", [])) if checkpoint else set()

    if checkpoint:
        logger.info(f"resuming from checkpoint: {len(processed_groups)} groups already processed")

    results = []
    group_ids = sorted(image_groups.keys())

    for idx, group_id in enumerate(group_ids, 1):
        fused_filename = f"{group_id}_2_segments.npz"
        fused_path = os.path.join(output_dirs["segments"], fused_filename)

        if group_id in processed_groups or os.path.exists(fused_path):
            logger.info(f"[{idx}/{len(group_ids)}] skip {group_id} (done)")
            continue

        logger.info(f"[{idx}/{len(group_ids)}] {group_id}")

        try:
            images_dict = {}
            pgw_dict = {}

            for img_file in image_groups[group_id]:
                img_path = os.path.join(images_dir, img_file)
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
                logger.warning(f"  missing scales {missing} — skipping")
                continue

            masks_by_scale = {}
            total_raw = 0

            with utils.Timer() as segment_timer:
                for scale_name in REQUIRED_SCALES:
                    raw_masks = sam_generator.generate(images_dict[scale_name])
                    masks_by_scale[scale_name] = raw_masks
                    total_raw += len(raw_masks)

            logger.info(f"  segmented {total_raw} masks in {segment_timer.elapsed:.2f}s")

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

            logger.info(f"  fused to {len(fused_masks)} masks in {fusion_timer.elapsed:.2f}s")

            total_time = segment_timer.elapsed + fusion_timer.elapsed
            metrics = utils.calculate_segment_metrics(fused_masks)
            metrics["image"] = f"{group_id}_2.png"
            metrics["city"] = config.CITY_NAME
            metrics["processing_time_seconds"] = round(total_time, 3)
            results.append(metrics)

            utils.save_segments_npz(fused_masks, fused_path)

            logger.info(f"  segments: {metrics['segment_count']} | "
                        f"coverage: {metrics['pixel_coverage_percent']:.1f}% | "
                        f"confidence: {metrics['mean_confidence']:.3f}")
            logger.info(f"  time: {total_time:.2f}s | saved: {fused_path}")

            processed_groups.add(group_id)

            if config.ENABLE_CHECKPOINTING and idx % config.CHECKPOINT_INTERVAL == 0:
                utils.save_checkpoint(
                    {"processed_groups": list(processed_groups), "last_index": idx},
                    checkpoint_path
                )
                logger.info(f"  checkpoint saved ({idx} groups)")

            if idx % config.CLEAR_CACHE_INTERVAL == 0:
                utils.clear_gpu_cache()
                mem = utils.get_memory_usage()
                logger.info(f"  memory: CPU {mem['cpu_used_gb']:.1f}GB / "
                            f"GPU {mem.get('gpu_allocated_gb', 0):.1f}GB")

        except Exception as e:
            logger.error(f"  ERROR {group_id}: {e}")
            continue

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("checkpoint removed (processing complete)")

    return results

# Main
def main():
    params = config.EXP4_PARAMS

    print("=" * 70)
    print("EXPERIMENT 04: MULTI-SCALE SEGMENTATION — FINE-TUNED PARAMETERS")
    print("=" * 70)
    for key, val in params.items():
        print(f"{key}: {val}")
    print("=" * 70)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    if not os.path.exists(config.SAM_CHECKPOINT):
        print(f"ERROR: SAM checkpoint not found: {config.SAM_CHECKPOINT}")
        return

    if not os.path.exists(config.IMAGES_DIR):
        print(f"ERROR: images directory not found: {config.IMAGES_DIR}")
        return

    print(f"loading SAM model: {config.SAM_MODEL_TYPE}")
    sam = sam_model_registry[config.SAM_MODEL_TYPE](
        checkpoint=config.SAM_CHECKPOINT
    ).to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=params["points_per_side"],
        pred_iou_thresh=params["pred_iou_thresh"],
        stability_score_thresh=params["stability_score_thresh"],
        min_mask_region_area=params["min_mask_region_area"],
    )
    print("SAM model loaded\n")

    output_dirs = utils.create_output_directories(
        config.OUTPUT_BASE_DIR,
        params["name"],
    )

    log_file = os.path.join(output_dirs["logs"], "processing.log")
    terminal_log_file = os.path.join(output_dirs["metrics"], config.TERMINAL_LOG_FILE)
    logger = utils.setup_logging(log_file, terminal_log_file, config.LOG_LEVEL)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 04: MULTI-SCALE SEGMENTATION — FINE-TUNED PARAMETERS")
    logger.info("=" * 70)
    logger.info(f"city: {config.CITY_NAME} | images: {config.IMAGES_DIR}")
    logger.info(f"output: {output_dirs['base']} | device: {device}")
    logger.info(f"SAM: {config.SAM_CHECKPOINT}")
    logger.info(f"params: {params}")
    logger.info("=" * 70)

    run_start = time.time()
    results = process_dataset(config.IMAGES_DIR, output_dirs, logger, mask_generator)
    total_time = time.time() - run_start

    if results:
        per_image_path = os.path.join(output_dirs["metrics"], config.PER_IMAGE_METRICS_FILE)
        summary_path = os.path.join(output_dirs["metrics"], config.SUMMARY_METRICS_FILE)
        utils.save_per_image_metrics(results, per_image_path)
        utils.save_summary_metrics(results, summary_path, total_time)
        logger.info(f"metrics: {per_image_path}")
        logger.info(f"summary: {summary_path}")

    logger.info("=" * 70)
    logger.info("EXPERIMENT 04 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"groups: {len(results)} | time: {total_time/60:.1f}min ({total_time:.0f}s)")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        logger.info(f"segments: {df['segment_count'].mean():.1f} | "
                    f"coverage: {df['pixel_coverage_percent'].mean():.1f}% | "
                    f"confidence: {df['mean_confidence'].mean():.3f}")
        logger.info(f"stability: {df['mean_stability'].mean():.3f} | "
                    f"time/group: {df['processing_time_seconds'].mean():.2f}s")

    print("=" * 70)
    print("EXPERIMENT 04 COMPLETE")
    print(f"results: {output_dirs['base']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
