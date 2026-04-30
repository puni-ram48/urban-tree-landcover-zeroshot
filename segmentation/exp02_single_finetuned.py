#!/usr/bin/env python3
"""
Experiment 02: Single-scale segmentation with fine-tuned SAM parameters.
Fine-tuned params: higher points_per_side, lower thresholds, minimum region area.
Input: 225 DPI images from IMAGES_DIR
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

# Processing
def process_dataset(images_dir: str, output_dirs: dict,
                    logger, sam_generator) -> list:
    """Run SAM segmentation on all images, save segments and metrics."""
    image_pattern = os.path.join(images_dir, config.SINGLE_SCALE_PATTERN)
    all_images = sorted(glob.glob(image_pattern))

    if not all_images:
        logger.warning(f"no images found matching: {image_pattern}")
        return []

    logger.info(f"found {len(all_images)} images to process")
    logger.info("=" * 70)

    checkpoint_path = os.path.join(output_dirs["base"], "checkpoint.npy")
    checkpoint = utils.load_checkpoint(checkpoint_path)
    processed_images = set(checkpoint.get("processed_images", [])) if checkpoint else set()

    if checkpoint:
        logger.info(f"resuming from checkpoint: {len(processed_images)} already processed")

    results = []

    for idx, img_path in enumerate(all_images, 1):
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        segment_filename = f"{base}_segments.npz"
        segment_path = os.path.join(output_dirs["segments"], segment_filename)

        if filename in processed_images or os.path.exists(segment_path):
            logger.info(f"[{idx}/{len(all_images)}] skip {filename} (done)")
            continue

        logger.info(f"[{idx}/{len(all_images)}] {filename}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))

            with utils.Timer() as t:
                masks = sam_generator.generate(image)

            metrics = utils.calculate_segment_metrics(masks)
            metrics["image"] = filename
            metrics["city"] = config.CITY_NAME
            metrics["processing_time_seconds"] = round(t.elapsed, 3)
            results.append(metrics)

            utils.save_segments_npz(masks, segment_path)

            logger.info(f"  segments: {metrics['segment_count']} | "
                        f"coverage: {metrics['pixel_coverage_percent']:.1f}% | "
                        f"confidence: {metrics['mean_confidence']:.3f}")
            logger.info(f"  time: {t.elapsed:.2f}s | saved: {segment_path}")

            processed_images.add(filename)

            if config.ENABLE_CHECKPOINTING and idx % config.CHECKPOINT_INTERVAL == 0:
                utils.save_checkpoint(
                    {"processed_images": list(processed_images), "last_index": idx},
                    checkpoint_path
                )
                logger.info(f"  checkpoint saved ({idx} images)")

            if idx % config.CLEAR_CACHE_INTERVAL == 0:
                utils.clear_gpu_cache()
                mem = utils.get_memory_usage()
                logger.info(f"  memory: CPU {mem['cpu_used_gb']:.1f}GB / "
                            f"GPU {mem.get('gpu_allocated_gb', 0):.1f}GB")

        except Exception as e:
            logger.error(f"  ERROR {filename}: {e}")
            continue

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("checkpoint removed (processing complete)")

    return results

# Main
def main():
    params = config.EXP2_PARAMS

    print("=" * 70)
    print("EXPERIMENT 02: SINGLE-SCALE SEGMENTATION — FINE-TUNED PARAMETERS")
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
    logger.info("EXPERIMENT 02: SINGLE-SCALE SEGMENTATION — FINE-TUNED PARAMETERS")
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
    logger.info("EXPERIMENT 02 COMPLETE")
    logger.info("=" * 70)
    logger.info(f"images: {len(results)} | time: {total_time/60:.1f}min ({total_time:.0f}s)")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        logger.info(f"segments: {df['segment_count'].mean():.1f} | "
                    f"coverage: {df['pixel_coverage_percent'].mean():.1f}% | "
                    f"confidence: {df['mean_confidence'].mean():.3f}")
        logger.info(f"stability: {df['mean_stability'].mean():.3f} | "
                    f"time/img: {df['processing_time_seconds'].mean():.2f}s")

    print("=" * 70)
    print("EXPERIMENT 02 COMPLETE")
    print(f"results: {output_dirs['base']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
