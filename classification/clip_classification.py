#!/usr/bin/env python3
"""
CLIP Classification Experimentation

Runs CLIP classification on SAM segments with configurable:
  - Context-handling approaches (zero, highlight, larger_crop, dual_composite)
  - Aggregation modes (single, average)
  - Prompt versions (via config.PROMPT_VERSION)

Usage:
  # Test run (first 5 images)
  python classification/clip_classification.py --approach zero --test
  python classification/clip_classification.py --approach highlight --test

  # Full run
  python classification/clip_classification.py --approach zero
  python classification/clip_classification.py --approach highlight
  python classification/clip_classification.py --approach larger_crop
  python classification/clip_classification.py --approach dual_composite

  # With aggregation mode
  python classification/clip_classification.py --approach zero --aggregation average
  python classification/clip_classification.py --approach zero --aggregation single
"""

import os
import glob
import time
import argparse
import numpy as np
import torch
from PIL import Image
import pandas as pd
import config
import utils

def main():
    parser = argparse.ArgumentParser(description=" CLIP classification")
    parser.add_argument("--approach", required=True, choices=list(config.APPROACHES.keys()),
                       help="context-handling approach")
    parser.add_argument("--aggregation", choices=["single", "average"], default="single",
                       help="prompt aggregation mode")
    parser.add_argument("--test", action="store_true",
                       help=f"test on first {config.TEST_SAMPLE_SIZE} images")
    args = parser.parse_args()

    approach = args.approach
    aggregation = args.aggregation
    test_mode = args.test
    approach_info = config.APPROACHES[approach]

    output_name = approach if aggregation == "single" else f"{approach}_{aggregation}"

   
    # HEADER
    print("=" * 70)
    print(f"CLIP Classification — {approach.upper()} / {aggregation.upper()}"
          f"{' (TEST)' if test_mode else ''}")
    print("=" * 70)
    print(f"description       : {approach_info['description']}")
    print(f"context preserved : {approach_info['preserves_context']}")
    print(f"prompt version    : {config.PROMPT_VERSION}")
    print(f"aggregation       : {aggregation}")
    print(f"output folder     : {output_name}")
    print("=" * 70)

    # DEVICE
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # VERIFY INPUTS
    if not os.path.exists(config.SEGMENTS_DIR):
        print(f"ERROR: segments dir not found: {config.SEGMENTS_DIR}")
        return
    if not os.path.exists(config.IMAGES_DIR):
        print(f"ERROR: images dir not found: {config.IMAGES_DIR}")
        return

    # FIND SEGMENTS
    segment_files = sorted(glob.glob(os.path.join(config.SEGMENTS_DIR, "*_segments.npz")))
    if not segment_files:
        print(f"ERROR: no segment files found in {config.SEGMENTS_DIR}")
        return

    total_available = len(segment_files)
    if test_mode:
        segment_files = segment_files[:config.TEST_SAMPLE_SIZE]
        print(f"found {total_available} segment files, using first {len(segment_files)}")
    else:
        print(f"found {len(segment_files)} segment files")

    # LOAD CLIP
    print(f"\nloading CLIP: {config.CLIP_MODEL_PATH}")
    clip_model, clip_processor = utils.load_clip_model(config.CLIP_MODEL_NAME,device,cache_dir=config.CLIP_MODEL_PATH)
    print("CLIP loaded\n")

    # OUTPUT DIRECTORIES
    output_dirs = utils.create_output_directories(config.OUTPUT_BASE_DIR, output_name)

    # LOGGING
    log_file = os.path.join(output_dirs["logs"], "processing.log")
    terminal_log = os.path.join(output_dirs["metrics"], config.TERMINAL_LOG_FILE)
    logger = utils.setup_logging(log_file, terminal_log, config.LOG_LEVEL)

    logger.info("=" * 70)
    logger.info(f"CLIP Classification — {approach.upper()} / {aggregation.upper()}"
                f"{' (TEST)' if test_mode else ''}")
    logger.info("=" * 70)
    logger.info(f"city              : {config.CITY_NAME}")
    logger.info(f"approach          : {approach}")
    logger.info(f"aggregation       : {aggregation}")
    logger.info(f"prompt version    : {config.PROMPT_VERSION}")
    logger.info(f"prompts           : {sum(len(p) for p in config.PROMPTS.values())} total")
    logger.info(f"segments dir      : {config.SEGMENTS_DIR}")
    logger.info(f"images dir        : {config.IMAGES_DIR}")
    logger.info(f"output dir        : {output_dirs['base']}")
    logger.info(f"device            : {device}")
    logger.info(f"images to process : {len(segment_files)}"
                f"{f' (of {total_available})' if test_mode else ''}")

    # CROP PARAMETERS
    crop_kwargs = {}
    if approach in ("highlight", "dual_composite"):
        crop_kwargs["highlight_color"] = config.HIGHLIGHT_COLOR
        crop_kwargs["highlight_alpha"] = config.HIGHLIGHT_ALPHA
    elif approach == "larger_crop":
        crop_kwargs["context_factor"] = config.LARGER_CROP_FACTOR

    logger.info(f"crop kwargs       : {crop_kwargs}")

    # CHECKPOINT
    checkpoint_path = os.path.join(output_dirs["base"], "checkpoint.npy")
    processed_images = set()

    if not test_mode:
        checkpoint = utils.load_checkpoint(checkpoint_path)
        if checkpoint:
            processed_images = set(checkpoint.get("processed_images", []))
            logger.info(f"resuming from checkpoint: {len(processed_images)} already processed")

    # MAIN LOOP
    results = []
    run_start = time.time()

    for idx, seg_path in enumerate(segment_files, 1):
        filename = os.path.basename(seg_path)
        base_name = filename.replace("_segments.npz", "")
        class_path = os.path.join(output_dirs["classifications"], f"{base_name}_classification.npy")

        if not test_mode and (filename in processed_images or os.path.exists(class_path)):
            logger.info(f"[{idx}/{len(segment_files)}] skip {base_name} (done)")
            continue

        logger.info(f"[{idx}/{len(segment_files)}] {base_name}")

        try:
            masks = utils.load_segments_from_npz(seg_path)
            logger.info(f"  segments: {len(masks)}")

            img_path = os.path.join(config.IMAGES_DIR, f"{base_name}.png")
            if not os.path.exists(img_path):
                logger.warning(f"  image not found: {img_path}")
                continue

            image_np = np.array(Image.open(img_path).convert("RGB"))
            logger.info(f"  image shape: {image_np.shape}")

            with utils.Timer() as t:
                predictions, confidences, embeddings = utils.classify_segments(
                    image_np=image_np, masks=masks, prompts_dict=config.PROMPTS,
                    clip_model=clip_model, clip_processor=clip_processor,
                    approach=approach, aggregation=aggregation,
                    target_size=config.CLIP_INPUT_SIZE, batch_size=config.CLIP_BATCH_SIZE,
                    **crop_kwargs,
                )

            metrics = utils.calculate_all_metrics(
                masks, predictions, confidences, embeddings,
                image_np.shape[:2], config.CLASS_NAMES,
                config.LOW_CONFIDENCE_THRESHOLD, config.SPATIAL_NEIGHBOR_RADIUS,
            )
            metrics.update({
                "image": base_name,
                "city": config.CITY_NAME,
                "approach": approach,
                "aggregation": aggregation,
                "prompt_version": config.PROMPT_VERSION,
                "processing_time_seconds": round(t.elapsed, 3),
            })
            results.append(metrics)

            utils.save_classification_results(predictions, confidences, embeddings, class_path)

            logger.info(f"  time         : {t.elapsed:.2f}s")
            logger.info(f"  confidence   : {metrics['mean_confidence_overall']:.3f}")
            logger.info(f"  silhouette   : {metrics['silhouette_score']:.3f}")
            logger.info(f"  low-conf %   : {metrics['low_confidence_percentage']:.1f}%")
            logger.info(f"  spatial cons : {metrics['spatial_consistency_score']:.3f}")
            logger.info(f"  veg/bldg/rd  : {metrics['vegetation_count']}/{metrics['building_count']}/{metrics['road_count']}")

            processed_images.add(filename)

            if not test_mode and config.ENABLE_CHECKPOINTING and idx % config.CHECKPOINT_INTERVAL == 0:
                utils.save_checkpoint({"processed_images": list(processed_images)}, checkpoint_path)
                logger.info(f"  checkpoint saved ({idx} images)")

            if idx % config.CLEAR_CACHE_INTERVAL == 0:
                utils.clear_gpu_cache()
                mem = utils.get_memory_usage()
                logger.info(f"  memory: CPU {mem['cpu_used_gb']:.1f}GB GPU {mem.get('gpu_allocated_gb', 0):.1f}GB")

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            continue

    total_time = time.time() - run_start

    # CLEANUP
    if not test_mode and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("checkpoint removed")

    # SAVE METRICS
    if results:
        per_path = os.path.join(output_dirs["metrics"], config.PER_IMAGE_METRICS_FILE)
        sum_path = os.path.join(output_dirs["metrics"], config.SUMMARY_METRICS_FILE)
        utils.save_per_image_metrics(results, per_path)
        utils.save_summary_metrics(results, sum_path, total_time)
        logger.info(f"per-image metrics: {per_path}")
        logger.info(f"summary metrics  : {sum_path}")

    # SUMMARY
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"CLIP Classification COMPLETE — {approach.upper()} / {aggregation.upper()}")
    logger.info("=" * 70)
    logger.info(f"images processed: {len(results)}")
    logger.info(f"total time      : {total_time/60:.1f} min ({total_time:.0f}s)")

    if results:
        df = pd.DataFrame(results)
        logger.info(f"avg confidence  : {df['mean_confidence_overall'].mean():.3f} ± {df['mean_confidence_overall'].std():.3f}")
        logger.info(f"avg silhouette  : {df['silhouette_score'].mean():.3f} ± {df['silhouette_score'].std():.3f}")
        logger.info(f"avg spatial con : {df['spatial_consistency_score'].mean():.3f}")
        logger.info(f"avg low-conf %  : {df['low_confidence_percentage'].mean():.1f}%")
        logger.info(f"avg veg cover % : {df['vegetation_coverage_percent'].mean():.1f}%")
        logger.info(f"avg bldg cover %: {df['building_coverage_percent'].mean():.1f}%")
        logger.info(f"avg road cover %: {df['road_coverage_percent'].mean():.1f}%")
        logger.info(f"avg time/img    : {df['processing_time_seconds'].mean():.2f}s")

    print("\n" + "=" * 70)
    print(f"CLIP Classification COMPLETE — {approach.upper()} / {aggregation.upper()}"
          f"{' (TEST)' if test_mode else ''}")
    print("=" * 70)
    print(f"results: {output_dirs['base']}")
    if test_mode:
        print(f"\ntest run: {len(results)}/{total_available} images")
        print(f"remove --test to run on all {total_available} images")

if __name__ == "__main__":
    main()
