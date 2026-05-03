#!/usr/bin/env python3
"""
Configuration for SAM aerial tree segmentation pipeline.
Six experiments: single-scale (exp01-02) and multi-scale (exp03-06) with parameter tuning.
To switch datasets: update IMAGES_DIR, CITY_NAME, OUTPUT_BASE_DIR.
"""
import os

# Project root directory
BASE_DIR = "/path/to/project/root"

# Data paths (relative to project root)
IMAGES_DIR = os.path.join(BASE_DIR, "dataset/<images_folder>")
GROUND_TRUTH = os.path.join(BASE_DIR, "dataset/<ground_truth_file>")
SAM_CHECKPOINT = os.path.join(BASE_DIR, "models/sam_model/sam_vit_h.pth")
SAM_MODEL_TYPE = "vit_h"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "<output_folder>")

# Dataset configuration
CITY_NAME = "erlangen"
N_TREES = 96

# Image naming: {devEUI}_{suffix}.png (e.g., 8C1F640980000014_2.png)
RESOLUTION_SUFFIX = {
    "300dpi": "_1",   
    "225dpi": "_2",   # reference
    "150dpi": "_3",  
}

SINGLE_SCALE_SUFFIX = "_2"
SINGLE_SCALE_PATTERN = "*_2.png"

MULTISCALE_PATTERNS = {
    "high": "*_1.png",
    "medium": "*_2.png",   # fusion reference
    "low": "*_3.png",
}

FUSION_TARGET = "medium"  # 225 DPI

# Experiment 01: single-scale, default
EXP1_PARAMS = {
    "name": "exp01_single_default",
    "description": "Single-scale 225 DPI with SAM default parameters",
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "min_mask_region_area": 0,
}

# Experiment 02: single-scale, fine-tuned
EXP2_PARAMS = {
    "name": "exp02_single_finetuned",
    "description": "Single-scale 225 DPI with fine-tuned parameters",
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100,
}

# Experiment 03: multi-scale 300+225+150 DPI, default parameters
EXP3_PARAMS = {
    "name": "exp03_multiscale_default",
    "description": "Multi-scale fusion (300+225+150 DPI) with SAM default parameters",
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "min_mask_region_area": 0,
    "fusion_merge_threshold": 0.60,   
    "fusion_min_pixels": 50,   
}

# Experiment 04: multi-scale 300+225+150 DPI, fine-tuned parameters
EXP4_PARAMS = {
    "name": "exp04_multiscale_finetuned",
    "description": "Multi-scale fusion (300+225+150 DPI) with fine-tuned parameters",
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100,
    "fusion_merge_threshold": 0.60,
    "fusion_min_pixels": 50,
}

# Experiment 05: multi-scale 225+150 DPI, fine-tuned parameters
EXP5_PARAMS = {
    "name": "exp05_multiscale_225_150_dpi",
    "description": "Multi-scale fusion (225+150 DPI only) with fine-tuned parameters",
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100,
    "fusion_merge_threshold": 0.60,
    "fusion_min_pixels": 50,
}

EXP5_MULTISCALE_PATTERNS = {
    "medium": "*_2.png",   # fusion reference
    "low": "*_3.png",
}

# Experiment 06: multi-scale 300+225 DPI, fine-tuned parameters
EXP6_PARAMS = {
    "name": "exp06_multiscale_300_225_dpi",
    "description": "Multi-scale fusion (300+225 DPI only) with fine-tuned parameters",
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100,
    "fusion_merge_threshold": 0.60,
    "fusion_min_pixels": 50,
}

EXP6_MULTISCALE_PATTERNS = {
    "high": "*_1.png",
    "medium": "*_2.png",   # fusion reference
}

# Processing configuration
DEVICE = "cuda"  # or "cpu" if CUDA unavailable
CLEAR_CACHE_INTERVAL = 5
CHECKPOINT_INTERVAL  = 20
ENABLE_CHECKPOINTING = True

# Metrics configuration
METRICS_TO_COMPUTE = [
    "segment_count", "mean_confidence", "std_confidence",
    "mean_stability", "std_stability", "mean_solidity", "std_solidity",
    "pixel_coverage_percent", "mean_area", "median_area", "std_area",
    "min_area", "max_area", "mean_aspect_ratio", "mean_compactness",
    "processing_time_seconds",
]

# Output subdirectory names
SEGMENTS_SUBDIR = "segments"
METRICS_SUBDIR = "metrics"
LOGS_SUBDIR = "logs"
VISUALIZATIONS_SUBDIR = "visualizations"

PER_IMAGE_METRICS_FILE = "per_image_metrics.csv"
SUMMARY_METRICS_FILE = "summary_metrics.csv"
TERMINAL_LOG_FILE = "terminal_log.txt"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Visualization settings
VIZ_DPI = 300
VIZ_FIGSIZE = (18, 6)
VIZ_ALPHA = 0.6

# Test configuration
TEST_SAMPLE_SIZE = 3
TEST_ENABLE_VISUALIZATION = True
