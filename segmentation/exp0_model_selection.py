#!/usr/bin/env python3
"""
SAM Model Selection Experiment (E0)
====================================

Evaluates three SAM model variants (ViT-B, ViT-L, ViT-H) on a 15-image subset
at 225 DPI to select the appropriate model capacity for the full segmentation
pipeline.

This script reproduces Experiment E0 from the thesis:
  - Compares inference time, segment count, and quality metrics across models
  - Uses SAM's default automatic mask generation parameters
  - Outputs per-image metrics and final model comparison summary

Default SAM parameters (from Kirillov et al. 2023):
    points_per_side        : 64
    pred_iou_thresh        : 0.80
    stability_score_thresh : 0.85
    min_mask_region_area   : 100

Metrics computed per image:
    - Pixel coverage (%)           — union of all predicted masks
    - Mean solidity                — mask area / convex hull area
    - Mean predicted IoU           — SAM confidence scores
    - Mean stability score         — mask consistency metric
    - Segment count                — number of masks predicted
    - Processing time (s/image)    — inference speed per model

Output files:
    outputs/sam_model_selection_finetuned/
        ├── model_selection_per_image.csv      (15 rows × metrics)
        ├── model_selection_final.csv          (3 rows: summary per model)

Usage:
    python sam_model_selection.py

Requirements:
    pip install segment-anything torch torchvision opencv-python numpy pandas scipy scikit-image

References:
    Kirillov et al. (2023) "Segment Anything"
    Paper: https://arxiv.org/abs/2304.02643
    Code: https://github.com/facebookresearch/segment-anything
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import ConvexHull

try:
    import sam_device_patch
except ImportError:
    pass  # Optional device-specific optimization

# CONFIGURATION — Update paths for your environment
BASE_DIR = "/path/to/project/root"

# Data paths
IMAGES_DIR = os.path.join(
    BASE_DIR, 
    "data/<dataset_name>/<images_folder>"
)

# SAM checkpoint path
SAM_CHECKPOINT = os.path.join(BASE_DIR, "models/sam_model/sam_vit_h.pth")

# Output directory
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "<output_folder>/segmentation/outputs")

# Dataset configuration
CITY_NAME = "erlangen"
EVAL_SUFFIX = "_2"  # 225 DPI images
N_IMAGES = 15

# Experiment E0: SAM model selection
E0_PARAMS = {
    "name": "exp00_model_selection",
    "description": "SAM model capacity comparison (ViT-B, ViT-L, ViT-H)",
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100,
}

# Model variants to evaluate
MODELS = {
    "ViT-B": {
        "model_type": "vit_b",
        "params_M": 91,
        "checkpoint": os.path.join(BASE_DIR, "models/sam_model/sam_vit_b.pth"),
    },
    "ViT-L": {
        "model_type": "vit_l",
        "params_M": 308,
        "checkpoint": os.path.join(BASE_DIR, "models/sam_model/sam_vit_l.pth"),
    },
    "ViT-H": {
        "model_type": "vit_h",
        "params_M": 636,
        "checkpoint": os.path.join(BASE_DIR, "models/sam_model/sam_vit_h.pth"),
    },
}

# Processing configuration
DEVICE = "cuda"  # or "cpu"
CLEAR_CACHE_INTERVAL = 3

# Metrics configuration
METRICS_TO_COMPUTE = [
    "segment_count",
    "pixel_coverage_percent",
    "mean_solidity",
    "mean_pred_iou",
    "mean_stability",
    "mean_area",
    "median_area",
    "processing_time_seconds",
]

# Output subdirectory names
METRICS_SUBDIR = "sam_model_selection"
PER_IMAGE_METRICS_FILE = "model_selection_per_image.csv"
SUMMARY_METRICS_FILE = "model_selection_summary.csv"
TERMINAL_LOG_FILE = "model_selection_log.txt"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
