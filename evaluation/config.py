#!/usr/bin/env python3
"""
Configuration for Ring-Based Buffer Evaluation — Pixel Ground Truth

Evaluates CLIP classification predictions against pixel-level ground truth
rasterized from QGIS vector shapefiles (.npz masks).

Ground truth source:
  Rasterized directly from shapefiles via rasterio — NOT from CSV areas
  or annotated colors. Each .npz contains per-pixel binary masks:
  vegetation, building, road (uint8: 0 or 1)

Ring definitions (annuli, NOT full circles):
  2.5m:  0.0–2.5 m  → area = 19.635 m²
  5.0m:  2.5–5.0 m  → area = 58.905 m²
  7.5m:  5.0–7.5 m  → area = 98.175 m²

Folder structure expected:
  <project_root>/
  ├── data/<dataset>/<images>/           ← original aerial images
  ├── data/<dataset>/pixel_groundtruth/  ← .npz GT masks
  └── <output>/
      ├── segmentation/outputs/exp04_multiscale_finetuned/segments/
      ├── classification/outputs/stage1_context/<approach>/classifications/
      └── evaluation/outputs/<experiment>/

File naming conventions:
  GT .npz:        {devEUI}{suffix}_gt.npz
  CLIP .npy:      {devEUI}{suffix}_classification.npy
  SAM .npz:       {devEUI}{suffix}_segments.npz
  Images:         {devEUI}{suffix}.png

Usage:
  import config
  To switch experiments: update BASE_DIR, paths, and EXPERIMENT_NAME only
"""

import os
import math


# PATHS
BASE_DIR = "/path/to/project/root"

# Input data
IMAGES_DIR = os.path.join(BASE_DIR, "data/<dataset>/<images_folder>")
PIXEL_GT_DIR = os.path.join(BASE_DIR, "data/<dataset>/pixel_groundtruth_<dpi>")
SEGMENTS_DIR = os.path.join(BASE_DIR, "<output>/segmentation/outputs/exp04_multiscale_finetuned/segments")
CLASSIFICATIONS_DIR = os.path.join(BASE_DIR, "<output>/classification/outputs/stage1_context/<approach>/classifications")

# Output
EXPERIMENT_NAME = "<experiment_identifier>"
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, f"<output>/evaluation/outputs/{EXPERIMENT_NAME}")


# IMAGE & SCALE CONFIGURATION
EVAL_SUFFIX = "_2"        # 225 DPI reference suffix
EVAL_DPI = 225
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
CENTER_ROW = IMAGE_HEIGHT // 2   # 512 — tree at center
CENTER_COL = IMAGE_WIDTH // 2    # 512
SCALE = 500               # QGIS export scale 1:500

# Meters per pixel (must match pixel_gt_generator formula)
METERS_PER_PIXEL = ((IMAGE_WIDTH / EVAL_DPI) * 0.0254 * SCALE) / IMAGE_WIDTH

# RING DEFINITIONS
RINGS = [
    {
        "suffix": "2m5",
        "inner_m": 0.0,
        "outer_m": 2.5,
        "theory_m2": math.pi * 2.5 ** 2,
    },
    {
        "suffix": "5m0",
        "inner_m": 2.5,
        "outer_m": 5.0,
        "theory_m2": math.pi * 5.0 ** 2 - math.pi * 2.5 ** 2,
    },
    {
        "suffix": "7m5",
        "inner_m": 5.0,
        "outer_m": 7.5,
        "theory_m2": math.pi * 7.5 ** 2 - math.pi * 5.0 ** 2,
    },
]

# CLASSES
CLASS_NAMES = ["Vegetation", "Building", "Road"]

# GT .npz keys (must match pixel_gt_generator output)
GT_NPZ_KEYS = {
    "Vegetation": "vegetation",
    "Building": "building",
    "Road": "road",
}

# COLORS
# CLIP prediction overlay (matches classification config.py)
CLIP_CLASS_COLORS_RGBA = {
    "Vegetation": [0.059, 0.416, 0.196, 0.6],  # dark green #0F6A32
    "Building": [0.624, 0.184, 0.184, 0.6],    # dark red #9F2F2F
    "Road": [0.365, 0.349, 0.349, 0.6],        # dark gray #5D5959
}

# Ground truth overlay (darker shades for distinction from CLIP predictions)
GT_CLASS_COLORS_RGBA = {
    "Vegetation": [0.0, 0.502, 0.0, 0.6],      # dark green #008000
    "Building": [0.502, 0.0, 0.0, 0.6],        # dark red #800000
    "Road": [0.365, 0.349, 0.349, 0.6],        # dark gray #5D5959
}

# Ring visualization
RING_CIRCLE_COLOR = "black"
RING_LINEWIDTH = 2.0

# OUTPUT
PER_TREE_METRICS_FILE = "per_tree_metrics_pixelgt.csv"
SUMMARY_FILE = "summary_pixelgt.csv"
FINAL_SUMMARY_FILE = "final_summary_pixelgt.csv"

# DATASET
CITY_NAME = "erlangen"
