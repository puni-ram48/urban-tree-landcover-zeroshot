#!/usr/bin/env python3
"""
Configuration for Stage 1 — CLIP Classification Experiments

Configurable parameters:
  - Context-handling approaches (zero, highlight, larger_crop, dual_composite)
  - Prompt versions (0, 1, 1.1, 1.2, 1.3)

Pipeline: SAM segments (.npz) -> CLIP classification -> metrics CSV

Usage:
  To switch dataset: update IMAGES_DIR, SEGMENTS_DIR, CITY_NAME, OUTPUT_BASE_DIR
  To switch prompt version: change PROMPT_VERSION
  To switch context approach: pass --approach flag to exp01_clip_classification.py
"""

import os

# PATHS
BASE_DIR = "/path/to/project/root"

IMAGES_DIR = os.path.join(BASE_DIR, "data/<dataset_name>/<images_folder>")
SEGMENTS_DIR = os.path.join(BASE_DIR, "<output_folder>/segmentation/outputs/exp04_multiscale_finetuned/segments")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models/clip_model/clip-vit-large-patch14-336/")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "<output_folder>/classification/outputs/stage1_context")


# DATASET
CITY_NAME = "erlangen"
EVAL_SUFFIX = "_2"  # 225 DPI reference


# PROMPTS — SELECT ACTIVE VERSION
PROMPT_VERSION = "1.3"  # Options: "0", "1", "1.1", "1.2", "1.3"

PROMPT_VERSIONS = {
    "0": {
        "Vegetation": ["a photo of vegetation"],
        "Road": ["a photo of a road"],
        "Building": ["a photo of a building"],
    },
    "1": {
        "Vegetation": ["green trees or bushes seen from above", "aerial view of a grassy lawn or field", "leafy tree canopy from above"],
        "Road": ["long narrow strip of asphalt for cars from above", "gray road or street with lane markings from above", "aerial view of a paved road surface"],
        "Building": ["red tile rooftop of a residential building from above", "gray or flat rooftop of a building seen from above", "aerial view of a house or building with roof edges and corners"],
    },
    "1.1": {
        "Vegetation": ["green trees or bushes seen from above", "aerial view of a grassy lawn or field", "leafy tree canopy from above"],
        "Road": ["long narrow strip of asphalt for cars from above", "gray road or street with lane markings from above", "aerial view of a paved road surface"],
        "Building": ["red tile rooftop made of clay from above", "gray concrete or metal rooftop from above", "building roof with roofing material seen from above"],
    },
    "1.2": {
        "Vegetation": ["green trees or bushes seen from above", "aerial view of a grassy lawn or field", "leafy tree canopy from above", "vegetation with green yellow or brown colors from above"],
        "Road": ["long narrow strip of asphalt for cars from above", "gray road or street with lane markings from above", "aerial view of a paved road surface"],
        "Building": ["red tile rooftop made of clay from above", "gray concrete or metal rooftop from above", "building roof with roofing material seen from above"],
    },
    "1.3": {
        "Vegetation": ["green trees or bushes seen from above", "aerial view of a grassy lawn or field", "leafy tree canopy from above", "natural vegetation in green or brown from above"],
        "Road": ["long narrow strip of asphalt for cars from above", "gray road or street with lane markings from above", "aerial view of a paved road surface", "dark gray pavement for vehicles from above"],
        "Building": ["red tile rooftop made of clay from above", "gray flat rooftop made of concrete from above", "building roof with shingles or tiles from above"],
    },
}

PROMPTS = PROMPT_VERSIONS[PROMPT_VERSION]

# CLIP
CLIP_INPUT_SIZE = 336
CLIP_BATCH_SIZE = 16
DEVICE = "cuda"

# CLASSES
CLASS_NAMES = ["Vegetation", "Building", "Road"]
CLASS_COLORS = {
    "Vegetation": [0.0, 1.0, 0.039, 0.6],
    "Building": [1.0, 0.196, 0.0, 0.6],
    "Road": [1.0, 0.843, 0.0, 0.6],
}

# APPROACHES
APPROACHES = {
    "zero": {"description": "isolated segment, black background", "preserves_context": False},
    "highlight": {"description": "full image with cyan highlight", "preserves_context": True},
    "larger_crop": {"description": "3x bounding box with context", "preserves_context": True},
    "dual_composite": {"description": "side-by-side global + local", "preserves_context": True},
}

HIGHLIGHT_COLOR = (0, 255, 255)
HIGHLIGHT_ALPHA = 0.35
LARGER_CROP_FACTOR = 3

# METRICS
METRICS_TO_COMPUTE = [
    "mean_confidence_overall", "mean_confidence_vegetation", "mean_confidence_building", "mean_confidence_road",
    "std_confidence_overall", "min_confidence", "max_confidence", "low_confidence_percentage",
    "total_segments", "vegetation_count", "building_count", "road_count",
    "vegetation_coverage_percent", "building_coverage_percent", "road_coverage_percent",
    "spatial_consistency_score", "class_transition_count", "isolation_ratio",
    "silhouette_score", "silhouette_score_vegetation", "silhouette_score_building", "silhouette_score_road",
    "processing_time_seconds",
]

LOW_CONFIDENCE_THRESHOLD = 0.3
SPATIAL_NEIGHBOR_RADIUS = 50

# PROCESSING
TEST_SAMPLE_SIZE = 5
CLEAR_CACHE_INTERVAL = 5
CHECKPOINT_INTERVAL = 20
ENABLE_CHECKPOINTING = True

# OUTPUT
CLASSIFICATIONS_SUBDIR = "classifications"
METRICS_SUBDIR = "metrics"
LOGS_SUBDIR = "logs"
PER_IMAGE_METRICS_FILE = "per_image_metrics.csv"
SUMMARY_METRICS_FILE = "summary_metrics.csv"
TERMINAL_LOG_FILE = "terminal_log.txt"
COMPARISON_SUMMARY_FILE = "stage1_approach_comparison.csv"

# LOGGING & VISUALIZATION
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
VIZ_DPI = 150
VIZ_FIGSIZE = (20, 6)
VIZ_ALPHA = 0.6
