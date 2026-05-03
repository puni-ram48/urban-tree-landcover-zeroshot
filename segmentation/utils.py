#!/usr/bin/env python3
"""
Utility functions for SAM segmentation pipeline.
Covers: logging, metrics, I/O, GPU management, checkpoints, PGW parsing, mask fusion.
"""
import os
import sys
import time
import logging
import numpy as np
import torch
import psutil
from typing import List, Dict, Tuple, Optional
from skimage.measure import regionprops, label
import pandas as pd
from collections import defaultdict

# Logging
def setup_logging(log_file: str, terminal_log_file: str,
                  log_level: str = "INFO") -> logging.Logger:
    """Configure logging to file and terminal simultaneously."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, log_level))
    fh.setFormatter(formatter)

    th = logging.FileHandler(terminal_log_file, mode="w", encoding="utf-8")
    th.setLevel(getattr(logging, log_level))
    th.setFormatter(formatter)
                    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level))
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(th)
    logger.addHandler(ch)

    return logger

# GPU and memory management
def clear_gpu_cache():
    """Clear GPU memory cache to prevent out-of-memory errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> Dict[str, float]:
    """Return current CPU and GPU memory usage (in GB)."""
    stats = {
        "cpu_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "cpu_available_gb": psutil.virtual_memory().available / (1024 ** 3),
    }
    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        stats["gpu_reserved_gb"]  = torch.cuda.memory_reserved() / (1024 ** 3)
    return stats

# Shape metrics
def calculate_solidity(mask: np.ndarray) -> float:
    """Calculate solidity (mask area / convex hull area). Range: [0, 1]."""
    try:
        props = regionprops(label(mask.astype(int)))
        return props[0].solidity if props else 0.0
    except Exception:
        return 0.0

def calculate_aspect_ratio(mask: np.ndarray) -> float:
    """Calculate aspect ratio (height / width). ~1.0 = square, >1.0 = elongated."""
    try:
        props = regionprops(label(mask.astype(int)))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            width = maxc - minc
            return (maxr - minr) / width if width > 0 else 0.0
        return 0.0
    except Exception:
        return 0.0

def calculate_compactness(mask: np.ndarray) -> float:
    """Calculate compactness (4π·area/perimeter²). Circle = 1.0, irregular < 1.0."""
    try:
        props = regionprops(label(mask.astype(int)))
        if props:
            area      = props[0].area
            perimeter = props[0].perimeter
            return (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        return 0.0
    except Exception:
        return 0.0

# Segment metrics
def calculate_segment_metrics(masks: List[Dict]) -> Dict[str, float]:
    """Compute quality and shape metrics for SAM masks."""
    empty = {
        "segment_count": 0,
        "mean_confidence": 0.0,
        "std_confidence": 0.0,
        "mean_stability": 0.0,
        "std_stability": 0.0,
        "mean_solidity": 0.0,
        "std_solidity": 0.0,
        "pixel_coverage_percent": 0.0,
        "mean_area":  0.0,
        "median_area": 0.0,
        "std_area": 0.0,
        "min_area": 0.0,
        "max_area": 0.0,
        "mean_aspect_ratio": 0.0,
        "mean_compactness": 0.0,
    }
    if not masks:
        return empty
      
    confidences = [m["predicted_iou"] for m in masks]
    stabilities = [m["stability_score"] for m in masks]
    areas = [m["area"] for m in masks]

    solidities, aspect_ratios, compactnesses = [], [], []
    for m in masks:
        if m["area"] > 100:
            seg = m["segmentation"]
            solidities.append(calculate_solidity(seg))
            aspect_ratios.append(calculate_aspect_ratio(seg))
            compactnesses.append(calculate_compactness(seg))

    coverage = np.zeros(masks[0]["segmentation"].shape, dtype=bool)
    for m in masks:
        coverage |= m["segmentation"]
    pixel_coverage = float(np.sum(coverage) / coverage.size * 100)

    return {
        "segment_count": len(masks),
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "mean_stability": float(np.mean(stabilities)),
        "std_stability": float(np.std(stabilities)),
        "mean_solidity": float(np.mean(solidities)) if solidities else 0.0,
        "std_solidity": float(np.std(solidities)) if solidities else 0.0,
        "pixel_coverage_percent": pixel_coverage,
        "mean_area": float(np.mean(areas)),
        "median_area": float(np.median(areas)),
        "std_area": float(np.std(areas)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
        "mean_compactness": float(np.mean(compactnesses)) if compactnesses else 0.0,
    }



# File I/O — segments
def save_segments_npz(masks: List[Dict], filepath: str):
    """ Save SAM segmentation masks to a compressed .npz file """
    filepath = os.path.splitext(filepath)[0] + ".npz"
    np.savez_compressed(
        filepath,
        segmentations = np.array([m["segmentation"] for m in masks]),
        areas = np.array([m["area"] for m in masks]),
        bboxes = np.array([m["bbox"] for m in masks]),
        predicted_ious = np.array([m["predicted_iou"] for m in masks]),
        stability_scores = np.array([m["stability_score"] for m in masks]),
    )

def load_segments_npz(filepath: str) -> List[Dict]:
    """Load SAM masks from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    masks = []
    for i in range(len(data["segmentations"])):
        masks.append({
            "segmentation": data["segmentations"][i],
            "area": int(data["areas"][i]),
            "bbox": data["bboxes"][i].tolist(),
            "predicted_iou": float(data["predicted_ious"][i]),
            "stability_score": float(data["stability_scores"][i]),
        })
    return masks

# File I/O — metrics
def save_per_image_metrics(metrics_list: List[Dict], output_path: str):
    """Save per-image metrics to CSV."""
    pd.DataFrame(metrics_list).to_csv(output_path, index=False)

def save_summary_metrics(metrics_list: List[Dict], output_path: str,
                         total_time_seconds: float):
    """Save summary metrics (mean, std, min, max) to CSV."""
    df = pd.DataFrame(metrics_list)
    numeric = df.select_dtypes(include=[np.number])

    summary = pd.DataFrame({
        "metric": numeric.columns,
        "mean": numeric.mean().values,
        "std": numeric.std().values,
        "min": numeric.min().values,
        "max": numeric.max().values,
    })

    total_row = pd.DataFrame([{
        "metric": "total_run_time_seconds",
        "mean": round(total_time_seconds, 2),
        "std": "",
        "min": "",
        "max": "",
    }])
    summary = pd.concat([summary, total_row], ignore_index=True)
    summary.to_csv(output_path, index=False)

# Output directories
def create_output_directories(base_dir: str,
                              experiment_name: str) -> Dict[str, str]:
    """Create standardized output directory structure for one experiment."""
    exp_dir = os.path.join(base_dir, experiment_name)

    dirs = {
        "base": exp_dir,
        "segments": os.path.join(exp_dir, "segments"),
        "metrics": os.path.join(exp_dir, "metrics"),
        "visualizations": os.path.join(exp_dir, "visualizations"),
        "logs": os.path.join(exp_dir, "logs"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs

# Checkpoint
def save_checkpoint(state: Dict, checkpoint_path: str):
    """Save processing state for resuming interrupted runs."""
    np.save(checkpoint_path, state, allow_pickle=True)


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load processing state from checkpoint."""
    if os.path.exists(checkpoint_path):
        return np.load(checkpoint_path, allow_pickle=True).item()
    return None

# Timer
class Timer:
    """Context manager for measuring elapsed time."""
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

# PGW world file parsing
def parse_pgw(pgw_path: str) -> Dict[str, float]:
    """
    Parse a PGW world file into a georeferencing parameter dictionary.
    PGW line order:
        1  pixel_size_x   (metres per pixel, x direction)
        2  rotation_y     (always 0 for north-up images)
        3  rotation_x     (always 0 for north-up images)
        4  pixel_size_y   (negative metres per pixel, y direction)
        5  top_left_x     (x coordinate of upper-left pixel centre)
        6  top_left_y     (y coordinate of upper-left pixel centre)
    """
    with open(pgw_path, "r") as f:
        lines = f.readlines()
    return {
        "pixel_size_x": float(lines[0].strip()),
        "rotation_y": float(lines[1].strip()),
        "rotation_x": float(lines[2].strip()),
        "pixel_size_y": float(lines[3].strip()),
        "top_left_x": float(lines[4].strip()),
        "top_left_y": float(lines[5].strip()),
    }

# Coordinate transformation
def pixel_to_world(row: int, col: int,
                   pgw: Dict[str, float]) -> Tuple[float, float]:
    """Convert pixel (row, col) to world (x, y) coordinates."""
    x = pgw["top_left_x"] + col * pgw["pixel_size_x"]
    y = pgw["top_left_y"] + row * pgw["pixel_size_y"]
    return x, y

def world_to_pixel(x: float, y: float,
                   pgw: Dict[str, float]) -> Tuple[int, int]:
    """Convert world (x, y) coordinates to pixel (row, col)."""
    col = (x - pgw["top_left_x"]) / pgw["pixel_size_x"]
    row = (y - pgw["top_left_y"]) / pgw["pixel_size_y"]
    return int(round(row)), int(round(col))

def transform_mask_to_target(mask: np.ndarray,
                             source_pgw: Dict,
                             target_pgw: Dict,
                             target_shape: Tuple[int, int]) -> np.ndarray:
    """Reproject mask from source to target image coordinates using PGW."""
    h_tgt, w_tgt = target_shape
    transformed = np.zeros(target_shape, dtype=bool)
    rows, cols = np.where(mask)

    for row, col in zip(rows, cols):
        x, y = pixel_to_world(row, col, source_pgw)
        tgt_row, tgt_col = world_to_pixel(x, y, target_pgw)
        if 0 <= tgt_row < h_tgt and 0 <= tgt_col < w_tgt:
            transformed[tgt_row, tgt_col] = True

    return transformed

# Multi-scale fusion utilities
def compute_bbox(mask: np.ndarray) -> List[int]:
    """Compute axis-aligned bounding box [x_min, y_min, width, height]."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return [0, 0, 0, 0]
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def merge_overlapping_masks(masks: List[Dict],
                            threshold: float = 0.5) -> List[Dict]:
    """Merge masks exceeding overlap threshold, average metadata."""
    if not masks:
        return []

    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    merged_indices = set()
    final_masks = []

    for i, parent_dict in enumerate(sorted_masks):
        if i in merged_indices:
            continue

        parent = parent_dict["segmentation"].copy()
        meta_list = [parent_dict]

        for j in range(i + 1, len(sorted_masks)):
            if j in merged_indices:
                continue
            child = sorted_masks[j]["segmentation"]
            intersect = np.logical_and(parent, child)
            if not np.any(intersect):
                continue
            overlap = np.sum(intersect) / sorted_masks[j]["area"]
            if overlap > threshold:
                parent = np.logical_or(parent, child)
                merged_indices.add(j)
                meta_list.append(sorted_masks[j])

        final_masks.append({
            "segmentation": parent,
            "area": int(np.sum(parent)),
            "predicted_iou": float(np.mean([m["predicted_iou"] for m in meta_list])),
            "stability_score": float(np.mean([m["stability_score"] for m in meta_list])),
            "bbox": compute_bbox(parent),
        })

    return final_masks


def fuse_multiscale_masks(masks_by_scale: Dict[str, List[Dict]],
                          pgw_by_scale:   Dict[str, Dict],
                          target_scale:   str,
                          target_shape:   Tuple[int, int],
                          min_pixels:     int   = 50,
                          merge_threshold: float = 0.3) -> List[Dict]:
    """
    Fuse masks from multiple DPI scales into a single mask set at the
    target scale resolution.
    Steps:
        1. Reproject all masks to target image coordinates via PGW files.
        2. Discard masks with fewer than min_pixels after reprojection.
        3. Merge overlapping masks with merge_overlapping_masks().
    """
    target_pgw = pgw_by_scale[target_scale]
    all_transformed = []

    for scale_name, masks in masks_by_scale.items():
        source_pgw = pgw_by_scale[scale_name]
        for mask_dict in masks:
            seg = mask_dict["segmentation"]
            if scale_name == target_scale:
                transformed = seg
            else:
                transformed = transform_mask_to_target(
                    seg, source_pgw, target_pgw, target_shape
                )
            pixel_count = int(np.sum(transformed))
            if pixel_count >= min_pixels:
                all_transformed.append({
                    "segmentation": transformed,
                    "area": pixel_count,
                    "predicted_iou": mask_dict["predicted_iou"],
                    "stability_score": mask_dict["stability_score"],
                    "bbox": compute_bbox(transformed),
                    "source_scale": scale_name,
                })
    return merge_overlapping_masks(all_transformed, threshold=merge_threshold)

# Image grouping
def group_images_by_prefix(image_files: List[str]) -> Dict[str, List[str]]:
    """
    Group image filenames by their devEUI prefix.
    Expected naming: {devEUI}_{suffix}.png
    Example: 8C1F640980000014_2.png  ->  prefix = '8C1F640980000014'
    """
    groups: Dict[str, List[str]] = defaultdict(list)

    for img in image_files:
        prefix = os.path.splitext(img)[0].rsplit("_", 1)[0]
        groups[prefix].append(img)

    return {prefix: sorted(files) for prefix, files in groups.items()}
