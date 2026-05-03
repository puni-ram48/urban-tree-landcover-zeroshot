#!/usr/bin/env python3
"""
Utility Functions for Stage 1 Classification

Shared functions for exp01_clip_classification.py and visualize_samples.py:
  - Logging, GPU memory, timing
  - Directory and checkpoint I/O
  - Segment loading, CLIP model loading
  - Four crop-preparation strategies
  - Classification and metrics computation
  - Results I/O (.npy and .csv)
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import psutil
import pandas as pd
from typing import List, Dict, Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


# LOGGING
def setup_logging(log_file: str, terminal_log_file: str,
                  log_level: str = "INFO") -> logging.Logger:
    """Configure logging to file, terminal log, and stdout."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for handler in [
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.FileHandler(terminal_log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setLevel(getattr(logging, log_level))
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger

# GPU & TIMING
def clear_gpu_cache():
    """Free GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage() -> Dict[str, float]:
    """Return current CPU and GPU memory usage in GB."""
    stats = {
        "cpu_used_gb": psutil.virtual_memory().used / (1024 ** 3),
        "cpu_available_gb": psutil.virtual_memory().available / (1024 ** 3),
    }
    if torch.cuda.is_available():
        stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
    return stats

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start

# DIRECTORIES & CHECKPOINTS
def create_output_directories(base_dir: str, approach_name: str) -> Dict[str, str]:
    """Create output directory structure for one approach."""
    exp_dir = os.path.join(base_dir, approach_name)
    dirs = {
        "base": exp_dir,
        "classifications": os.path.join(exp_dir, "classifications"),
        "metrics": os.path.join(exp_dir, "metrics"),
        "logs": os.path.join(exp_dir, "logs"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def save_checkpoint(state: Dict, checkpoint_path: str):
    """Save processing state for resuming interrupted runs."""
    np.save(checkpoint_path, state, allow_pickle=True)

def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        return np.load(checkpoint_path, allow_pickle=True).item()
    return None

# SEGMENT & MODEL LOADING
def load_segments_from_npz(npz_path: str) -> List[np.ndarray]:
    """Load boolean segment masks from SAM .npz file."""
    data = np.load(npz_path)
    return [seg for seg in data["segmentations"]]
  
def load_clip_model(model_name: str, device, cache_dir: str = None):
    model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir
    ).to(device)

    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

# CROP PREPARATION
def _resize_and_pad(pil_img: Image.Image, target_size: int) -> Image.Image:
    """Resize image to fit within target_size, pad with black canvas."""
    img = pil_img.copy()
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas

def prepare_zero(image_np: np.ndarray, mask: np.ndarray, target_size: int = 336, **kwargs) -> Image.Image:
    """Isolate segment on black background."""
    isolated = image_np.copy()
    isolated[~mask] = 0
    y, x = np.where(mask)
    crop = isolated[y.min():y.max() + 1, x.min():x.max() + 1]
    return _resize_and_pad(Image.fromarray(crop), target_size)

def prepare_highlight(image_np: np.ndarray, mask: np.ndarray, target_size: int = 336,
                      highlight_color: tuple = (0, 255, 255), highlight_alpha: float = 0.35, **kwargs) -> Image.Image:
    """Full image with cyan highlight on target segment."""
    result = image_np.copy().astype(np.float32)
    for c, v in enumerate(highlight_color):
        result[:, :, c][mask] = result[:, :, c][mask] * (1 - highlight_alpha) + v * highlight_alpha
    result = result.clip(0, 255).astype(np.uint8)

    y, x = np.where(mask)
    cy, cx = (y.min() + y.max()) // 2, (x.min() + x.max()) // 2
    h, w = image_np.shape[:2]
    pad = max(y.max() - y.min(), x.max() - x.min()) * 3 // 2
    pad = max(pad, 64)

    y1, y2 = max(0, cy - pad), min(h, cy + pad)
    x1, x2 = max(0, cx - pad), min(w, cx + pad)
    return _resize_and_pad(Image.fromarray(result[y1:y2, x1:x2]), target_size)

def prepare_larger_crop(image_np: np.ndarray, mask: np.ndarray, target_size: int = 336,
                        context_factor: int = 3, **kwargs) -> Image.Image:
    """Crop context_factor × larger region with real surrounding pixels."""
    y, x = np.where(mask)
    cy, cx = (y.min() + y.max()) // 2, (x.min() + x.max()) // 2
    h, w = image_np.shape[:2]

    seg_h, seg_w = y.max() - y.min(), x.max() - x.min()
    pad_h = max(seg_h * context_factor // 2, 64)
    pad_w = max(seg_w * context_factor // 2, 64)

    y1, y2 = max(0, cy - pad_h), min(h, cy + pad_h)
    x1, x2 = max(0, cx - pad_w), min(w, cx + pad_w)
    return _resize_and_pad(Image.fromarray(image_np[y1:y2, x1:x2]), target_size)

def prepare_dual_composite(image_np: np.ndarray, mask: np.ndarray, target_size: int = 336,
                           highlight_color: tuple = (0, 255, 255), highlight_alpha: float = 0.35, **kwargs) -> Image.Image:
    """Side-by-side: full image with highlight + isolated segment."""
    half = target_size // 2

    highlighted = image_np.copy().astype(np.float32)
    for c, v in enumerate(highlight_color):
        highlighted[:, :, c][mask] = highlighted[:, :, c][mask] * (1 - highlight_alpha) + v * highlight_alpha
    highlighted = highlighted.clip(0, 255).astype(np.uint8)
    left_pil = Image.fromarray(highlighted).resize((half, target_size), Image.Resampling.LANCZOS)

    isolated = image_np.copy()
    isolated[~mask] = 0
    y, x = np.where(mask)
    crop = isolated[y.min():y.max() + 1, x.min():x.max() + 1]
    right_pil = Image.fromarray(crop)
    right_pil.thumbnail((half, target_size), Image.Resampling.LANCZOS)

    right_canvas = Image.new("RGB", (half, target_size), (0, 0, 0))
    offset = ((half - right_pil.width) // 2, (target_size - right_pil.height) // 2)
    right_canvas.paste(right_pil, offset)

    composite = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    composite.paste(left_pil, (0, 0))
    composite.paste(right_canvas, (half, 0))
    return composite

CROP_FUNCTIONS = {
    "zero": prepare_zero,
    "highlight": prepare_highlight,
    "larger_crop": prepare_larger_crop,
    "dual_composite": prepare_dual_composite,
}


# CLASSIFICATION
def classify_segments(image_np: np.ndarray, masks: List[np.ndarray], prompts_dict: Dict[str, List[str]],
                      clip_model: CLIPModel, clip_processor: CLIPProcessor, approach: str,
                      aggregation: str = "single", target_size: int = 336, batch_size: int = 16,
                      **crop_kwargs) -> Tuple[List[str], List[float], np.ndarray]:
    """Classify segments using specified crop approach and aggregation."""
    if approach not in CROP_FUNCTIONS:
        raise ValueError(f"unknown approach '{approach}'")
    if aggregation not in ("single", "average"):
        raise ValueError(f"unknown aggregation '{aggregation}'")

    crop_fn = CROP_FUNCTIONS[approach]

    all_prompts = []
    prompt_to_cls = {}
    class_slices = {}
    cursor = 0
    for cls_name, prompt_list in prompts_dict.items():
        for p in prompt_list:
            all_prompts.append(p)
            prompt_to_cls[p] = cls_name
        class_slices[cls_name] = (cursor, cursor + len(prompt_list))
        cursor += len(prompt_list)

    crops = []
    valid_idx = []
    for i, mask in enumerate(masks):
        y, x = np.where(mask)
        if len(y) == 0:
            continue
        crops.append(crop_fn(image_np, mask, target_size, **crop_kwargs))
        valid_idx.append(i)

    predictions = ["Unclassified"] * len(masks)
    confidences = [0.0] * len(masks)
    embeddings = np.zeros((len(masks), 768))

    if not crops:
        return predictions, confidences, embeddings

    device = clip_model.device

    for k in range(0, len(crops), batch_size):
        batch_crops = crops[k:k + batch_size]
        batch_idx = valid_idx[k:k + batch_size]

        inputs = clip_processor(text=all_prompts, images=batch_crops,
                               return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            image_embeds = outputs.image_embeds.cpu().numpy()

        if aggregation == "single":
            top_idx = probs.argmax(dim=1).cpu().numpy()
            top_probs = probs.max(dim=1).values.cpu().numpy()

            for local_i, true_i in enumerate(batch_idx):
                best_prompt = all_prompts[top_idx[local_i]]
                predictions[true_i] = prompt_to_cls[best_prompt]
                confidences[true_i] = float(top_probs[local_i])
                embeddings[true_i] = image_embeds[local_i]
        else:
            probs_np = probs.cpu().numpy()
            for local_i, true_i in enumerate(batch_idx):
                class_avgs = {cls_name: float(probs_np[local_i, start:end].mean())
                             for cls_name, (start, end) in class_slices.items()}
                best_cls = max(class_avgs, key=class_avgs.get)
                predictions[true_i] = best_cls
                confidences[true_i] = class_avgs[best_cls]
                embeddings[true_i] = image_embeds[local_i]

    return predictions, confidences, embeddings

# METRICS
def calculate_silhouette_metrics(embeddings: np.ndarray, predictions: List[str],
                                class_names: List[str]) -> Dict[str, float]:
    """Compute silhouette scores for CLIP embeddings."""
    metrics = {"silhouette_score": 0.0, "silhouette_score_vegetation": 0.0,
               "silhouette_score_building": 0.0, "silhouette_score_road": 0.0}

    valid_mask = np.array([p != "Unclassified" for p in predictions])
    if valid_mask.sum() < 2:
        return metrics

    valid_embeds = embeddings[valid_mask]
    valid_preds = np.array(predictions)[valid_mask]

    unique_classes = set(valid_preds)
    if len(unique_classes) < 2:
        return metrics

    try:
        metrics["silhouette_score"] = float(silhouette_score(valid_embeds, valid_preds, metric='cosine'))
        for cls in class_names:
            cls_mask = valid_preds == cls
            if cls_mask.sum() < 1:
                continue
            binary_labels = ["this_class" if p == cls else "other" for p in valid_preds]
            if len(set(binary_labels)) >= 2:
                cls_score = silhouette_score(valid_embeds, binary_labels, metric='cosine')
                metrics[f"silhouette_score_{cls.lower()}"] = float(cls_score)
    except Exception:
        pass

    return metrics

def calculate_confidence_metrics(confidences: List[float], predictions: List[str],
                                class_names: List[str], low_threshold: float = 0.3) -> Dict[str, float]:
    """Compute confidence-based metrics."""
    confs = np.array(confidences)
    preds = np.array(predictions)

    m = {
        "mean_confidence_overall": float(np.mean(confs)) if len(confs) else 0.0,
        "std_confidence_overall": float(np.std(confs)) if len(confs) else 0.0,
        "min_confidence": float(np.min(confs)) if len(confs) else 0.0,
        "max_confidence": float(np.max(confs)) if len(confs) else 0.0,
        "low_confidence_percentage": float(100 * np.sum(confs < low_threshold) / len(confs)) if len(confs) else 0.0,
    }

    for cls in class_names:
        cls_confs = confs[preds == cls]
        m[f"mean_confidence_{cls.lower()}"] = float(np.mean(cls_confs)) if len(cls_confs) else 0.0

    return m

def calculate_class_distribution(masks: List[np.ndarray], predictions: List[str],
                                class_names: List[str], image_shape: Tuple[int, int]) -> Dict[str, float]:
    """Compute class counts and pixel coverage."""
    total_pixels = image_shape[0] * image_shape[1]
    m = {"total_segments": len(masks)}

    for cls in class_names:
        count = sum(1 for p in predictions if p == cls)
        pixels = sum(int(np.sum(msk)) for msk, p in zip(masks, predictions) if p == cls)
        m[f"{cls.lower()}_count"] = count
        m[f"{cls.lower()}_coverage_percent"] = round(100 * pixels / total_pixels, 4) if total_pixels else 0.0

    return m

def calculate_spatial_consistency(masks: List[np.ndarray], predictions: List[str],
                                 neighbor_radius: int = 50) -> Dict[str, float]:
    """Compute spatial consistency metrics."""
    if not masks:
        return {"spatial_consistency_score": 0.0, "class_transition_count": 0, "isolation_ratio": 0.0}

    centroids = []
    for mask in masks:
        y, x = np.where(mask)
        centroids.append((float(np.mean(y)), float(np.mean(x))) if len(y) else (0.0, 0.0))
    centroids = np.array(centroids)
    distances = cdist(centroids, centroids)

    agreements = []
    transitions = 0
    isolated_count = 0

    for i in range(len(masks)):
        neighbours = np.where((distances[i] < neighbor_radius) & (distances[i] > 0))[0]
        if len(neighbours) == 0:
            isolated_count += 1
            continue
        same = sum(1 for j in neighbours if predictions[j] == predictions[i])
        agreements.append(same / len(neighbours))
        transitions += len(neighbours) - same

    return {
        "spatial_consistency_score": float(np.mean(agreements)) if agreements else 0.0,
        "class_transition_count": transitions,
        "isolation_ratio": round(100 * isolated_count / len(masks), 4),
    }

def calculate_all_metrics(masks: List[np.ndarray], predictions: List[str], confidences: List[float],
                         embeddings: np.ndarray, image_shape: Tuple[int, int], class_names: List[str],
                         low_conf_threshold: float = 0.3, neighbor_radius: int = 50) -> Dict[str, float]:
    """Compute all per-image metrics."""
    m = {}
    m.update(calculate_confidence_metrics(confidences, predictions, class_names, low_conf_threshold))
    m.update(calculate_class_distribution(masks, predictions, class_names, image_shape))
    m.update(calculate_spatial_consistency(masks, predictions, neighbor_radius))
    m.update(calculate_silhouette_metrics(embeddings, predictions, class_names))
    return m

# I/O
def save_classification_results(predictions: List[str], confidences: List[float],
                               embeddings: np.ndarray, output_path: str):
    """Save classification results to .npy file."""
    np.save(output_path,
            {"predictions": np.array(predictions), "confidences": np.array(confidences), "embeddings": embeddings},
            allow_pickle=True)

def load_classification_results(input_path: str) -> Tuple[List[str], List[float], np.ndarray]:
    """Load classification results from .npy file."""
    data = np.load(input_path, allow_pickle=True).item()
    return list(data["predictions"]), list(data["confidences"]), data.get("embeddings", np.array([]))

def save_per_image_metrics(metrics_list: List[Dict], output_path: str):
    """Save per-image metrics to CSV."""
    pd.DataFrame(metrics_list).to_csv(output_path, index=False)

def save_summary_metrics(metrics_list: List[Dict], output_path: str, total_time_seconds: float):
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
    pd.concat([summary, total_row], ignore_index=True).to_csv(output_path, index=False)
