# 🧠 SAM Segmentation Pipeline

Automated tree segmentation using the **Segment Anything Model (SAM)** with single-scale and multi-scale configurations.

---

## 📌 Overview

This pipeline generates high-quality tree segmentation masks using SAM and supports:

* Single-scale segmentation (baseline)
* Multi-scale segmentation (recommended)
* Fine-tuned and default variants
* GPU-accelerated batch processing
* Experiment-level comparison and visualization

---

## ⚙️ Installation

```bash
pip install -r requirements.txt --break-system-packages
```

---

## 📦 Model Setup

Download SAM checkpoints:

```bash
mkdir -p models/sam_model

# ViT-H (Best quality)
wget -O models/sam_model/sam_vit_h.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ViT-L (Balanced)
wget -O models/sam_model/sam_vit_l.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-B (Fastest)
wget -O models/sam_model/sam_vit_b.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
BASE_DIR = "/path/to/project/root"

IMAGES_DIR = BASE_DIR + "/dataset/images"
GROUND_TRUTH = BASE_DIR + "/dataset/ground_truth"

SAM_CHECKPOINT = BASE_DIR + "/models/sam_model/sam_vit_h.pth"
SAM_MODEL_TYPE = "vit_h"

OUTPUT_BASE_DIR = BASE_DIR + "/segmentation/outputs"

CITY_NAME = "erlangen"
N_TREES = 96
```

---

## 🧪 Quick Validation (Recommended)

Run lightweight tests before full execution:

```bash
python segmentation/test_singlescale_pipeline.py
python segmentation/test_multiscale_pipeline.py
```

---

## 🚀 Running Experiments

### 🔹 Model Selection (Baseline Comparison)

```bash
python segmentation/exp00_model_selection.py
```

---

### 🔹 Single-Scale Experiments

```bash
python segmentation/exp01_single_default.py
python segmentation/exp02_single_finetuned.py
```

---

### 🔹 Multi-Scale Experiments (Recommended)

```bash
python segmentation/exp03_multiscale_default.py
python segmentation/exp04_multiscale_finetuned.py   # ⭐ Best performance
python segmentation/exp05_multiscale_225_150_dpi.py
python segmentation/exp06_multiscale_300_225_dpi.py
```

---

## ⭐ Recommended Configuration

* **Best overall model:** `exp04_multiscale_finetuned`
* **Best trade-off:** multi-scale (225 + 150 DPI)
* **Fast baseline:** `exp01_single_default`

---

## 📊 Visualization

### Single Experiment

```bash
python segmentation/visualize_samples.py \
  --experiment exp04_multiscale_finetuned \
  --num_samples 3
```

### Cross-Experiment Comparison

```bash
python segmentation/visualize_samples.py \
  --all_experiments --comparisons --num_samples 3
```

---

## 📁 Output Structure

```
outputs/
├── exp01_single_default/
├── exp02_single_finetuned/
├── exp04_multiscale_finetuned/
│   ├── segments/            # Mask outputs (.npz)
│   ├── metrics/             # Quantitative evaluation
│   │   ├── per_image_metrics.csv
│   │   └── summary_metrics.csv
│   ├── visualizations/      # Qualitative results
│   └── logs/                # Execution logs
├── exp05_multiscale_225_150_dpi/
├── exp06_multiscale_300_225_dpi/
└── comparisons/             # Cross-experiment figures
```

---

## 📈 Key Metrics

| Metric                    | Meaning                  |
| ------------------------- | ------------------------ |
| `segment_count`           | Number of masks per tree |
| `mean_confidence`         | Prediction quality (0–1) |
| `mean_stability`          | Mask consistency (0–1)   |
| `pixel_coverage_percent`  | Tree coverage ratio      |
| `processing_time_seconds` | Runtime per image        |

**Good performance indicators:**

* Confidence > 0.85
* Stability > 0.90
* Coverage: 40–70%

---

## 🧠 Key Experiments Summary

| Exp | Type                    | Description         | Recommendation |
| --- | ----------------------- | ------------------- | -------------- |
| 01  | Single-scale            | Baseline            | Reference only |
| 02  | Single-scale fine-tuned | Improved baseline   | OK             |
| 03  | Multi-scale default     | Comparison          | Intermediate   |
| 04  | Multi-scale fine-tuned  | Best quality        | ⭐ Recommended  |
| 05  | 225 + 150 DPI           | Noise reduction     | Useful         |
| 06  | 300 + 225 DPI           | Detail preservation | Useful         |

---

## ⚙️ SAM Parameters (Exp04)

```python
{
    "points_per_side": 64,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.85,
    "min_mask_region_area": 100
}
```

---

## ⚠️ Common Issues

| Issue              | Fix                                |
| ------------------ | ---------------------------------- |
| Missing images     | Verify `IMAGES_DIR`                |
| Missing checkpoint | Download SAM weights               |
| CUDA OOM           | Reduce `points_per_side`           |
| Slow runtime       | Check GPU via `nvidia-smi`         |
| Missing PGW files  | Required for multi-scale alignment |

---

## 🔧 Utilities

```bash
# Resume interrupted run
python segmentation/exp04_multiscale_finetuned.py

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f outputs/exp04_multiscale_finetuned/logs/processing.log
```

---

## 📌 Workflow

```bash
# 1. Run segmentation
python segmentation/exp04_multiscale_finetuned.py

# 2. Check outputs
python segmentation/visualize_samples.py --all_experiments

# 3. Use segments in classification pipeline
```

---

## 🎯 Purpose

This pipeline provides:

* High-quality tree segmentation using SAM
* Multi-scale spatial robustness
* Reproducible experimental design
* Direct integration with downstream CLIP classification and evaluation

---
