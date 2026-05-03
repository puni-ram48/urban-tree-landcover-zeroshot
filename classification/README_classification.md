# 🧠 CLIP Classification Pipeline

This pipeline performs **semantic classification of SAM-generated segments** using **CLIP (Contrastive Language–Image Pretraining)** with multiple context-aware strategies and prompt engineering variants.

It is designed to evaluate how **visual context + prompt design** influence classification performance in aerial imagery.

---

## 📌 Overview

The system classifies each SAM segment into:

* Vegetation
* Building
* Road

It supports multiple **context-handling strategies** and **prompt versions** for ablation studies.

---

## ⚙️ Installation

```bash id="j5c7l2"
pip install -r requirements.txt --break-system-packages
```

---

## 🤖 CLIP Model Setup

The CLIP model is automatically downloaded on first run.

To ensure reproducibility, the model is stored locally:

```
models/clip_model/
```

Default model:

```
clip-vit-large-patch14-336
```

---

## ⚙️ Configuration

Edit `config.py`:

```python id="3p9kqv"
BASE_DIR = "/path/to/project/root"

IMAGES_DIR = BASE_DIR + "/dataset/images"
SEGMENTS_DIR = BASE_DIR + "/segmentation/outputs/exp04/segments"

CLIP_MODEL_PATH = BASE_DIR + "/models/clip_model/clip-vit-large-patch14-336/"
OUTPUT_BASE_DIR = BASE_DIR + "/classification/outputs"

PROMPT_VERSION = "1.3"  # 0, 1, 1.1, 1.2, 1.3
DEVICE = "cuda"
```

---

## 🧪 Quick Test (Recommended)

```bash id="v9q1n3"
python classification/clip_classification.py --approach zero --test
python classification/visualize_samples.py --approach zero
```

---

## 🚀 Running Experiments

### 🔹 Core Approaches

| Approach       | Description                   |
| -------------- | ----------------------------- |
| zero           | Isolated segment (no context) |
| highlight      | Full image + cyan highlight   |
| larger_crop    | Enlarged local crop (3× bbox) |
| dual_composite | Global + local side-by-side ⭐ |

```bash id="k2m8xw"
python classification/clip_classification.py --approach zero
python classification/clip_classification.py --approach highlight
python classification/clip_classification.py --approach larger_crop
python classification/clip_classification.py --approach dual_composite
```

---

### 🔹 Aggregation Mode (Optional)

```bash id="q7n4zd"
python classification/clip_classification.py --approach zero --aggregation average
```

Modes:

* `average` → robust mean prediction across segments

---

## 🧾 Prompt Engineering

Controlled via `PROMPT_VERSION` in `config.py`.

| Version | Description                 | Usage       |
| ------- | --------------------------- | ----------- |
| 0       | Minimal prompts             | Baseline    |
| 1       | Basic aerial text           | Standard    |
| 1.1     | Material-aware prompts      | Buildings   |
| 1.2     | Vegetation-enhanced prompts | Green areas |
| 1.3     | Balanced prompts ⭐          | Recommended |

---

## 📊 Output Structure

```id="x8c2ba"
outputs/
├── zero_single/
│   ├── classifications/
│   ├── metrics/
│   │   ├── per_image_metrics.csv
│   │   ├── summary_metrics.csv
│   ├── logs/
│
├── highlight_single/
├── larger_crop_single/
├── dual_composite_single/
│
├── *_average/                     # aggregation mode results
│
├── visualizations/
│   ├── zero_single/
│   ├── highlight_single/
│   ├── larger_crop_single/
│   ├── dual_composite_single/
│   ├── comparison/
│   │   └── {devEUI}_comparison.png
│
└── clip_aggregate_summary.csv
```

---

## 📈 Key Metrics

### Model Performance

* **mean_confidence_overall** → classification confidence (0–1)
* **silhouette_score** → embedding separation quality
* **spatial_consistency_score** → spatial smoothness of predictions

### Reliability Indicators

* **low_confidence_percentage** → uncertainty ratio
* **pixel_coverage_percent** → class distribution
* **processing_time_seconds** → runtime efficiency

---

## 🧠 Interpretation Guide

| Metric              | Good Range | Meaning                    |
| ------------------- | ---------- | -------------------------- |
| Confidence          | > 0.50     | Reliable predictions       |
| Silhouette Score    | > 0.30     | Good class separation      |
| Spatial Consistency | > 0.50     | Stable spatial predictions |
| Low Confidence %    | < 20%      | Few uncertain segments     |

---

## 📊 Visualization

### Single Approach

```bash id="p4q8tz"
python classification/visualize_samples.py --approach zero
```

### Cross-Approach Comparison

```bash id="l7v1dc"
python classification/visualize_samples.py --compare --max_images 10
```

Outputs:

* Original image
* Class overlay
* Heatmap
* Side-by-side method comparison

---

## 📁 Key Scripts

| File                     | Purpose                    |
| ------------------------ | -------------------------- |
| `clip_classification.py` | Main pipeline              |
| `visualize_samples.py`   | Visualization & comparison |
| `config.py`              | Experiment configuration   |
| `utils.py`               | CLIP + metrics utilities   |

---

## ⚙️ Context Strategies

### Zero (Baseline)

* No spatial context
* Each segment classified independently

### Highlight

* Full image context
* Segment highlighted in cyan overlay

### Larger Crop

* Enlarged bounding box (3×)
* Preserves local surroundings

### Dual Composite ⭐

* Global + local view side-by-side
* Best overall performance

---

## ⚠️ Common Issues

| Issue              | Fix                                   |
| ------------------ | ------------------------------------- |
| Missing segments   | Run SAM pipeline first                |
| CLIP model missing | Ensure HuggingFace download completes |
| CUDA OOM           | Reduce batch size in config           |
| Slow inference     | Verify GPU via `nvidia-smi`           |

---

## 🔧 Debugging & Monitoring

```bash id="u3n9kq"
# Resume run (auto-skips processed images)
python clip_classification.py --approach zero

# Monitor GPU
watch -n 1 nvidia-smi

# Logs
tail -f outputs/zero_single/logs/processing.log
```

---

## 📌 Research Context

This pipeline evaluates:

* Impact of **visual context (zero vs multi-context)**
* Effect of **prompt engineering (1.0–1.3 variants)**
* Robustness of CLIP on **aerial segmentation tasks**
* Integration with **SAM-based region proposals**

---

## 🔄 Workflow

```bash id="w2k9pz"
# 1. Run classification
python classification/clip_classification.py --approach dual_composite

# 2. Visual inspection
python classification/visualize_samples.py --compare

# 3. Analyze metrics
cat outputs/clip_aggregate_summary.csv
```

---

## 🎯 Purpose

This module enables:

* Systematic evaluation of CLIP for remote sensing
* Controlled ablation of context strategies
* Prompt engineering analysis
* Integration with SAM segmentation + ring evaluation pipeline

---
