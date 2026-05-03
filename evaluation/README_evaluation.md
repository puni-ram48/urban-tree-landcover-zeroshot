# 🌳 Ring-Based Buffer Evaluation (CLIP vs Pixel-Level Ground Truth)

This repository evaluates **CLIP-based land cover classification** against **pixel-level ground truth derived from QGIS vector data**, using concentric **ring buffers (annuli)** around tree centers.

It supports:

* Multi-ring spatial evaluation (2.5m / 5m / 7.5m)
* Pixel-accurate performance metrics
* Statistical comparison of experiments
* Per-tree visual diagnostics

---

## 📌 Overview

The pipeline compares:

* **Predictions:** SAM segments + CLIP classification
* **Ground Truth:** Pixel-level rasterized masks (`.npz` from QGIS vectors)

### Classes

* Vegetation
* Building
* Road

---

## 📂 Project Structure

```
evaluation/
├── evaluate_rings.py              # Main evaluation pipeline
├── statistical_tests.py           # Wilcoxon statistical testing
├── config.py                      # Experiment configuration
└── outputs/
    └── <experiment>/
        ├── per_tree_metrics_pixelgt.csv
        ├── summary_pixelgt.csv
        ├── final_summary_pixelgt.csv
        ├── statistical_tests_results.csv
        └── visualizations/
            └── {tree_id}_ring_eval.png
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt --break-system-packages
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
BASE_DIR = "/path/to/project/root"

IMAGES_DIR = BASE_DIR + "/dataset/images"
GROUND_TRUTH = BASE_DIR + "/dataset/pixel_gt.npz"

SEGMENTS_DIR = BASE_DIR + "/segmentation/outputs/exp04/segments"
CLASSIFICATIONS_DIR = BASE_DIR + "/classification/outputs/clip/classifications"

OUTPUT_BASE_DIR = BASE_DIR + "/evaluation/outputs"

EXPERIMENT_NAME = "exp_name"
EVAL_SUFFIX = "_2"  # 225 DPI reference
```

---

## 🔵 Ring Definition

Evaluation is performed using concentric buffers around tree centers:

| Ring | Inner Radius | Outer Radius | Area (m²) |
| ---- | ------------ | ------------ | --------- |
| 2.5m | 0.0 m        | 2.5 m        | 19.64     |
| 5.0m | 2.5 m        | 5.0 m        | 58.91     |
| 7.5m | 5.0 m        | 7.5 m        | 98.18     |

---

## 🚀 Run Evaluation

```bash
python evaluation/evaluate_rings.py
```

### Outputs

| File                           | Description                         |
| ------------------------------ | ----------------------------------- |
| `per_tree_metrics_pixelgt.csv` | Metrics per tree × ring × class     |
| `summary_pixelgt.csv`          | Global mean/std statistics          |
| `final_summary_pixelgt.csv`    | Aggregated ring + class performance |
| `visualizations/`              | Per-tree diagnostic plots           |

---

## 📊 Metrics

All metrics are computed using **true pixel overlap**.

### Segmentation & Classification Metrics

* **MAE** – Mean absolute percentage error
* **Precision** – Correctness of predictions
* **Recall** – Coverage of ground truth
* **F1-score** – Balanced precision/recall
* **IoU** – Intersection over Union

### Area Estimation Metrics

* **R²** – Correlation between predicted vs ground truth area
* **MBE** – Mean bias error (systematic over/underestimation)

---

## 📈 Interpretation Guide

### R² (Area Reliability)

* > 0.7 → Strong correlation
* 0.4–0.7 → Moderate
* < 0.4 → Weak

### MBE (Bias)

* +ve → Overestimation
* -ve → Underestimation

### Example

```
Ring 2.5m – Vegetation
R² = 0.75 → strong reliability  
MBE = +3.1% → slight overestimation
```

---

## 🧪 Statistical Testing

Run Wilcoxon signed-rank tests to compare experiments:

```bash
python evaluation/statistical_tests.py
```

### Output

```
statistical_tests_results.csv
```

| RQ  | Comparison | Metric | Mean A | Mean B | Diff | p-value | Significant |
| --- | ---------- | ------ | ------ | ------ | ---- | ------- | ----------- |
| RQ1 | E1 vs E4   | F1     | 0.42   | 0.58   | 0.16 | 0.0001  | YES         |

---

## 🔬 Research Questions

* **RQ1:** Does multi-scale fusion improve performance?
* **RQ2:** Do prompt variations improve classification?
* **RQ3:** Does cropping strategy affect accuracy?
* **RQ4:** Does the pipeline generalize across datasets?

---

## 📊 Visual Outputs

Each tree generates a 5-panel visualization:

1. Original image + rings
2. SAM segmentation + rings
3. CLIP predictions + rings
4. Ground truth overlay
5. Ring-wise bar chart comparison

---

## ⚠️ Common Issues

| Problem          | Fix                             |
| ---------------- | ------------------------------- |
| Missing images   | Check `IMAGES_DIR`              |
| Missing segments | Run SAM pipeline first          |
| Missing GT       | Ensure `.npz` keys match config |
| Misaligned rings | Verify pixel-to-meter scaling   |
| Memory issues    | Process subset of trees         |

---

## 📌 Workflow Summary

```bash
# 1. Run evaluation
python evaluation/evaluate_rings.py

# 2. Run statistical tests
python evaluation/statistical_tests.py

# 3. Inspect results
open outputs/<experiment>/final_summary_pixelgt.csv
```

---

## 📦 Key Outputs for Analysis

* **final_summary_pixelgt.csv** → Main results table
* **statistical_tests_results.csv** → Significance testing
* **visualizations/** → Qualitative validation

---

## 🎯 Purpose

This framework enables:

* Fine-grained spatial evaluation of vision models
* Ring-based ecological / urban analysis
* Robust statistical comparison of pipelines
* Publication-ready quantitative + qualitative results

---
