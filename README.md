# Zero-Shot Segmentation and Classification of Urban Land Types Around Greenery Using Foundation Models with Spatial Buffer Analysis

An end-to-end **SAM + CLIP pipeline** for multi-scale urban tree analysis using aerial imagery and GIS-derived ground truth.

---

## 🎯 Overview

This project implements a complete pipeline for:

* 📍 Data preparation from QGIS vector layers
* 🧩 Multi-scale segmentation using SAM (Segment Anything Model)
* 🏷️ Context-aware classification using CLIP
* 📊 Pixel-level evaluation using ground truth masks
* 📐 Ring-based spatial analysis around trees
* 📈 Statistical significance testing of model variants

---

## 🧠 Pipeline Summary

```text id="z7k1qp"
📍 Data Preparation
   → QGIS image extraction (Aerial 2020/2025, Atlanta satellite)
   → Pixel-level ground truth generation (.npz)

        ↓

🧩 Segmentation (SAM)
   → Multi-scale tree segmentation (ViT-H)
   → Mask generation per tree region

        ↓

🏷️ Classification (CLIP)
   → Segment-level classification
   → Context strategies (zero, highlight, crop, composite)
   → Prompt engineering (v0–v1.3)

        ↓

📊 Evaluation
   → Pixel-level comparison with ground truth
   → Ring buffers (2.5m / 5m / 7.5m)
   → Metrics: MAE, IoU, F1, R², MBE
   → Statistical testing (Wilcoxon)
```

---

## 📦 Dataset

The datasets used in this project are provided externally.

### 🌍 Available Datasets for Erlangen City

* Aerial imagery (2020)
* Aerial imagery (2025)
* Google Satellite imagery 

📥 **Download dataset:**

> [Dataset GoogleDrive Link](https://drive.google.com/drive/folders/1iYzQ5BwWaJfY9Vw63S6ZcfNObQTcwXuD?usp=sharing)

---

### 📌 Dataset Contents

* Tree-centered image crops (300 DPI, 225 DPI, 150 DPI)
* Supporting PGW files for each DPI
* Pixel-level ground truth masks (QGIS rasterized)
* Vector annotations (trees, buildings, vegetation)

📖 For full dataset details and preprocessing steps, refer to the **project report**.

---

## 📁 Repository Structure

```text id="p4q8zn"
urban-tree-landcover-zeroshot/
├── data_preparation/
├── segmentation/
├── classification/
├── evaluation/
├── models/
├── dataset/
├── outputs/
├── requirements.txt
└── README.md
```

## 🧩 Key Modules

### 📍 Data Preparation

* QGIS-based image extraction
* Ground truth rasterization (.npz)
📖 For detailed workflow, refer to: 👉 [Data Preparation Documentation](data_preparation/README_datapreparation.md)

### 🧩 Segmentation (SAM)

* Multi-scale segmentation (ViT-H)
* Tree mask generation
* Experiment variants (exp01–exp06)
📖 For detailed experiments, configurations, and outputs, refer to: 👉 [Segmentation Documentation](segmentation/README_segmentation.md)

### 🏷️ Classification (CLIP)

* 4 context strategies:

  * Zero
  * Highlight
  * Larger crop
  * Dual composite 
* Prompt versions (0–1.3)
📖 For architecture, prompts, and evaluation setup, refer to: 👉 [Classification Documentation](classification/README_classification.md)

### 📊 Evaluation

* Pixel-level accuracy
* Ring-based spatial analysis
* Statistical significance testing
📖 For full metric definitions, formulas, and outputs, refer to: 👉 [Evaluation Documentation](evaluation/README_evaluation.md)
---

## 📊 Key Metrics

| Metric   | Description               |
| -------- | ------------------------- |
| MAE      | Mean Absolute Error       |
| IoU      | Intersection over Union   |
| F1-score | Balanced precision/recall |
| R²       | Area correlation          |
| MBE      | Systematic bias           |

---

## 📐 Ring-Based Evaluation

| Ring | Radius           |
| ---- | ---------------- |
| 2.5m | Core tree region |
| 5.0m | Mid context      |
| 7.5m | Extended context |

---

## 📈 Outputs

### Segmentation

* Tree masks (.npz)
* Confidence maps
* Visual overlays

### Classification

* Segment predictions
* Context comparisons
* Prompt-based results

### Evaluation

* Per-tree metrics (CSV)
* Final summary tables
* Statistical test results
* Visualization figures

---

## 🧪 Research Questions

* **RQ1:** How does multi-scale segmentation fusion influence spatial coverage and structural
consistency in aerial imagery compared to single-scale segmentation?
*  **RQ2:** How does aerial-specific prompt design influence land-use classification performance
in a zero-shot setting?
* **RQ3:** How does the choice of buffer radius affect the stability of land-use composition
estimates around urban trees, and how does class-wise reliability vary across scales?
* **RQ4:** How robust is the proposed zero-shot pipeline when applied to imagery from
different years, seasons, and sensors without parameter modification?

---

## ⚙️ Requirements

* Python 3.8+
* CUDA-enabled GPU (recommended)
* QGIS 3.16+ (for data extraction)

---

## 🙏 Acknowledgements

I would like to sincerely thank **Prof. Dr. Anne Koelewijn** for the opportunity to carry out this thesis at FAU Erlangen-Nürnberg.

Special thanks to my supervisors **Thomas Maier** and **Naga Venkata Sai Jitin Jami** for their continuous guidance and support throughout this work.

I am also grateful to **NHR@FAU** for providing computational resources.

Finally, I thank my family and friends for their constant encouragement.

---

## 📌 Notes

* Dataset is hosted externally (see link above)
* Full experimental details are available in the project report
* All modules can be run independently

---

## 📧 Contact

For questions regarding this project, please refer to the thesis documentation.

---

## 🎓 Summary

This project provides a **reproducible, modular, and research-grade pipeline** for:

* Urban tree segmentation
* Context-aware visual classification
* Pixel-level geospatial evaluation
* Statistical model comparison

---
## 📄 License

This project is released under the MIT License.

You are free to use, modify, and distribute this software for academic and research purposes with proper attribution.

See the [LICENSE](LICENSE) file for full details.
