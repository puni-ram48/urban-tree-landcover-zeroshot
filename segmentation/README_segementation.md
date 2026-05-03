# SAM Segmentation Pipeline

Automated tree segmentation using Segment Anything Model (SAM) with single-scale and multi-scale variants.

---

## Quick Start

### 0. Model Selection (E0) — Optional
If you want to reproduce the model capacity comparison (ViT-B vs ViT-L vs ViT-H):
```bash
python exp0_model_selection.py  
```
Output: `outputs/sam_model_selection/model_selection_summary.csv`
Recommended: **ViT-H** (best quality, 636M params)

---
### 1. Setup
```bash
pip install torch segment-anything Pillow numpy pandas matplotlib scikit-image tqdm

# Download SAM model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h.pth
```

### 2. Configure
Edit `config.py`:
```python
BASE_DIR = "/path/to/project"
IMAGES_DIR = os.path.join(BASE_DIR, "data/images")
SAM_CHECKPOINT = os.path.join(BASE_DIR, "models/sam_vit_h.pth")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "outputs")
CITY_NAME = "erlangen"
N_TREES = 96
```

### 3. Test (Optional)
```bash
python test_single_pipeline.py      # Validate single-scale on 3 images
python test_multiscale_pipeline.py  # Validate multi-scale on 3 groups
```

### 4. Run Experiments

**Model Selection (optional baseline):**
```bash
python exp0_model_selection.py  
```

**Single-scale (fast, baseline):**
```bash
python exp01_single_default.py      
python exp02_single_finetuned.py    
```

**Multi-scale (better quality):**
```bash
python exp03_multiscale_default.py      
python exp04_multiscale_finetuned.py    
python exp05_multiscale_225_150_dpi.py  
python exp06_multiscale_300_225_dpi.py 
```

### 5. Visualize Results
```bash
# Single experiment
python visualize_samples.py --experiment exp04_multiscale_finetuned --num_samples 10

# All experiments comparison
python visualize_samples.py --all_experiments --comparisons --num_samples 5
```

---

## Experiments Overview

| Exp | Name | Type | Scales | Best For |
|-----|------|------|--------|----------|
| **01** | Single-Scale Default | Single | 225 DPI | Baseline |
| **02** | Single-Scale Fine-tuned | Single | 225 DPI | Better single |
| 03 | Multi-Scale Default | Multi | 300+225+150 | Comparison |
| **04** | Multi-Scale Fine-tuned | Multi | 300+225+150 | ⭐ **RECOMMENDED** |
| 05 | Two-Scale (225+150) | Multi | 225+150 | Avoid edge artifacts |
| 06 | Two-Scale (300+225) | Multi | 300+225 | Avoid noise |

---

## Output Structure

```
outputs/
├── sam_model_selection/                 ← E0 results
│   ├── model_selection_per_image.csv
│   └── model_selection_summary.csv
├── exp01_single_default/
├── exp02_single_finetuned/
├── exp04_multiscale_finetuned/          ← Best results here
│   ├── segments/
│   │   ├── tree_001_2_segments.npz
│   │   └── ... (one per tree)
│   ├── metrics/
│   │   ├── per_image_metrics.csv
│   │   └── summary_metrics.csv
│   ├── visualizations/
│   │   └── viz_*.png
│   └── logs/
│       └── processing.log
├── exp05_multiscale_225_150_dpi/
├── exp06_multiscale_300_225_dpi/
└── comparisons/                         ← Cross-experiment grids
    └── comparison_*.png
```

---

## Results Interpretation

**Key Metrics (from CSV files):**

- **segment_count** — Number of tree parts found (150-250 normal)
- **mean_confidence** — Quality score 0-1 (>0.85 is good)
- **mean_stability** — Reliability 0-1 (>0.90 is good)
- **pixel_coverage_percent** — Tree coverage 40-70% is normal
- **processing_time_seconds** — Computation time per image/group

---

## Files

| File | Purpose |
|------|---------|
| **exp00_model_selection.py** | SAM variant comparison (E0) |
| **exp01-exp06** | Experiment scripts (run these) |
| **test_single_pipeline.py** | Quick validation (3 images) |
| **test_multiscale_pipeline.py** | Quick validation (3 groups) |
| **visualize_samples.py** | Generate visualizations |
| **config.py** | Configuration & parameters |
| **utils.py** | Shared utilities |
| **sam_device_patch.py** | SAM device compatibility |


---

## Configuration Reference

### Essential Settings
```python
# Paths (edit in config.py)
IMAGES_DIR = "/path/to/images"
SAM_CHECKPOINT = "/path/to/sam_vit_h.pth"
OUTPUT_BASE_DIR = "/path/to/outputs"
DEVICE = "cuda"  # or "cpu"
```

### SAM Parameters (exp04 recommended)
```python
EXP4_PARAMS = {
    "points_per_side": 64,              # Grid density
    "pred_iou_thresh": 0.80,            # Quality threshold
    "stability_score_thresh": 0.85,     # Stability threshold
    "min_mask_region_area": 100,        # Remove noise masks
}
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Images not found | Check `IMAGES_DIR` path in config.py |
| SAM checkpoint missing | Download from FB and update `SAM_CHECKPOINT` |
| CUDA out of memory | Reduce `points_per_side` from 64 to 32 |
| PGW file missing | Multi-scale needs `.pgw` world files |
| Slow processing | Ensure GPU available: `nvidia-smi` |

---

## Tips

**Resume interrupted run:**
```bash
python exp04_multiscale_finetuned.py
# Automatically skips already-processed trees
```

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

**Check logs:**
```bash
tail -f outputs/exp04_multiscale_finetuned/logs/processing.log
```

---

## For More Details

- **Architecture & theory** — See research paper
- **All configuration options** — See `config.py` comments
- **Utility functions** — See `utils.py` docstrings

---

## Next Steps

After segmentation:
1. Review metrics: `outputs/exp04_multiscale_finetuned/metrics/summary_metrics.csv`
2. View visualizations: `python visualize_samples.py --all_experiments --comparisons`
3. Use segments for classification or evaluation
