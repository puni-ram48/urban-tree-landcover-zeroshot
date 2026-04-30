# Evaluation — Ring-Based Buffer Analysis

Evaluates CLIP classification predictions against pixel-level ground truth rasterized from QGIS vector shapefiles using concentric ring buffers (annuli) around tree centers.

---

## Quick Start

### 1. Setup
```bash
pip install numpy pandas matplotlib Pillow
```

### 2. Configure
Edit `config.py`:
```python
BASE_DIR = "/path/to/project/root"
IMAGES_DIR = os.path.join(BASE_DIR, "data/<dataset>/<images>")
PIXEL_GT_DIR = os.path.join(BASE_DIR, "data/<dataset>/pixel_groundtruth_<dpi>")
SEGMENTS_DIR = os.path.join(BASE_DIR, "<output>/segmentation/outputs/exp04_multiscale_finetuned/segments")
CLASSIFICATIONS_DIR = os.path.join(BASE_DIR, "<output>/classification/outputs/stage1_context/<approach>/classifications")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "<output>/evaluation/outputs/<experiment>")

EXPERIMENT_NAME = "<experiment_identifier>"
EVAL_SUFFIX = "_2"  # 225 DPI
```

### 3. Run Evaluation
```bash
python evaluate_rings.py
```

Output:
```
outputs/<experiment>/
├── per_tree_metrics_pixelgt.csv      ← per tree × ring × class
├── summary_pixelgt.csv               ← mean ± std across trees
├── final_summary_pixelgt.csv         ← aggregated by ring + class
└── visualizations/
    └── {devEUI}_ring_eval.png        ← 5-panel figure per tree
```

---

## Ring Definitions

Concentric annuli centered at tree (image center):

| Ring | Inner Radius | Outer Radius | Theoretical Area |
|------|-------------|-------------|------------------|
| **2.5m** | 0.0 m | 2.5 m | 19.635 m² |
| **5.0m** | 2.5 m | 5.0 m | 58.905 m² |
| **7.5m** | 5.0 m | 7.5 m | 98.175 m² |

---

## Methodology

**Ground Truth Source:**
- Pixel-level binary masks rasterized from QGIS vector shapefiles (.npz format)
- Keys: `vegetation`, `building`, `road` (uint8: 0 or 1 per pixel)
- Not derived from manual annotations or area CSV — true pixel coverage

**Prediction Pipeline:**
1. SAM segments (N × H × W boolean masks from `exp04_multiscale_finetuned`)
2. CLIP classification (class label per segment from `stage1_context`)
3. Combined into per-pixel class layer (H × W uint8: 0=unclassified, 1=Vegetation, 2=Building, 3=Road)

**Metrics per Ring per Class:**
- **MAE** — |predicted% − gt%| (percentage point error)
- **Precision** — TP / (TP + FP) (how many predicted pixels are correct?)
- **Recall** — TP / (TP + FN) (how many true pixels were found?)
- **F1-score** — Harmonic mean of precision & recall
- **IoU** — TP / (TP + FP + FN) (intersection over union)
- **R²** — Coefficient of determination (area estimation quality; 0–1 scale)
- **MBE** — Mean Bias Error (systematic over/underestimation in percentage points)

All metrics use true pixel overlap, NOT area approximations.

---

## Output Files

**per_tree_metrics_pixelgt.csv**
- One row per tree
- Columns: dev_eui, vegetation_2m5_mae, vegetation_2m5_precision, ... (per ring × class × metric)
- Use to: Identify trees where CLIP performs well/poorly

**summary_pixelgt.csv**
- Aggregated statistics across all trees
- Columns: metric, mean, std, min, max
- Use to: Quick overview of all metrics

**final_summary_pixelgt.csv**
- Aggregated by ring + class
- Columns: ring, class, mean_mae, std_mae, mean_precision, mean_recall, mean_f1, mean_iou, r2, mbe
- Use to: Compare approaches and classes

**visualizations/{devEUI}_ring_eval.png**
- 5-panel figure per tree:
  1. Original image + ring circles
  2. SAM segmentation overlay + rings
  3. CLIP prediction overlay + rings
  4. Pixel GT overlay + rings
  5. Bar chart (CLIP% vs GT% per ring per class)

---

## Results Interpretation

### Per-Ring Metrics

**MAE (Mean Absolute Error):**
- Lower is better (0% = perfect)
- Typical range: 5–25% depending on ring and class
- < 10%: Excellent | 10–15%: Good | 15–20%: Fair | > 20%: Poor

**Precision & Recall:**
- Precision: How many predicted pixels are correct?
- Recall: How many true pixels were found?
- F1: Harmonic mean (balances both; 0–1 scale)
- Typical range: 0.3–0.9
- > 0.75: Strong | 0.5–0.75: Moderate | < 0.5: Weak

**IoU (Intersection over Union):**
- Gold standard for segmentation (0–1 scale)
- Typical range: 0.2–0.7
- > 0.6: Excellent | 0.4–0.6: Good | < 0.4: Poor

### R² and MBE (Area Estimation Quality)

**R² — Correlation Quality:**
- Range: 0.0 to 1.0
- **> 0.7:** Strong correlation (CLIP reliably estimates areas)
- **0.4–0.7:** Moderate (useful but not precise)
- **0.2–0.4:** Weak (limited value)
- **< 0.2:** No correlation (CLIP% ≠ GT%)

**MBE — Systematic Bias:**
- Positive: CLIP overestimates by N pp
- Negative: CLIP underestimates by N pp
- Example: MBE = +5.2 pp → "CLIP predicts 5.2% more vegetation than GT on average"

**Example Interpretation:**
```
Ring 2.5m, Vegetation
  R² = 0.75, MBE = +3.1 pp
  → Strong correlation; CLIP slightly overestimates vegetation
  → Useful for area-based analysis; expect ~3% systematic overestimation

Ring 5.0m, Building
  R² = 0.15, MBE = -0.8 pp
  → Weak correlation; buildings rare in ring, CLIP struggles with class balance
  → Not reliable for this ring × class combination
```

---

## Configuration Reference

### Essential
```python
# Dataset paths
BASE_DIR = "/path/to/project/root"
IMAGES_DIR = "/path/to/images"
PIXEL_GT_DIR = "/path/to/pixel_groundtruth"
SEGMENTS_DIR = "/path/to/sam/segments"
CLASSIFICATIONS_DIR = "/path/to/clip/classifications"
OUTPUT_BASE_DIR = "/path/to/evaluation/outputs"

# Experiment identifier
EXPERIMENT_NAME = "<experiment_identifier>"

# Image scale
EVAL_SUFFIX = "_2"  # 225 DPI
EVAL_DPI = 225
```

### Ring Definitions
```python
RINGS = [
    {"suffix": "2m5", "inner_m": 0.0, "outer_m": 2.5, ...},
    {"suffix": "5m0", "inner_m": 2.5, "outer_m": 5.0, ...},
    {"suffix": "7m5", "inner_m": 5.0, "outer_m": 7.5, ...},
]
```

### Classes & Colors
```python
CLASS_NAMES = ["Vegetation", "Building", "Road"]

GT_NPZ_KEYS = {
    "Vegetation": "vegetation",  # key inside .npz file
    "Building": "building",
    "Road": "road",
}

# CLIP prediction overlay (matches classification config)
CLIP_CLASS_COLORS_RGBA = {
    "Vegetation": [0.059, 0.416, 0.196, 0.6],  # dark green
    "Building": [0.624, 0.184, 0.184, 0.6],    # dark red
    "Road": [0.365, 0.349, 0.349, 0.6],        # dark gray
}

# Ground truth overlay
GT_CLASS_COLORS_RGBA = {
    "Vegetation": [0.0, 0.502, 0.0, 0.6],      # dark green
    "Building": [0.502, 0.0, 0.0, 0.6],        # dark red
    "Road": [0.365, 0.349, 0.349, 0.6],        # dark gray
}
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Images not found | Check `IMAGES_DIR` in config.py |
| Segments not found | Ensure exp04 completed, check `SEGMENTS_DIR` |
| Classifications missing | Run `exp01_clip_classification.py` first |
| Pixel GT missing | Ensure `.npz` files in `PIXEL_GT_DIR` with correct keys |
| Memory error on large datasets | Process trees in batches; reduce to subset first |
| Ring circles don't align | Check `METERS_PER_PIXEL` formula matches GT generation |

---

## Tips

**Inspect one tree's results:**
```bash
# View visualization
open outputs/<experiment>/visualizations/{devEUI}_ring_eval.png

# Check metrics for that tree
grep {devEUI} outputs/<experiment>/per_tree_metrics_pixelgt.csv
```

**Compare experiments:**
```bash
# Run with different approaches/prompts
# Edit config.py: EXPERIMENT_NAME = "...prompt1.2..."
python evaluate_rings.py

# Compare final summaries
diff final_summary_v1.csv final_summary_v2.csv
```

**Identify best-performing approaches:**
```bash
# Sort by mean F1 score
tail -n +2 final_summary_pixelgt.csv | sort -t, -k6 -rn | head -3
```

**Export for publication:**
```bash
# Final summary has all key metrics with R² and MBE
cat outputs/<experiment>/final_summary_pixelgt.csv | column -t -s,
```

---

## Files

| File | Purpose |
|------|---------|
| **evaluate_rings.py** | Main evaluation script (run this) |
| **config.py** | Configuration & ring definitions |

---

## For More Details

- **Ground truth generation** — See `pixel_gt_generator.py` (companion script in data preparation)
- **Ring geometry** — Inner/outer radius converted to pixels using DPI and QGIS export scale
- **Metrics definitions** — Standard computer vision (precision, recall, F1, IoU, R², MBE)
- **SAM segments source** — From `exp04_multiscale_finetuned`
- **CLIP predictions source** — From `stage1_context` with selected approach and prompt

---

## Workflow

### Basic Evaluation
```bash
# Configure config.py with your paths
python evaluate_rings.py

# Review results
cat outputs/<experiment>/final_summary_pixelgt.csv
open outputs/<experiment>/visualizations/sample_tree_ring_eval.png
```

### Compare Approaches
```bash
# Run with approach 1
# Edit config.py: CLASSIFICATIONS_DIR = "...zero/..."
# Edit config.py: EXPERIMENT_NAME = "...zero_prompt1.3..."
python evaluate_rings.py

# Run with approach 2
# Edit config.py: CLASSIFICATIONS_DIR = "...dual_composite/..."
# Edit config.py: EXPERIMENT_NAME = "...dual_composite_prompt1.3..."
python evaluate_rings.py

# Compare final summaries
diff outputs/.../zero_prompt1.3/final_summary_pixelgt.csv outputs/.../dual_composite_prompt1.3/final_summary_pixelgt.csv
```

### Compare Prompt Versions
```bash
# Stage 1: Run classification with different prompts
# config.py: PROMPT_VERSION = "1.2"
python exp01_clip_classification.py --approach dual_composite

# Stage 1: Run classification with prompt 1.3
# config.py: PROMPT_VERSION = "1.3"
python exp01_clip_classification.py --approach dual_composite

# Evaluation: Compare both
# edit config.py for each run, change EXPERIMENT_NAME
python evaluate_rings.py  # run twice with different CLASSIFICATIONS_DIR
```

---

## Next Steps

After evaluation:
1. Review final summary: `final_summary_pixelgt.csv`
2. Check visualizations: `visualizations/`
3. Identify rings and classes where CLIP performs well vs. poorly
4. Compare R² and MBE across approaches/prompts
5. Use insights for:
   - Final recommendations on best approach/prompt
   - Ablation study analysis
   - Cross-dataset validation
   - Publication of results

---

## Reproducibility

To reproduce results:
1. Ensure pixel GT .npz files match image dimensions (1024×1024)
2. Verify GT .npz keys match `GT_NPZ_KEYS` in config
3. Set `EVAL_SUFFIX = "_2"` (225 DPI reference)
4. Use best-performing approach from Stage 1:
   ```bash
   EXPERIMENT_NAME = "aerial2020_dual_composite_prompt1.3_single"
   CLASSIFICATIONS_DIR = ".../stage1_context/dual_composite/classifications"
   python evaluate_rings.py
   ```
5. Review `final_summary_pixelgt.csv` for summary metrics with R² and MBE

---

## Citation

If using this evaluation framework, cite:
- Ring-based buffer methodology for tree-centric spatial analysis
- QGIS rasterization from vector shapefiles for ground truth
- Standard metrics: precision, recall, F1, IoU, R², MBE
```
- ✓ Workflow examples (basic, compare approaches, compare prompts)
- ✓ Next steps
- ✓ Reproducibility instructions
