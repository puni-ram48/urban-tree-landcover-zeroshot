# Evaluation : Ring-Based Buffer Analysis

Evaluates CLIP classification predictions against pixel-level ground truth rasterized from QGIS vector shapefiles using concentric ring buffers (annuli) around tree centers.

---

## Quick Start

### 1. Setup
```bash
## already installed requirements.txt in segmentation phase
pip install -r requirements.txt --break-system-packages
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
python evaluation/evaluate_rings.py
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

### 4. Run Statistical Tests
```bash
python statistical_tests.py
```

Output:
```
outputs/statistical_tests_results.csv
  ├── RQ, Comparison, Metric, N, Mean A, Mean B, Difference, p-value, Significant
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

**statistical_tests_results.csv**
- Statistical significance test results (Wilcoxon signed-rank)
- Columns: RQ, Comparison, Metric, N, Mean A, Mean B, Difference, Statistic, p-value, Significant
- Use to: Determine if differences are statistically significant (p < 0.05)

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

### Statistical Significance

**Wilcoxon Signed-Rank Test:**
- Non-parametric paired test (does not assume normal distribution)
- Tests if two matched samples differ significantly
- Significance level: α = 0.05
- Result: YES (p < 0.05) = statistically significant | NO (p ≥ 0.05) = not significant

**Research Questions Addressed:**
- **RQ1:** Does multi-scale fusion improve coverage? (E1 vs E4)
- **RQ2:** Does aerial-specific prompt improve classification? (Prompt 0 vs 1.3)
- **RQ3:** Does crop method affect accuracy? (Zero vs Dual Composite)
- **RQ4:** Does pipeline generalize across datasets? (Aerial 2020 vs Google Satellite)

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

### Statistical Tests Configuration
```python
# statistical_tests.py paths (edit to point to your evaluation outputs)
PATH_E1 = "<output>/segmentation/outputs/exp01_single_default/metrics/per_image_metrics.csv"
PATH_E4 = "<output>/segmentation/outputs/exp04_multiscale_finetuned/metrics/per_image_metrics.csv"
PATH_PROMPT0 = "<output>/evaluation/outputs/aerial2020_zero_prompt0_results/per_tree_metrics_pixelgt.csv"
PATH_PROMPT13 = "<output>/evaluation/outputs/aerial2020_zero_prompt1.3_results/per_tree_metrics_pixelgt.csv"
# ... (other paths for crop methods and cross-dataset)

ALPHA = 0.05  # significance level
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
| Statistical tests fail | Update paths in `statistical_tests.py` to point to your evaluation outputs |

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
python evaluate_rings.py  # run multiple times with different config.py

# Compare final summaries
diff final_summary_v1.csv final_summary_v2.csv
```

**Identify best-performing approaches:**
```bash
# Sort by mean F1 score
tail -n +2 final_summary_pixelgt.csv | sort -t, -k6 -rn | head -3
```

**Find failure modes (low-performing trees):**
```python
import pandas as pd
df = pd.read_csv("outputs/<experiment>/per_tree_metrics_pixelgt.csv")
# Find trees with very low vegetation F1 in Ring 1
low_veg = df[df["vegetation_2m5_f1"] < 0.3]
print(f"Trees with Vegetation F1 < 0.3: {len(low_veg)}")
print(low_veg[["dev_eui", "vegetation_2m5_f1", "vegetation_2m5_pred_pct"]])
```

**Export for publication:**
```bash
# Final summary has all key metrics with R² and MBE
cat outputs/<experiment>/final_summary_pixelgt.csv | column -t -s,

# Statistical test results
cat outputs/statistical_tests_results.csv | column -t -s,
```

---

## Files

| File | Purpose |
|------|---------|
| **evaluate_rings.py** | Main evaluation script (per-tree metrics & visualizations) |
| **statistical_tests.py** | Statistical significance testing (Wilcoxon tests for RQ1–RQ4) |
| **config.py** | Configuration & ring definitions |

---

## For More Details

- **Ground truth generation** — See `pixel_gt_generator.py` (companion script in data preparation)
- **Ring geometry** — Inner/outer radius converted to pixels using DPI and QGIS export scale
- **Metrics definitions** — Standard computer vision (precision, recall, F1, IoU, R², MBE)
- **SAM segments source** — From `exp04_multiscale_finetuned`
- **CLIP predictions source** — From `stage1_context` with selected approach and prompt
- **Statistical methodology** — Wilcoxon signed-rank test (non-parametric, paired, two-sided)

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
python evaluate_rings.py  # (edit config.py for approach 1)

# Run with approach 2
python evaluate_rings.py  # (edit config.py for approach 2)

# Compare final summaries
diff final_summary_v1.csv final_summary_v2.csv
```

### Statistical Significance Testing
```bash
# After running evaluations for both conditions, update paths in statistical_tests.py
python statistical_tests.py

# Review results
cat outputs/statistical_tests_results.csv

# Example output:
# RQ1,E1 vs E4,Pixel Coverage (%),95,48.5,52.3,3.8,1234.5,0.0043,YES (p=0.0043)
# RQ2,Prompt0 vs Prompt1.3,Vegetation F1 Ring 1,95,0.42,0.58,0.16,2105.0,0.0001,YES (p=0.0001)
```

---

## Next Steps

After evaluation:
1. Run `evaluate_rings.py` to generate per-tree metrics and visualizations
2. Run `statistical_tests.py` to test research questions
3. Review final summary: `final_summary_pixelgt.csv`
4. Check visualizations: `visualizations/`
5. Review significance tests: `statistical_tests_results.csv`
6. Identify rings and classes where CLIP performs well vs. poorly
7. Compare R² and MBE across approaches/prompts
8. Use insights for:
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
4. Run evaluation:
   ```bash
   EXPERIMENT_NAME = "aerial2020_dual_composite_prompt1.3_single"
   CLASSIFICATIONS_DIR = ".../stage1_context/dual_composite/classifications"
   python evaluate_rings.py
   ```
5. Run statistical tests:
   ```bash
   python statistical_tests.py
   ```
6. Review `final_summary_pixelgt.csv` and `statistical_tests_results.csv`

---

## Citation

If using this evaluation framework, cite:
- Ring-based buffer methodology for tree-centric spatial analysis
- QGIS rasterization from vector shapefiles for ground truth
- Standard metrics: precision, recall, F1, IoU, R², MBE
- Wilcoxon signed-rank test for non-parametric paired statistical testing
- ✓ Updated Next Steps to include statistical tests
- ✓ Updated Reproducibility to include statistical tests
