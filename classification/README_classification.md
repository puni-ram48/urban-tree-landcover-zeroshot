# CLIP Classification Pipeline

Contrastive Language–Image Pretraining (CLIP-based classification of SAM segments with configurable context-handling approaches and prompt versions.

---

## Quick Start

### 1. Setup 
```bash
# already installed requirements.txt in segmentation phase
pip install -r requirements.txt --break-system-packages 
```
### CLIP Model Setup
The CLIP model is automatically downloaded the first time the pipeline is executed.

Unlike the default Hugging Face cache (~/.cache/huggingface), this project stores the model locally inside the project directory:
```bash
models/clip_model/
```

### 2. Configure
Edit `config.py`:
```python
BASE_DIR = "/path/to/project/root"
IMAGES_DIR = os.path.join(BASE_DIR, "dataset/<images_folder>")
SEGMENTS_DIR = os.path.join(BASE_DIR, "segmentation/outputs/<experiment_name>/segments")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models/clip_model/clip-vit-large-patch14-336/")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "classification/outputs/")

PROMPT_VERSION = "1.3"  # Options: "0", "1", "1.1", "1.2", "1.3"
DEVICE = "cuda"
```

### 3. Test (Optional)
```bash
python classification/clip_classification.py --approach zero --test       # Test run
python classification/visualize_samples.py --approach zero --test               # Visualize test
```

### 4. Run Experiments

**All four context-handling approaches:**
```bash
python classification/clip_classification.py  --approach zero              
python classification/clip_classification.py --approach highlight         
python classification/clip_classification.py --approach larger_crop       
python classification/clip_classification.py --approach dual_composite    
```

**With aggregation mode (optional):**
```bash
python classification/clip_classification.py --approach zero --aggregation average
python classification/clip_classification.py --approach highlight --aggregation average
```

### 5. Visualize Results
```bash
# Single approach
python classification/visualize_samples.py  --approach zero
python classification/visualize_samples.py  --approach highlight

# Compare all approaches side-by-side
python classification/visualize_samples.py  --compare
python classification/visualize_samples.py  --compare --max_images 10
```

---

## Experiments Overview

| Approach | Type | Context | Best For |
|----------|------|---------|----------|
| **zero** | Baseline | Isolated segment, black background | Segment alone |
| **highlight** | Context | Full image with cyan highlight | Global + visual cue |
| **larger_crop** | Context | 3× bounding box with real pixels | Natural surroundings |
| **dual_composite** | Context | Side-by-side global + local | ⭐ **RECOMMENDED** |

**Prompt Versions (select via `PROMPT_VERSION` in config.py):**

| Version | Description | Use |
|---------|-------------|-----|
| **0** | Minimal ("a photo of vegetation") | Baseline |
| **1** | Basic aerial descriptions | Standard |
| **1.1** | Material-focused (buildings) | Conservative |
| **1.2** | Color diversity (vegetation) | Bolder |
| **1.3** | Balanced across all classes | ⭐ **RECOMMENDED** |

---

## Output Structure

```
outputs/
├── zero_single/
│   ├── classifications/           ← {devEUI}_classification.npy
│   ├── metrics/
│   │   ├── per_image_metrics.csv
│   │   ├── summary_metrics.csv
│   │   └── terminal_log.txt
│   └── logs/
│       └── processing.log
├── highlight_single/
├── larger_crop_single/
├── dual_composite_single/
├── zero_average/                  ← alternative aggregation mode
├── highlight_average/
├── larger_crop_average/
├── dual_composite_average/
├── visualizations/
│   ├── zero_single/
│   │   └── {devEUI}_viz.png       ← 3-panel: original | overlay | heatmap
│   ├── highlight_single/
│   ├── larger_crop_single/
│   ├── dual_composite_single/
│   ├── comparison/
│   │   └── {devEUI}_comparison.png ← all 4 approaches side-by-side
│   └── stage1_approach_comparison.csv
├── stage1_aggregate_summary.csv    ← summary across all approaches
└── checkpoint.npy                 ← intermediate checkpoint (removed on completion)
```

---

## Results Interpretation

**Key Metrics (from CSV files):**

- **mean_confidence_overall** — Average CLIP confidence 0-1 (>0.50 is good)
- **silhouette_score** — Embedding space separation (>0.3 is good)
- **spatial_consistency_score** — Neighbour agreement 0-1 (>0.5 is good)
- **low_confidence_percentage** — % of segments with confidence <0.3
- **pixel_coverage_percent** — % of image covered by each class
- **processing_time_seconds** — Time per image

---

## Files

| File | Purpose |
|------|---------|
| **clip_classification.py** | Main experiment script (run this) |
| **visualize_samples.py** | Generate visualization PNGs |
| **config.py** | Configuration & parameters |
| **utils.py** | Shared utilities (logging, CLIP, metrics, I/O) |

---

## Configuration Reference

### Essential Settings
```python
# Paths (edit in config.py)
BASE_DIR = "/path/to/project/root"
IMAGES_DIR = "/path/to/images"
SEGMENTS_DIR = "/path/to/segments"
CLIP_MODEL_PATH = "/path/to/clip/model"
OUTPUT_BASE_DIR = "/path/to/outputs"

# Prompt selection
PROMPT_VERSION = "1.3"  # "0", "1", "1.1", "1.2", or "1.3"

# Device
DEVICE = "cuda"  # or "cpu"
```

### Context-Handling Parameters
```python
# For highlight & dual_composite
HIGHLIGHT_COLOR = (0, 255, 255)    # cyan tint
HIGHLIGHT_ALPHA = 0.35             # opacity

# For larger_crop
LARGER_CROP_FACTOR = 3             # bounding box multiplier
```

### Classes & Colors
```python
CLASS_NAMES = ["Vegetation", "Building", "Road"]

CLASS_COLORS = {
    "Vegetation": [0.059, 0.416, 0.196, 0.6],  # dark green
    "Building":   [0.624, 0.184, 0.184, 0.6],  # dark red
    "Road":       [0.365, 0.349, 0.349, 0.6],  # dark gray
}
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Images not found | Check `IMAGES_DIR` in config.py |
| Segments not found | Ensure exp04 completed, check `SEGMENTS_DIR` |
| CLIP model missing | Download from HuggingFace, update `CLIP_MODEL_PATH` |
| CUDA out of memory | Reduce `CLIP_BATCH_SIZE` from 16 to 8 in config.py |
| Slow processing | Verify GPU: `nvidia-smi` |

---

## Tips

**Resume interrupted run:**
```bash
python exp01_clip_classification.py --approach zero
# Automatically skips already-processed images via checkpoint
```

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

**Check logs:**
```bash
tail -f outputs/stage1_context/zero_single/logs/processing.log
```

**Test different prompt versions:**
```bash
# Edit config.py: PROMPT_VERSION = "1.3"
python exp01_clip_classification.py --approach zero

# Then try another version
# Edit config.py: PROMPT_VERSION = "1.2"
python exp01_clip_classification.py --approach zero

# Compare results
python visualize_samples.py --approach zero
```

---

## For More Details

- **All configuration options** — See `config.py` comments
- **Utility functions** — See `utils.py` docstrings
- **CLIP model details** — Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
- **SAM segments source** — From `exp04_multiscale_finetuned`

---

## Next Steps

After classification:
1. Review aggregate metrics: `outputs/stage1_context/stage1_aggregate_summary.csv`
2. View visualizations: `python visualize_samples.py --compare`
3. Compare approaches and aggregation modes
4. Analyze per-image metrics for failure modes
```
