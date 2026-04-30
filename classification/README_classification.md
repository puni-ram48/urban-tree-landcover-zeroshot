# Stage 1 Classification — CLIP Context Handling

CLIP-based classification of SAM segments with configurable context-handling approaches and prompt versions.

---

## Quick Start

### 1. Setup
```bash
pip install torch transformers Pillow numpy pandas matplotlib scikit-learn scipy psutil
```

### 2. Configure
Edit `config.py`:
```python
BASE_DIR = "/path/to/project/root"
IMAGES_DIR = os.path.join(BASE_DIR, "data/<dataset_name>/<images_folder>")
SEGMENTS_DIR = os.path.join(BASE_DIR, "<output_folder>/segmentation/outputs/exp04_multiscale_finetuned/segments")
CLIP_MODEL_PATH = os.path.join(BASE_DIR, "models/clip_model/clip-vit-large-patch14-336/")
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "<output_folder>/classification/outputs/stage1_context")

PROMPT_VERSION = "1.3"  # Options: "0", "1", "1.1", "1.2", "1.3"
DEVICE = "cuda"
```

### 3. Test (Optional)
```bash
python exp01_clip_classification.py --approach zero --test       # Test run
python visualize_samples.py --approach zero --test               # Visualize test
```

### 4. Run Experiments

**All four context-handling approaches:**
```bash
python exp01_clip_classification.py --approach zero              # ~30 min
python exp01_clip_classification.py --approach highlight         # ~30 min
python exp01_clip_classification.py --approach larger_crop       # ~30 min
python exp01_clip_classification.py --approach dual_composite    # ~30 min ⭐ BEST
```

**With aggregation mode (optional):**
```bash
python exp01_clip_classification.py --approach zero --aggregation average
python exp01_clip_classification.py --approach highlight --aggregation average
```

### 5. Visualize Results
```bash
# Single approach
python visualize_samples.py --approach zero
python visualize_samples.py --approach highlight

# Compare all approaches side-by-side
python visualize_samples.py --compare
python visualize_samples.py --compare --max_images 10
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
