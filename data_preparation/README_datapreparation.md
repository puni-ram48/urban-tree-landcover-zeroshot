# Data Preparation — QGIS Dataset Extraction

Extracts tree-centered image crops and generates pixel-level ground truth masks from TreeDataBase vector layers.

---

## Dataset: TreeDataBase

**Status:** Unpublished (request access from authors)

**Includes:**
- ✅ Tree locations, building polygons, vegetation polygons
- ❌ Aerial imagery (you provide)

---

## Quick Start

### 1. Get TreeDataBase
Request access to unpublished TreeDataBase dataset from authors.

### 2. Add Aerial Imagery
Load imagery layer in QGIS (Google Satellite Maps, local, etc.)

### 3. Extract Images
```python
# Edit qgis_image_extraction.py paths, then run in QGIS Console:
exec(open('qgis_image_extraction.py').read())
```

### 4. Generate Ground Truth
```bash
# Edit qgis_pixel_groundtruth_generator.py paths, then run:
python qgis_pixel_groundtruth_generator.py
```

---

## Files

| File | Purpose |
|------|---------|
| `qgis_image_extraction.py` | Extract PNG crops from aerial imagery |
| `qgis_pixel_groundtruth_generator.py` | Generate .npz masks from shapefiles |

---

## Configure

**qgis_image_extraction.py:**
```python
TREE_LAYER_PATHS = ["TreeDataBase/CityErlangen/VectorLayers/treeLocations.shp"]
OUTPUT_DIR = "/path/to/output/images/"
```

**qgis_pixel_groundtruth_generator.py:**
```python
BASE_DIR = "TreeDataBase/CityErlangen/"
OUTPUT_DIR = "/path/to/output/groundtruth/"
```

---

## Output

```
outputs/
├── images/          ← PNG crops + world files
└── groundtruth/     ← .npz masks (vegetation, building, road)
```

---

## Citation

TreeDataBase: [Citation to be provided upon publication]

---

## Next Steps

→ [Segmentation](../segmentation/README.md)
```

---
