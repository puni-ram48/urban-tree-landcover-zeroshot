#!/usr/bin/env python3
"""
QGIS Image Extraction Script — Tree-Centered Crops from Aerial Imagery

Extracts tree-centered rectangular image crops from aerial/satellite imagery
at multiple DPI levels using QGIS rendering engine.

Input:
    - Aerial/satellite imagery layer (loaded in QGIS, visible in canvas)
    - Vector shapefile with tree locations (point layer, devEUI field)

Output:
    - PNG images: {devEUI}_{suffix}.png (tree-centered crops)
    - PGW world files: {devEUI}_{suffix}.pgw (geo-coordinate reference)
    - Multiple DPI levels: 300 (_1), 225 (_2), 150 (_3)

DPI Levels & Coverage:
    300 DPI (_1): 39.5m × 39.5m (Fine detail)
    225 DPI (_2): 52.6m × 52.6m (Balanced) ← Recommended for evaluation
    150 DPI (_3): 78.9m × 78.9m (Wide view)

Usage:
    Run in QGIS Python Console:
    exec(compile(Path('/path/to/qgis_image_extraction.py').read_text(),
                 '/path/to/qgis_image_extraction.py', 'exec'))

    Or paste directly into QGIS Console:
    exec(open('/path/to/qgis_image_extraction.py').read())

Requirements:
    - QGIS 3.16+ with Python 3.8+
    - Aerial/satellite imagery layer loaded and visible in map canvas
    - Vector shapefile with tree locations and devEUI field
    - Active QGIS project

Example Output:
    8C1F640980000014_1.png
    8C1F640980000014_1.pgw
    8C1F640980000014_2.png
    8C1F640980000014_2.pgw
    ...
"""

from qgis.core import (QgsProject, QgsRectangle, QgsMapSettings,
                       QgsMapRendererParallelJob)
from qgis.utils import iface
from PyQt6.QtCore import QSize, QEventLoop
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor
import os
import time

# ============================================================================
# CONFIGURATION — Edit these paths for your dataset
# ============================================================================

# Tree location shapefiles (add both CityErlangen and UniversityErlangen)
TREE_LAYER_PATHS = [
    "/path/to/GISData/CityErlangen/VectorLayers/treeLocations.shp",
    "/path/to/GISData/UniversityErlangen/VectorLayers/treeLocations.shp",
]

# Output directory for extracted images
OUTPUT_DIR = "/path/to/output/Erlangen_All_TreeImages/"

# Image parameters (must match pixel ground truth generation)
SCALE = 500              # Map scale 1:SCALE
OUTPUT_WIDTH = 1024      # Image width in pixels
OUTPUT_HEIGHT = 1024     # Image height in pixels

# DPI levels (DPI → suffix mapping)
DPI_CONFIGS = [
    {"dpi": 300, "suffix": "_1", "name": "Fine"},
    {"dpi": 225, "suffix": "_2", "name": "Medium"},
    {"dpi": 150, "suffix": "_3", "name": "Coarse"},
]

# ============================================================================
# UTILITIES
# ============================================================================

def compute_coverage_meters(dpi):
    """Calculate image coverage in meters for given DPI."""
    return (OUTPUT_WIDTH / dpi) * 0.0254 * SCALE


def log_header(title):
    """Print formatted header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def log_config():
    """Print export configuration summary."""
    log_header("EXPORT CONFIGURATION")
    print(f"\nImage parameters:")
    print(f"  Scale: 1:{SCALE}")
    print(f"  Size: {OUTPUT_WIDTH}×{OUTPUT_HEIGHT} pixels")
    print(f"\nDPI Levels:")
    for cfg in DPI_CONFIGS:
        coverage = compute_coverage_meters(cfg["dpi"])
        print(f"  Level {cfg['suffix']}: {cfg['dpi']:>3} DPI → "
              f"{coverage:>5.1f}m × {coverage:>5.1f}m ({cfg['name']})")
    print()


# ============================================================================
# LOAD TREE LAYERS
# ============================================================================

print("\n" + "=" * 75)
print("  QGIS IMAGE EXTRACTION")
print("=" * 75)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nInitializing QGIS...")
project = QgsProject.instance()
canvas = iface.mapCanvas()
visible_layers = canvas.layers()

print(f" QGIS project loaded")
print(f" Visible layers in canvas:")
for lyr in visible_layers:
    print(f"    - {lyr.name()}")

# Find tree layers matching TREE_LAYER_PATHS
print(f"\nSearching for tree location layers...")
tree_layers = []
for lyr in QgsProject.instance().mapLayers().values():
    if lyr.source() in TREE_LAYER_PATHS:
        tree_layers.append(lyr)
        print(f"   Found: {lyr.name()} ({lyr.featureCount()} trees)")

if not tree_layers:
    print(f"\n ERROR: No tree layers found!")
    print(f"  Check TREE_LAYER_PATHS in script configuration")
    raise SystemExit(1)

# ============================================================================
# COLLECT TREE FEATURES
# ============================================================================

print(f"\nCollecting tree features...")
all_features = []
skipped_count = 0

for layer in tree_layers:
    for feature in layer.getFeatures():
        dev_eui = feature["devEUI"]
        
        # Validate devEUI
        if not dev_eui or str(dev_eui).strip() == "":
            skipped_count += 1
            continue
        
        point = feature.geometry().asPoint()
        all_features.append({
            "devEUI": str(dev_eui).strip(),
            "point": point,
            "layer": layer.name(),
        })

total_trees = len(all_features)
total_images = total_trees * len(DPI_CONFIGS)

print(f"✓ Collected {total_trees} trees")
if skipped_count > 0:
    print(f" Skipped {skipped_count} features without devEUI")
print(f"  Total images to export: {total_images} "
      f"({total_trees} trees × {len(DPI_CONFIGS)} DPI levels)")

# Check for duplicate devEUIs
dev_euids = [f["devEUI"] for f in all_features]
duplicates = [d for d in set(dev_euids) if dev_euids.count(d) > 1]

if duplicates:
    print(f"\n WARNING: Duplicate devEUIs found: {duplicates}")
    print(f"  These trees will overwrite each other during export!")
else:
    print(f" No duplicate devEUIs")

log_config()

# ============================================================================
# EXPORT IMAGES
# ============================================================================

log_header("EXPORTING IMAGES")

start_time = time.time()
images_exported = 0
errors = 0

for idx, tree in enumerate(all_features):
    dev_eui = tree["devEUI"]
    point = tree["point"]
    
    for cfg in DPI_CONFIGS:
        dpi = cfg["dpi"]
        suffix = cfg["suffix"]
        
        try:
            # Compute extent (tree-centered rectangle)
            meters_width = compute_coverage_meters(dpi)
            meters_height = compute_coverage_meters(dpi)
            
            extent = QgsRectangle(
                point.x() - meters_width / 2,
                point.y() - meters_height / 2,
                point.x() + meters_width / 2,
                point.y() + meters_height / 2,
            )
            
            # Configure map rendering settings
            settings = QgsMapSettings()
            settings.setLayers(visible_layers)
            settings.setBackgroundColor(QColor(255, 255, 255))
            settings.setOutputSize(QSize(OUTPUT_WIDTH, OUTPUT_HEIGHT))
            settings.setExtent(extent)
            settings.setOutputDpi(dpi)
            settings.setDestinationCrs(project.crs())
            
            # Quality flags
            settings.setFlag(QgsMapSettings.Flag.Antialiasing, True)
            settings.setFlag(QgsMapSettings.Flag.UseAdvancedEffects, True)
            settings.setFlag(QgsMapSettings.Flag.DrawLabeling, True)
            settings.setFlag(QgsMapSettings.Flag.HighQualityImageTransforms, True)
            
            # Create image canvas
            image = QImage(
                QSize(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                QImage.Format.Format_ARGB32_Premultiplied
            )
            image.setDotsPerMeterX(int(dpi / 25.4 * 1000))
            image.setDotsPerMeterY(int(dpi / 25.4 * 1000))
            image.fill(QColor(255, 255, 255))
            
            # Render map to image
            job = QgsMapRendererParallelJob(settings)
            loop = QEventLoop()
            job.finished.connect(loop.quit)
            job.start()
            loop.exec()
            
            # Allow rendering to complete
            time.sleep(2)
            QApplication.processEvents()
            
            image = job.renderedImage()
            
            # Save PNG image
            image_path = os.path.join(OUTPUT_DIR, f"{dev_eui}{suffix}.png")
            image.save(image_path, "PNG", 100)
            
            # Save PGW world file (geo-reference)
            pgw_path = os.path.join(OUTPUT_DIR, f"{dev_eui}{suffix}.pgw")
            pixel_size_x = meters_width / OUTPUT_WIDTH
            pixel_size_y = meters_height / OUTPUT_HEIGHT
            
            with open(pgw_path, "w") as f:
                f.write(f"{pixel_size_x}\n")
                f.write("0.0\n")
                f.write("0.0\n")
                f.write(f"{-pixel_size_y}\n")
                f.write(f"{extent.xMinimum() + pixel_size_x / 2}\n")
                f.write(f"{extent.yMaximum() - pixel_size_y / 2}\n")
            
            images_exported += 1
        
        except Exception as e:
            print(f"  ✗ Error exporting {dev_eui}{suffix}: {e}")
            errors += 1
    
    # Progress report every 10 trees
    if (idx + 1) % 10 == 0 or idx == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / max(images_exported, 1)
        remaining_time = (total_images - images_exported) * avg_time
        
        print(f"  [{idx+1:>3}/{total_trees}] {dev_eui:20s} "
              f"({tree['layer']}) → {len(DPI_CONFIGS)} DPI | "
              f"~{remaining_time/60:.1f} min remaining")

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

log_header("EXPORT COMPLETE")

print(f"\nResults:")
print(f"  Trees processed    : {total_trees}")
print(f"  Images exported    : {images_exported}")
print(f"  Errors             : {errors}")
print(f"  Total time         : {total_time/60:.1f} minutes")
print(f"  Average per image  : {(total_time/max(images_exported,1)):.2f} seconds")

print(f"\nOutput:")
print(f"  Directory: {OUTPUT_DIR}")
print(f"  File format: {{devEUI}}{{suffix}}.png / {{devEUI}}{{suffix}}.pgw")

if all_features:
    example_eui = all_features[0]["devEUI"]
    print(f"\nExample files:")
    print(f"  {example_eui}_1.png  (300 DPI)")
    print(f"  {example_eui}_2.png  (225 DPI) ← Use for evaluation")
    print(f"  {example_eui}_3.png  (150 DPI)")

print("\n Images ready for next step: Pixel ground truth generation")
print("=" * 75 + "\n")
