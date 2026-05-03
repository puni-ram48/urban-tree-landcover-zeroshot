#!/usr/bin/env python3
"""
QGIS Pixel-Level Ground Truth Generator

Generates pixel-level binary masks from vector shapefiles for ground truth
evaluation. Rasterizes building and vegetation polygons to match QGIS-exported
image dimensions and DPI levels.

Input:
    - Tree locations shapefile (point layer with devEUI field)
    - Building polygons (per-ring: buildings2m5.shp, buildings5m0.shp, buildings7m5.shp)
    - Vegetation polygons (per-ring: greenAtta*.shp, greenDeta*.shp)
    - All shapefiles in EPSG:25832 (UTM Zone 32N)

Output:
    - .npz files: {devEUI}_{suffix}_gt.npz
    - Keys: vegetation, building, road (binary masks, 1024×1024)
    - DPI levels: _1 (300 DPI), _2 (225 DPI), _3 (150 DPI)

File Structure:
    {devEUI}_1_gt.npz  ← 300 DPI (39.5m × 39.5m)
    {devEUI}_2_gt.npz  ← 225 DPI (52.6m × 52.6m) [For evaluation]
    {devEUI}_3_gt.npz  ← 150 DPI (78.9m × 78.9m)

Mask Classes:
    vegetation: Pixels covered by greenAtta OR greenDeta polygons
    building:   Pixels covered by building polygons
    road:       Remaining pixels (not vegetation, not building)

Usage:
    python qgis_pixel_groundtruth_generator.py

Requirements:
    - geopandas>=0.12.0
    - rasterio>=1.3.0
    - shapely>=2.0.0
    - numpy>=1.21.0

    Install with: pip install geopandas rasterio shapely numpy
"""

import os
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ============================================================================
# CONFIGURATION — Edit these paths for your dataset
# ============================================================================

# Base directory containing all GIS data
BASE_DIR = "/path/to/GISData/"

# Tree locations shapefile (point layer with devEUI field)
TREE_SHAPEFILE = os.path.join(BASE_DIR, "treeLocations.shp")

# Building polygon shapefiles (one per ring distance)
BUILDINGS = {
    "2m5": os.path.join(BASE_DIR, "buildings2m5.shp"),
    "5m0": os.path.join(BASE_DIR, "buildings5m0.shp"),
    "7m5": os.path.join(BASE_DIR, "buildings7m5.shp"),
}

# Vegetation attached polygons (greenAtta)
VEG_ATTA = {
    "2m5": os.path.join(BASE_DIR, "greenAtta2m5.shp"),
    "5m0": os.path.join(BASE_DIR, "greenAtta5m0.shp"),
    "7m5": os.path.join(BASE_DIR, "greenAtta7m5.shp"),
}

# Vegetation detached polygons (greenDeta)
VEG_DETA = {
    "2m5": os.path.join(BASE_DIR, "greenDeta2m5.shp"),
    "5m0": os.path.join(BASE_DIR, "greenDeta5m0.shp"),
    "7m5": os.path.join(BASE_DIR, "greenDeta7m5.shp"),
}

# Output directory for .npz ground truth files
OUTPUT_DIR = os.path.join(BASE_DIR, "pixel_groundtruth_225dpi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image parameters (MUST MATCH qgis_image_extraction.py)
IMAGE_SIZE = 1024        # Image width/height in pixels
SCALE = 500              # Map scale 1:SCALE

# DPI configurations (MUST MATCH qgis_image_extraction.py)
DPI_CONFIG = {
    300: "_1",  # Fine detail
    225: "_2",  # Balanced (Recommended for evaluation)
    150: "_3",  # Wide view
}

# Coordinate Reference System (should be UTM Zone 32N for Erlangen)
CRS = 25832

# ============================================================================
# UTILITIES
# ============================================================================

def compute_meter_span(dpi):
    """Calculate ground coverage in meters for given DPI level."""
    return (IMAGE_SIZE / dpi) * 0.0254 * SCALE


def create_transform(center_x, center_y, meter_span):
    """Create rasterio transform for tree-centered window."""
    pixel_size = meter_span / IMAGE_SIZE
    left = center_x - meter_span / 2
    top = center_y + meter_span / 2
    return from_origin(left, top, pixel_size, pixel_size)


def rasterize_geometries(geometries, transform, shape):
    """Rasterize geometry collection to binary mask."""
    if len(geometries) == 0:
        return np.zeros(shape, dtype=np.uint8)
    
    return rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )


def log_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


# ============================================================================
# LOAD SHAPEFILES
# ============================================================================

log_header("LOADING SHAPEFILES")

print("\nLoading tree locations...")
try:
    trees = gpd.read_file(TREE_SHAPEFILE).to_crs(CRS)
    print(f" Loaded {len(trees)} trees")
except Exception as e:
    print(f" ERROR loading trees: {e}")
    raise SystemExit(1)

print("\nLoading building polygons...")
try:
    buildings = {k: gpd.read_file(v).to_crs(CRS) for k, v in BUILDINGS.items()}
    for ring, gdf in buildings.items():
        print(f"   {ring}: {len(gdf)} polygons")
except Exception as e:
    print(f"✗ ERROR loading buildings: {e}")
    raise SystemExit(1)

print("\nLoading vegetation (attached)...")
try:
    veg_a = {k: gpd.read_file(v).to_crs(CRS) for k, v in VEG_ATTA.items()}
    for ring, gdf in veg_a.items():
        print(f"   {ring}: {len(gdf)} polygons")
except Exception as e:
    print(f"✗ ERROR loading vegetation (attached): {e}")
    raise SystemExit(1)

print("\nLoading vegetation (detached)...")
try:
    veg_d = {k: gpd.read_file(v).to_crs(CRS) for k, v in VEG_DETA.items()}
    for ring, gdf in veg_d.items():
        print(f"   {ring}: {len(gdf)} polygons")
except Exception as e:
    print(f"✗ ERROR loading vegetation (detached): {e}")
    raise SystemExit(1)

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

log_header("CONFIGURATION")

print(f"\nImage parameters:")
print(f"  Size: {IMAGE_SIZE}×{IMAGE_SIZE} pixels")
print(f"  Scale: 1:{SCALE}")
print(f"  CRS: EPSG:{CRS}")

print(f"\nDPI Levels:")
for dpi, suffix in DPI_CONFIG.items():
    span = compute_meter_span(dpi)
    print(f"  {suffix}: {dpi} DPI → {span:.1f}m × {span:.1f}m coverage")

print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# GENERATE GROUND TRUTH
# ============================================================================

log_header("GENERATING GROUND TRUTH")

total_trees = len(trees)
total_files = total_trees * len(DPI_CONFIG)
files_created = 0
errors = 0

print(f"\nProcessing {total_trees} trees × {len(DPI_CONFIG)} DPI levels = "
      f"{total_files} .npz files...\n")

for idx, (_, tree_row) in enumerate(trees.iterrows()):
    
    tree_id = tree_row["devEUI"]
    
    # Handle geometry safely
    geom = tree_row.geometry
    if geom.geom_type != "Point":
        geom = geom.centroid
    
    x, y = geom.x, geom.y
    
    # Process each DPI level
    for dpi, suffix in DPI_CONFIG.items():
        
        try:
            # Compute coverage and transform
            meter_span = compute_meter_span(dpi)
            transform = create_transform(x, y, meter_span)
            
            # Initialize combined masks
            vegetation_combined = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            building_combined = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            
            # Rasterize all rings and accumulate
            for ring in ["2m5", "5m0", "7m5"]:
                
                # Get polygon collections for this ring
                va_geoms = veg_a[ring].geometry.tolist()
                vd_geoms = veg_d[ring].geometry.tolist()
                bld_geoms = buildings[ring].geometry.tolist()
                
                # Rasterize each layer
                veg_att = rasterize_geometries(va_geoms, transform, 
                                               (IMAGE_SIZE, IMAGE_SIZE))
                veg_det = rasterize_geometries(vd_geoms, transform, 
                                               (IMAGE_SIZE, IMAGE_SIZE))
                bld_px = rasterize_geometries(bld_geoms, transform, 
                                              (IMAGE_SIZE, IMAGE_SIZE))
                
                # Combine (vegetation = attached OR detached)
                vegetation_combined |= (veg_att | veg_det).astype(np.uint8)
                building_combined |= bld_px.astype(np.uint8)
            
            # Road = pixels that are neither vegetation nor building
            road_combined = ((vegetation_combined + building_combined) == 0).astype(np.uint8)
            
            # Save .npz file
            output_path = os.path.join(OUTPUT_DIR, f"{tree_id}{suffix}_gt.npz")
            np.savez_compressed(
                output_path,
                vegetation=vegetation_combined,
                building=building_combined,
                road=road_combined
            )
            
            files_created += 1
        
        except Exception as e:
            print(f"   Error processing {tree_id}{suffix}: {e}")
            errors += 1
    
    # Progress report every 10 trees
    if (idx + 1) % 10 == 0 or idx == 0:
        print(f"  [{idx+1:>3}/{total_trees}] {tree_id:20s} → "
              f"{len(DPI_CONFIG)} DPI levels processed")

# ============================================================================
# SUMMARY
# ============================================================================

log_header("GENERATION COMPLETE")

print(f"\nResults:")
print(f"  Trees processed : {total_trees}")
print(f"  Files created   : {files_created}")
print(f"  Errors          : {errors}")
print(f"  Total files     : {total_files}")

print(f"\nOutput directory: {OUTPUT_DIR}")

print(f"\nExample files:")
if len(trees) > 0:
    example_eui = trees.iloc[0]["devEUI"]
    print(f"  {example_eui}_1_gt.npz  (300 DPI)")
    print(f"  {example_eui}_2_gt.npz  (225 DPI) ← Use for evaluation")
    print(f"  {example_eui}_3_gt.npz  (150 DPI)")

print(f"\nFile contents (.npz):")
print(f"   vegetation  (1024×1024, uint8, binary: 0 or 1)")
print(f"   building    (1024×1024, uint8, binary: 0 or 1)")
print(f"   road        (1024×1024, uint8, binary: 0 or 1)")

print("\n Ground truth ready for evaluation pipeline")
print("=" * 75 + "\n")
