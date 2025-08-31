#!/usr/bin/env python3
# scripts/generate_resnet_unet_labels.py
# -*- coding: utf-8 -*-
r"""
Generate a unified master dataset (features + both labels) from Google Earth Engine.

Exports TFRecord patches with:
  Inputs (bands):
    - B4, B3, B2 (RGB)                 [ResNet]
    - B8 (NIR)                          [U-Net]
    - Slope (from SRTM)                 [U-Net]
    - Precipitation (WorldClim bio12)   [U-Net]
    - Soil pH (OpenLandMap; fallback 7) [U-Net]

  Labels (bands):
    - resnet_label (degradation classes: 0=natural,1=moderate,2=degraded)
    - unet_label   (suitability: 0=unsuitable (water/built-up), 1=moderate, 2=protected)

AOI modes:
  - Single state/UT:   --state "Uttarakhand"
  - Pan-India:         --pan-india
  - Curated 50 regions: --catalog-50
"""

from __future__ import annotations

import argparse
import sys
import math
import logging
from typing import List, Optional, Dict, Any

import ee

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s"
)

# ------------------------------
# EE INIT
# ------------------------------
def ee_init(project_id: Optional[str] = None) -> None:
    """Initialize EE in the current environment."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception as e:
        sys.stderr.write(
            "[EE] Initialization failed. Make sure you ran `earthengine authenticate` "
            "and optionally `earthengine set_project <PROJECT_ID>`.\n"
            f"Error: {e}\n"
        )
        raise

# ------------------------------
# STRING / GEOMETRY HELPERS
# ------------------------------
def slugify(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )

def rect_geom(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> ee.Geometry:
    """Axis-aligned rectangle (geodesic=False for deterministic tiling)."""
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj=None, geodesic=False)

def circle_km(lon: float, lat: float, radius_km: float) -> ee.Geometry:
    """Circle in kilometers (approx)."""
    return ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000.0).bounds()

# ------------------------------
# AOI SOURCES
# ------------------------------
def state_geometry(state_name: str) -> ee.Geometry:
    """
    Returns the geometry for an Indian state/UT using GAUL level 1.
    Raises ValueError if not found.
    """
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")
    india_states = gaul.filter(ee.Filter.eq("ADM0_NAME", "India"))
    match = india_states.filter(ee.Filter.eq("ADM1_NAME", state_name))
    feat = ee.Feature(match.first())
    ok = ee.Algorithms.IsEqual(feat, None).Not()
    if ok.getInfo() is False:
        raise ValueError(
            f"State '{state_name}' not found in GAUL/2015 level1. "
            "Check spelling/casing (e.g., 'Uttarakhand', 'Maharashtra')."
        )
    return ee.Geometry(feat.geometry())

def get_india_geometry() -> ee.Geometry:
    """Returns a geometry for all of India (includes islands) from GAUL level 0."""
    india = (
        ee.FeatureCollection("FAO/GAUL/2015/level0")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .geometry()
    )
    return india.buffer(0)

# ------------------------------
# CURATED 50-REGION CATALOG
# ------------------------------
_STATES: List[str] = [
    # 28 States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]

# 8 UTs as in GAUL 2015 (pre-reorg; Ladakh separate likely not present)
_UTS: List[str] = [
    "Delhi", "Puducherry", "Chandigarh", "Daman and Diu",
    "Dadra and Nagar Haveli", "Andaman and Nicobar", "Lakshadweep",
    "Jammu and Kashmir",
]

# GAUL-2015 alias spellings to ensure ADM1 match
ADM1_ALIASES: Dict[str, List[str]] = {
    "Odisha": ["Odisha", "Orissa"],
    "Puducherry": ["Puducherry", "Pondicherry"],
    "Andaman and Nicobar": ["Andaman and Nicobar", "Andaman and Nicobar Islands"],
    "Delhi": ["Delhi", "NCT of Delhi"],
    "Jammu and Kashmir": ["Jammu and Kashmir", "Jammu & Kashmir"],
    "Uttarakhand": ["Uttarakhand", "Uttaranchal"],
    # usually fine, but harmless to include explicit forms:
    "Dadra and Nagar Haveli": ["Dadra and Nagar Haveli"],
    "Daman and Diu": ["Daman and Diu"],
    "Telangana": ["Telangana"],
}

def _rect_geom(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> ee.Geometry:
    # Create rectangles only after EE is initialized
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj=None, geodesic=False)

# 14 special ecoregions/landscapes to reach 50 total — store as bounds (no EE calls yet)
_SPECIAL_BOUNDS: List[Dict[str, Any]] = [
    {"name": "western_ghats_north",   "bounds": [72.8, 17.0, 75.3, 20.8]},
    {"name": "western_ghats_central", "bounds": [74.0, 13.0, 76.0, 16.5]},
    {"name": "western_ghats_south",   "bounds": [76.0,  8.0, 77.5, 12.2]},
    {"name": "eastern_ghats_north",   "bounds": [83.0, 18.0, 85.5, 21.0]},
    {"name": "eastern_ghats_south",   "bounds": [79.0, 13.0, 81.0, 16.0]},
    {"name": "himalaya_uttarakhand",  "bounds": [79.0, 29.5, 81.1, 31.5]},
    {"name": "himalaya_himachal",     "bounds": [76.0, 31.0, 78.0, 32.8]},
    {"name": "kashmir_valley",        "bounds": [74.0, 33.5, 75.7, 34.7]},
    {"name": "meghalaya_hills",       "bounds": [90.0, 25.0, 92.7, 26.5]},
    {"name": "assam_floodplains",     "bounds": [91.0, 26.5, 95.0, 27.7]},
    {"name": "thar_desert_core",      "bounds": [70.0, 25.0, 73.5, 28.5]},
    {"name": "rann_of_kutch",         "bounds": [68.5, 23.0, 71.0, 24.8]},
    {"name": "sundarbans",            "bounds": [88.0, 21.5, 89.6, 22.6]},
    {"name": "deccan_plateau_core",   "bounds": [75.0, 15.0, 80.0, 20.0]},
]

def resolve_state_geometry(canonical_name: str) -> ee.Geometry:
    """Try GAUL lookups with alias variants before giving up."""
    # try the canonical name first
    try:
        return state_geometry(canonical_name)
    except Exception:
        pass
    # then try aliases if defined
    for alt in ADM1_ALIASES.get(canonical_name, []):
        try:
            return state_geometry(alt)
        except Exception:
            continue
    raise ValueError(
        f"No GAUL match for '{canonical_name}' "
        f"(tried: {[canonical_name] + ADM1_ALIASES.get(canonical_name, [])})"
    )

def get_catalog_50() -> List[Dict[str, Any]]:
    """
    Returns 50 regions total:
      - 36 state/UT AOIs (resolved via GAUL with aliases)
      - 14 special AOIs (rectangles created post-init)
    Each item: { "name": <slug>, "geom": ee.Geometry, "state_name"?: str }
    """
    items: List[Dict[str, Any]] = []
    found, skipped = 0, 0

    # States & UTs
    for nm in _STATES + _UTS:
        try:
            geom = resolve_state_geometry(nm)
            items.append({"name": slugify(nm), "geom": geom, "state_name": nm})
            found += 1
        except Exception as e:
            logging.warning(f"Skipping '{nm}' after alias tries: {e}")
            skipped += 1

    # Special rectangles — turn bounds into geometries now (EE is initialized)
    for spec in _SPECIAL_BOUNDS:
        lon_min, lat_min, lon_max, lat_max = spec["bounds"]
        geom = _rect_geom(lon_min, lat_min, lon_max, lat_max)
        items.append({"name": slugify(spec["name"]), "geom": geom})

    logging.info(
        f"Catalog build summary → states/UTs found: {found}, "
        f"skipped: {skipped}, specials: {len(_SPECIAL_BOUNDS)}"
    )

    # Ensure we return at most 50; normally this will be exactly 50
    return items[:50]

# ------------------------------
# SENTINEL-2 + AUXILIARY LAYERS
# ------------------------------
def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using QA60 bits 10/11 (cloud & cirrus)."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask)

def _s2_rgbnir(aoi: ee.Geometry, start: str, end: str) -> ee.Image:
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(_mask_s2_clouds)
        .map(lambda img: img.updateMask(img.select("B8").gt(0)))
    )
    comp = s2.median().clip(aoi)
    return comp.select(["B4", "B3", "B2", "B8"]).multiply(0.0001).rename(["B4", "B3", "B2", "B8"])

def _slope(aoi: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003")
    return ee.Terrain.slope(dem).rename("Slope").clip(aoi)

def _precip(aoi: ee.Geometry) -> ee.Image:
    # Try WorldClim v1 (bio12). Fallback to v2 (BIO12) if needed.
    try:
        return ee.Image("WORLDCLIM/V1/BIO").select("bio12").rename("Precip").clip(aoi)
    except Exception:
        return ee.Image("WORLDCLIM/V2/BIO").select("BIO12").rename("Precip").clip(aoi)

def _soil_ph(aoi: ee.Geometry) -> ee.Image:
    # Safe fallback: constant 7.0 if dataset not available
    try:
        src = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
        band_names = src.bandNames()
        first = ee.String(band_names.get(0))
        return src.select(first).rename("SoilPH").clip(aoi)
    except Exception:
        return ee.Image.constant(7.0).rename("SoilPH").clip(aoi)

def build_master_feature_stack(aoi: ee.Geometry, start: str, end: str, scale: int) -> ee.Image:
    """
    Builds a comprehensive stack with all features for both ResNet and U-Net:
      B4,B3,B2,B8,Slope,Precip,SoilPH
    All reprojected to a common scale for stable patch extraction.
    """
    rgb_nir = _s2_rgbnir(aoi, start, end)  # B4,B3,B2,B8
    slope = _slope(aoi)
    precip = _precip(aoi)
    soil = _soil_ph(aoi)

    proj = rgb_nir.select("B4").projection().atScale(scale)
    stack = (
        rgb_nir
        .addBands(slope.reproject(proj))
        .addBands(precip.reproject(proj))
        .addBands(soil.reproject(proj))
        .toFloat()
        .clip(aoi)
    )
    return stack

# ------------------------------
# LABELS
# ------------------------------
def _worldcover_year_map(aoi: ee.Geometry, year: int) -> ee.Image:
    ic = ee.ImageCollection("ESA/WorldCover/v200")
    img = ic.filter(ee.Filter.eq("year", year)).first()
    img = ee.Image(ee.Algorithms.If(img, img, ic.mosaic()))
    return ee.Image(img).select("Map").clip(aoi).toInt()

def build_unet_label_mask(aoi: ee.Geometry, year: int = 2021) -> ee.Image:
    """
    U-Net suitability:
      2: Protected areas (presence)
      1: Elsewhere & forest-loss areas (restoration opportunity)
      0: Water or Built-up (unsuitable)
    """
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(aoi)
    prot_mask = wdpa.reduceToImage(properties=["STATUS_YR"], reducer=ee.Reducer.count()).gt(0)

    hansen = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")
    loss = hansen.select("loss").gt(0)

    wc = _worldcover_year_map(aoi, year)
    unsuitable = wc.eq(80).Or(wc.eq(50))  # water or built-up

    label = (
        ee.Image(1)
        .where(loss, 1)
        .where(prot_mask, 2)
        .where(unsuitable, 0)
        .rename("unet_label")
        .toInt()
    )
    return label.clip(aoi)

def _ndvi_from_rgbnir(rgbnir: ee.Image) -> ee.Image:
    b8 = rgbnir.select("B8")
    b4 = rgbnir.select("B4")
    return b8.subtract(b4).divide(b8.add(b4).max(1e-6)).rename("NDVI")

def generate_degradation_score(aoi: ee.Geometry, start: str, end: str, wc_year: int) -> ee.Image:
    """
    Heuristic multi-factor degradation score in [0..100]:
      - Higher if: low NDVI, Hansen loss=1, WorldCover built-up/cropland/barren
      - Lower if: high NDVI, natural cover
    """
    rgbnir = _s2_rgbnir(aoi, start, end)
    ndvi = _ndvi_from_rgbnir(rgbnir)

    hansen = ee.Image("UMD/hansen/global_forest_change_2022_v1_10")
    loss = hansen.select("loss").gt(0)  # 0/1

    wc = _worldcover_year_map(aoi, wc_year)
    built = wc.eq(50)
    cropland = wc.eq(40)
    barren = wc.eq(95)  # barren/sparse

    ndvi_norm = ndvi.add(1).divide(2).clamp(0, 1)
    ndvi_degrade = ee.Image(1).subtract(ndvi_norm)

    score = (
        ndvi_degrade.multiply(40)    # up to 40
        .add(loss.multiply(30))      # +30 if loss
        .add(built.multiply(20))     # +20 built-up
        .add(cropland.multiply(10))  # +10 cropland
        .add(barren.multiply(20))    # +20 barren
    ).rename("degradation_score")

    return score.clamp(0, 100).toFloat()

def classify_degradation(score_img: ee.Image) -> ee.Image:
    """
    Map continuous score to classes:
      0: natural/intact   (score < 33)
      1: moderate         (33 ≤ score < 66)
      2: degraded         (score ≥ 66)
    """
    cls = (
        ee.Image(0)
        .where(score_img.gte(33).And(score_img.lt(66)), 1)
        .where(score_img.gte(66), 2)
        .rename("resnet_label")
        .toInt()
    )
    return cls

# ------------------------------
# TILING / EXPORT
# ------------------------------
def subdivide_aoi(aoi: ee.Geometry, shards: int) -> ee.FeatureCollection:
    """Split AOI bounding rectangle into ~shards tiles (rows x cols grid)."""
    cols = int(math.ceil(math.sqrt(shards)))
    rows = int(math.ceil(shards / cols))

    bounds = ee.Geometry(aoi.bounds())
    coords = ee.List(bounds.coordinates().get(0))
    xmin = ee.Number(ee.List(coords.get(0)).get(0))
    ymin = ee.Number(ee.List(coords.get(0)).get(1))
    xmax = ee.Number(ee.List(coords.get(2)).get(0))
    ymax = ee.Number(ee.List(coords.get(2)).get(1))

    dx = xmax.subtract(xmin).divide(cols)
    dy = ymax.subtract(ymin).divide(rows)

    tiles: List[ee.Feature] = []
    for r in range(rows):
        for c in range(cols):
            x0 = xmin.add(dx.multiply(c))
            y0 = ymin.add(dy.multiply(r))
            x1 = x0.add(dx)
            y1 = y0.add(dy)
            geom = ee.Geometry.Rectangle(ee.List([x0, y0, x1, y1]), proj=None, geodesic=False)
            tiles.append(ee.Feature(geom, {"row": r, "col": c}))

    return ee.FeatureCollection(tiles)

def export_master_patches_to_drive(
    aoi: ee.Geometry,
    final_stack: ee.Image,
    *,
    description_prefix: str,
    drive_folder: str,
    shards: int,
    patch_size: int,
    scale: int,
) -> List[ee.batch.Task]:
    """
    Export TFRecord patches (one task per shard). Band order is fixed.
    """
    ordered_bands = [
        "B4", "B3", "B2", "B8",
        "Slope", "Precip", "SoilPH",
        "resnet_label", "unet_label",
    ]
    export_img = final_stack.select(ordered_bands)

    tiles = subdivide_aoi(aoi, shards)
    tile_list = tiles.toList(tiles.size())

    tasks: List[ee.batch.Task] = []
    for i in range(shards):
        feature = ee.Feature(tile_list.get(i))
        geom = ee.Geometry(feature.geometry())
        fn_prefix = f"{description_prefix}_shard_{i:03d}"

        task = ee.batch.Export.image.toDrive(
            image=export_img,
            description=fn_prefix,
            folder=drive_folder,
            fileNamePrefix=fn_prefix,
            region=geom,
            scale=scale,
            fileFormat="TFRecord",
            formatOptions={
                "patchDimensions": [patch_size, patch_size],
                "compressed": True,
            },
            maxPixels=1_000_000_000_000,
        )
        task.start()
        tasks.append(task)

    return tasks

# ------------------------------
# CLI
# ------------------------------
def parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Export unified master TFRecord patches (features + both labels) from GEE."
    )

    aoi_grp = ap.add_mutually_exclusive_group(required=True)
    aoi_grp.add_argument("--state", type=str, help='State/UT name (e.g., "Uttarakhand")')
    aoi_grp.add_argument("--pan-india", action="store_true", help="Use entire India as AOI")
    aoi_grp.add_argument("--catalog-50", action="store_true", help="Use curated 50-region catalog")

    ap.add_argument("--drive-folder", required=True, help="Google Drive folder to write TFRecords")
    ap.add_argument("--project", default=None, help="EE project id (optional if set via CLI)")
    ap.add_argument("--shards", type=int, default=20, help="Shards (tiles) for single AOI modes")
    ap.add_argument("--shards-per-region", type=int, default=8, help="Shards per region for --catalog-50")
    ap.add_argument("--patch-size", type=int, default=256, help="Patch dimensions (pixels)")
    ap.add_argument("--scale", type=int, default=10, help="Export scale (m)")
    ap.add_argument("--s2-start", default="2021-01-01", help="Sentinel-2 start date (YYYY-MM-DD)")
    ap.add_argument("--s2-end", default="2021-12-31", help="Sentinel-2 end date (YYYY-MM-DD)")
    ap.add_argument("--label-year", "--year", dest="label_year", type=int, default=2021,
                    help="WorldCover year for masks (alias: --year)")

    # Kept for API symmetry; unused by patch exporter
    ap.add_argument("--n-points", type=int, default=20000, help="(Unused in patch mode)")
    return ap

# ------------------------------
# MAIN
# ------------------------------
def main() -> None:
    args = parse_args().parse_args()
    ee_init(args.project)

    # Curated catalog of ~50 regions
    if args.catalog_50:
        catalog = get_catalog_50()
        logging.info(f"Submitting master exports for curated catalog: {len(catalog)} regions")
        for idx, entry in enumerate(catalog):
            name = slugify(entry["name"])
            aoi = entry["geom"]
            logging.info(f"[{idx + 1:02d}/{len(catalog)}] Region: {name}")

            features = build_master_feature_stack(aoi, args.s2_start, args.s2_end, args.scale)

            # Labels
            degr_score = generate_degradation_score(aoi, args.s2_start, args.s2_end, args.label_year)
            resnet_label = classify_degradation(degr_score)          # resnet_label
            unet_label = build_unet_label_mask(aoi, year=args.label_year)  # unet_label

            final_stack = features.addBands([resnet_label, unet_label])

            desc_prefix = f"master_{name}_{args.s2_start[:4]}"
            tasks = export_master_patches_to_drive(
                aoi=aoi,
                final_stack=final_stack,
                description_prefix=desc_prefix,
                drive_folder=args.drive_folder,
                shards=args.shards_per_region,
                patch_size=args.patch_size,
                scale=args.scale,
            )
            for t in tasks:
                print("-", getattr(t, "id", "<no-id>"), desc_prefix)
        print("✅ All catalog-region export tasks submitted.")
        return

    # Pan-India
    if args.pan_india:
        aoi = get_india_geometry()
        name = "pan_india"

        features = build_master_feature_stack(aoi, args.s2_start, args.s2_end, args.scale)
        degr_score = generate_degradation_score(aoi, args.s2_start, args.s2_end, args.label_year)
        resnet_label = classify_degradation(degr_score)
        unet_label = build_unet_label_mask(aoi, year=args.label_year)

        final_stack = features.addBands([resnet_label, unet_label])

        desc_prefix = f"master_{name}_{args.s2_start[:4]}"
        tasks = export_master_patches_to_drive(
            aoi=aoi,
            final_stack=final_stack,
            description_prefix=desc_prefix,
            drive_folder=args.drive_folder,
            shards=args.shards,
            patch_size=args.patch_size,
            scale=args.scale,
        )
        print(f"Started {len(tasks)} export task(s) for India.")
        for t in tasks:
            print("-", getattr(t, "id", "<no-id>"), desc_prefix)
        return

    # Single state/UT
    aoi = state_geometry(args.state)  # type: ignore[arg-type]
    name = slugify(args.state)        # type: ignore[arg-type]

    features = build_master_feature_stack(aoi, args.s2_start, args.s2_end, args.scale)
    degr_score = generate_degradation_score(aoi, args.s2_start, args.s2_end, args.label_year)
    resnet_label = classify_degradation(degr_score)
    unet_label = build_unet_label_mask(aoi, year=args.label_year)

    final_stack = features.addBands([resnet_label, unet_label])

    desc_prefix = f"master_{name}_{args.s2_start[:4]}"
    tasks = export_master_patches_to_drive(
        aoi=aoi,
        final_stack=final_stack,
        description_prefix=desc_prefix,
        drive_folder=args.drive_folder,
        shards=args.shards,
        patch_size=args.patch_size,
        scale=args.scale,
    )
    print(f"Started {len(tasks)} export task(s) for {args.state}.")
    for t in tasks:
        print("-", getattr(t, "id", "<no-id>"), desc_prefix)

if __name__ == "__main__":
    main()
