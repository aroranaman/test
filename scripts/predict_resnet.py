#!/usr/bin/env python3
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import json

import numpy as np
import torch
from torchvision import models, transforms

import rasterio
from rasterio.enums import Resampling

from PIL import Image

# Optional (for district lookup + HTML map)
import geopandas as gpd
import folium
from shapely.geometry import box

__BUILD__ = "2025-09-01c"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"[predict_resnet] build={__BUILD__} (tile-write fix, CRS-safe district lookup)")

# ---------- Color map & ESA mapping ----------
CLASS_TO_ESA = {
    0: 10, 1: 20, 2: 30, 3: 40, 4: 50, 5: 60,
    6: 70, 7: 80, 8: 90, 9: 95, 10: 100, 11: 110,
}

COLOR_MAP = {
    10:(0,100,0), 20:(255,187,34), 30:(255,255,76), 40:(240,150,255),
    50:(255,0,0), 60:(200,200,200), 70:(180,220,250), 80:(0,0,255),
    90:(0,160,230), 95:(0,120,120), 100:(190,255,190), 110:(120,120,120),
}

# ---------- Model helpers ----------
def load_model(model_path: Path, num_classes: int, device: torch.device):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def make_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # HWC[0..255] -> CHW[0..1]
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

# ---------- Raster I/O ----------
def predict_raster(
    image_path: Path,
    out_tif: Path,
    model_path: Path,
    num_classes: int = 12,
    write_esa_codes: bool = False,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
    logging.info(f"Using device: {device}")

    model = load_model(model_path, num_classes, device)
    tfm = make_transform()

    with rasterio.open(image_path) as src:
        bands_to_read = [1,2,3] if src.count >= 3 else [1] * min(src.count, 1)
        meta = src.meta.copy()
        meta.update({
            "count": 1,
            "dtype": "uint16" if write_esa_codes else "uint8",
            "compress": "lzw",
        })
        out_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_tif, "w", **meta) as dst:
            for _, win in src.block_windows(1):
                arr = src.read(indexes=bands_to_read, window=win)  # (B,H,W)

                # Ensure 3 channels
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                elif arr.shape[0] > 3:
                    arr = arr[:3]

                hwc = np.transpose(arr, (1,2,0)).astype(np.uint8)  # (H,W,C)

                with torch.no_grad():
                    x = tfm(hwc).unsqueeze(0).to(device)    # (1,3,H,W)
                    logits = model(x)                        # (1,C)
                    pred_idx = int(torch.argmax(logits, dim=1).item())

                val = CLASS_TO_ESA.get(pred_idx, pred_idx) if write_esa_codes else pred_idx
                dtype = np.uint16 if write_esa_codes else np.uint8

                # ✅ always write a full tile (1, H, W)
                tile = np.full((1, win.height, win.width), val, dtype=dtype)
                dst.write(tile, window=win, indexes=1)

    return out_tif

def array_to_color_png(class_arr: np.ndarray, output_png: Path, use_esa: bool):
    h, w = class_arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if use_esa:
        for k, color in COLOR_MAP.items():
            m = (class_arr == k)
            if np.any(m): rgb[m] = color
    else:
        esa_arr = np.vectorize(CLASS_TO_ESA.get)(class_arr)
        for k, color in COLOR_MAP.items():
            m = (esa_arr == k)
            if np.any(m): rgb[m] = color
    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_png)

def save_html_map(png_path: Path, geotiff_path: Path, html_out: Path):
    with rasterio.open(geotiff_path) as src:
        b = src.bounds
        west, south, east, north = b.left, b.bottom, b.right, b.top
        center = [(south+north)/2.0, (west+east)/2.0]

    m = folium.Map(location=center, zoom_start=16, tiles="CartoDB positron")
    folium.Rectangle([[south, west],[north, east]], color="#3388ff", weight=2, fill=False).add_to(m)
    folium.raster_layers.ImageOverlay(
        name="Prediction",
        image=str(png_path),
        bounds=[[south, west],[north, east]],
        opacity=0.6,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(html_out))
    logging.info(f"✅ HTML map saved to: {html_out}")

def lookup_districts(geotiff_path: Path, districts_path: Path, name_field: str = "DISTRICT") -> list[str]:
    try:
        with rasterio.open(geotiff_path) as src:
            raster_crs = src.crs
            b = src.bounds
            aoi_poly = box(b.left, b.bottom, b.right, b.top)

        gdf = gpd.read_file(districts_path)
        if raster_crs is not None:
            gdf = gdf.to_crs(raster_crs)

        hit = gdf[gdf.intersects(aoi_poly)]
        names = sorted({str(n) for n in hit[name_field].values if n})
        return names
    except Exception as e:
        logging.warning(f"District lookup failed: {e}")
        return []

# ---------- Orchestrator ----------
def predict(
    image_path: Path,
    out_tif: Path,
    out_png: Path | None = None,
    out_html: Path | None = None,
    model_path: Path | None = None,
    num_classes: int = 12,
    write_esa_codes: bool = False,
    districts_path: Path | None = None,
    district_name_field: str = "DISTRICT",
):
    if model_path is None:
        raise ValueError("--model path is required")

    pred_tif = predict_raster(
        image_path=image_path,
        out_tif=out_tif,
        model_path=model_path,
        num_classes=num_classes,
        write_esa_codes=write_esa_codes,
    )
    logging.info(f"✅ Prediction GeoTIFF saved: {pred_tif}")

    if out_png is not None:
        with rasterio.open(pred_tif) as src:
            arr = src.read(1)
        array_to_color_png(arr, out_png, use_esa=write_esa_codes)
        logging.info(f"✅ Color PNG saved: {out_png}")

    if districts_path is not None:
        names = lookup_districts(pred_tif, districts_path, district_name_field)
        if names:
            logging.info("📍 District(s): " + ", ".join(names))
        else:
            logging.info("📍 District(s): (none found)")

    if out_html is not None and out_png is not None:
        save_html_map(out_png, pred_tif, out_html)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path, help="Input AOI GeoTIFF (RGB preferred)")
    p.add_argument("--out-tif", required=True, type=Path, help="Output classified GeoTIFF")
    p.add_argument("--out-png", type=Path, default=None, help="Optional color PNG")
    p.add_argument("--out-html", type=Path, default=None, help="Optional Folium HTML map")
    p.add_argument("--model", required=True, type=Path, help="ResNet weights .pth")
    p.add_argument("--num-classes", type=int, default=12)
    p.add_argument("--write-esa-codes", action="store_true", help="Map class IDs to ESA codes in GeoTIFF")
    p.add_argument("--districts", type=Path, default=None, help="Districts GeoJSON/GeoPackage")
    p.add_argument("--district-name-field", type=str, default="DISTRICT")
    return p.parse_args()

def main():
    args = parse_args()
    predict(
        image_path=args.image,
        out_tif=args.out_tif,
        out_png=args.out_png,
        out_html=args.out_html,
        model_path=args.model,
        num_classes=args.num_classes,
        write_esa_codes=args.write_esa_codes,
        districts_path=args.districts,
        district_name_field=args.district_name_field,
    )

if __name__ == "__main__":
    main()
