# scripts/make_default_aoi.py
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

ACRES = 2.0
M2_PER_ACRE = 4046.8564224
AREA_M2 = ACRES * M2_PER_ACRE          # ~8093.7 m²
PIX_M = 10                             # pretend Sentinel-2 like pixel size
SIDE_M = int((AREA_M2 ** 0.5) // PIX_M * PIX_M)  # ~90 m
N = max(8, SIDE_M // PIX_M)            # ~9 px

# A central India default (Bhopal-ish). You can change these.
DEFAULT_LAT = 23.2599
DEFAULT_LON = 77.4126

def make_default_aoi(out_tif: Path, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    # crude deg/px near equator (fine for a demo)
    deg_per_m = 1.0 / 111_320.0
    px_deg = PIX_M * deg_per_m

    # simple 3-band “RGB” gradient (demo only; not real satellite)
    arr = np.zeros((3, N, N), dtype=np.uint8)
    yy = np.linspace(0, 255, N, dtype=np.uint8)
    xx = np.linspace(0, 255, N, dtype=np.uint8)
    arr[0] = yy[:, None]           # R gradient
    arr[1] = xx[None, :]           # G gradient
    arr[2] = 128                   # constant B

    transform = from_origin(lon, lat, px_deg, px_deg)  # upper-left origin
    with rasterio.open(
        out_tif, "w",
        driver="GTiff", height=N, width=N, count=3,
        dtype=arr.dtype, crs="EPSG:4326",
        transform=transform, compress="lzw"
    ) as dst:
        dst.write(arr)

if __name__ == "__main__":
    make_default_aoi(Path("data/inference/default_2acre.tif"))
    print("✅ Wrote data/inference/default_2acre.tif")
