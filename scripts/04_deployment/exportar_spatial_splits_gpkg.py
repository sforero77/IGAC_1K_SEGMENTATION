"""
Exporta los splits espaciales a un GPKG limpio (EPSG:9377 explícito) listo
para abrir en QGIS / ArcGIS Pro y verificar visualmente la partición.

Salidas:
  - results/figures/figura_3_3_spatial_split.gpkg
        capas: all_zones, train_zones, val_zones, test_zones, municipio_extents
"""

from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
SRC = ROOT / "data" / "processed" / "spatial_splits" / "all_zones.gpkg"
RASTERS = ROOT / "data" / "processed" / "rasters_etiquetas"
OUT = ROOT / "results" / "figures" / "figura_3_3_spatial_split.gpkg"
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

CIUDADES = ["madrid", "mosquera", "tabio", "tocancipa"]
TARGET_CRS = "EPSG:9377"

g = gpd.read_file(SRC)
g = g.set_crs(TARGET_CRS, allow_override=True)
g["area_km2"] = (g.geometry.area / 1e6).round(4)
g["lado_x_m"] = g.geometry.bounds.apply(lambda r: r["maxx"] - r["minx"], axis=1).round(2)
g["lado_y_m"] = g.geometry.bounds.apply(lambda r: r["maxy"] - r["miny"], axis=1).round(2)

g.to_file(OUT, driver="GPKG", layer="all_zones")

valid = g[g["is_valid"]].copy()
for split in ["train", "val", "test"]:
    sub = valid[valid["split"] == split]
    sub.to_file(OUT, driver="GPKG", layer=f"{split}_zones")

extents = []
for ciudad in CIUDADES:
    with rasterio.open(RASTERS / f"{ciudad}_labels.tif") as src:
        b = src.bounds
        extents.append({"ciudad": ciudad, "geometry": box(b.left, b.bottom, b.right, b.top)})
extent_gdf = gpd.GeoDataFrame(extents, crs=TARGET_CRS)
extent_gdf.to_file(OUT, driver="GPKG", layer="municipio_extents")

print(f"OK GPKG exportado: {OUT}")
print(f"  Capas: all_zones ({len(g)}), train_zones ({(valid['split']=='train').sum()}), "
      f"val_zones ({(valid['split']=='val').sum()}), test_zones ({(valid['split']=='test').sum()}), "
      f"municipio_extents ({len(extent_gdf)})")
print(f"  CRS: {TARGET_CRS}")
print(f"  Celdas exactas 1000x1000 m: "
      f"{((g['lado_x_m']==1000) & (g['lado_y_m']==1000)).sum()} / {len(g)}")
