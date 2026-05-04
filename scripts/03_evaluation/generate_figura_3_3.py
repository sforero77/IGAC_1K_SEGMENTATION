"""
Genera la Figura 3.3 — Esquema de partición espacial por grid de 1 km × 1 km.

Layout: 2x2 paneles (uno por municipio) con la cuadrícula coloreada por split
(Train=azul, Val=naranja, Test=verde). Anota porcentaje de área por conjunto y
número total de parches resultantes.

Entradas:
  - data/processed/spatial_splits/all_zones.gpkg
  - data/processed/spatial_splits/{train,val,test}_zones.gpkg
  - data/processed/rasters_etiquetas/{ciudad}_labels.tif (para extents)
  - data/processed/patches/patches_index.csv

Salida:
  - results/figures/figura_3_3_spatial_split.png
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from shapely.geometry import box

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
SPLITS_DIR = ROOT / "data" / "processed" / "spatial_splits"
RASTERS_DIR = ROOT / "data" / "processed" / "rasters_etiquetas"
PATCHES_INDEX = ROOT / "data" / "processed" / "patches" / "patches_index.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CIUDADES = ["madrid", "mosquera", "tabio", "tocancipa"]
CIUDAD_TITULOS = {
    "madrid": "Madrid",
    "mosquera": "Mosquera",
    "tabio": "Tabio",
    "tocancipa": "Tocancipá",
}

COLORS = {"train": "#2E86C1", "val": "#E67E22", "test": "#27AE60"}
LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def load_grid():
    gdf = gpd.read_file(SPLITS_DIR / "all_zones.gpkg")
    gdf = gdf[gdf["is_valid"]].copy()
    return gdf


def get_raster_extent(ciudad):
    path = RASTERS_DIR / f"{ciudad}_labels.tif"
    with rasterio.open(path) as src:
        b = src.bounds
        crs = src.crs
    return box(b.left, b.bottom, b.right, b.top), crs


def cells_for_ciudad(grid, ciudad):
    return grid[grid["ciudad"].fillna("").str.contains(ciudad, case=False, na=False)].copy()


def patches_summary():
    df = pd.read_csv(PATCHES_INDEX)
    total = df.groupby("split").size()
    per_ciudad = df.groupby(["ciudad", "split"]).size().unstack(fill_value=0)
    return total, per_ciudad


def area_summary(grid):
    grid = grid.copy()
    grid["area_km2"] = grid.geometry.area / 1e6
    by_split = grid.groupby("split")["area_km2"].sum()
    pct = by_split / by_split.sum() * 100
    return by_split, pct


def plot_inset(ax, ciudad, grid):
    """Dibuja un municipio (extent + grid coloreado por split) en el ax dado."""
    extent_poly, crs = get_raster_extent(ciudad)
    extent_gdf = gpd.GeoDataFrame(geometry=[extent_poly], crs=crs)

    cells = cells_for_ciudad(grid, ciudad)

    extent_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=3)

    for split, color in COLORS.items():
        sub = cells[cells["split"] == split]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, facecolor=color, edgecolor="white",
                 linewidth=0.6, alpha=0.9, zorder=2)

    minx, miny, maxx, maxy = extent_gdf.total_bounds
    pad = 0.08 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    n_train = (cells["split"] == "train").sum()
    n_val = (cells["split"] == "val").sum()
    n_test = (cells["split"] == "test").sum()
    n_total = n_train + n_val + n_test

    ax.set_title(
        f"{CIUDAD_TITULOS[ciudad]}\n"
        f"{n_total} celdas · {n_train}/{n_val}/{n_test}  (Tr/Va/Te)",
        fontsize=10, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)


def plot_overview(ax, grid):
    """Mapa global con los 4 municipios en sus coordenadas reales."""
    extents = []
    for ciudad in CIUDADES:
        poly, crs = get_raster_extent(ciudad)
        extents.append({"ciudad": ciudad, "geometry": poly})
    extent_gdf = gpd.GeoDataFrame(extents, crs=crs)

    extent_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=4)

    for split, color in COLORS.items():
        sub = grid[grid["split"] == split]
        if len(sub) == 0:
            continue
        sub.plot(ax=ax, facecolor=color, edgecolor="white",
                 linewidth=0.4, alpha=0.9, zorder=2)

    for _, row in extent_gdf.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        miny = row.geometry.bounds[1]
        ax.annotate(
            CIUDAD_TITULOS[row["ciudad"]],
            xy=(cx, miny),
            xytext=(cx, miny - 1500),
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#666666", alpha=0.9),
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("Este (m) · EPSG:9377", fontsize=10)
    ax.set_ylabel("Norte (m) · EPSG:9377", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    ax.set_title(
        "Disposición espacial real de los 4 municipios — "
        "cada celda = 1 km² asignada a un único conjunto",
        fontsize=11, fontweight="bold", pad=10,
    )

    # Barra de escala simple (5 km)
    minx, miny, maxx, maxy = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
    bar_len = 5000
    bar_x0 = minx + 0.04 * (maxx - minx)
    bar_y0 = miny + 0.05 * (maxy - miny)
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0],
            color="black", linewidth=3, zorder=6)
    ax.text(bar_x0 + bar_len / 2, bar_y0 + 0.012 * (maxy - miny),
            "5 km", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Flecha norte
    arrow_x = maxx - 0.06 * (maxx - minx)
    arrow_y = miny + 0.07 * (maxy - miny)
    ax.annotate("N", xy=(arrow_x, arrow_y + 0.04 * (maxy - miny)),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2),
                ha="center", va="center", fontsize=12, fontweight="bold")


def main():
    grid = load_grid()
    total_patches, _ = patches_summary()
    by_split_area, pct_area = area_summary(grid)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[3, 1.3],
        hspace=0.40, wspace=0.30,
        left=0.06, right=0.97, top=0.88, bottom=0.10,
    )

    ax_main = fig.add_subplot(gs[0, :])
    plot_overview(ax_main, grid)

    for i, ciudad in enumerate(CIUDADES):
        ax_inset = fig.add_subplot(gs[1, i])
        plot_inset(ax_inset, ciudad, grid)

    legend_elems = [
        mpatches.Patch(
            facecolor=COLORS["train"], edgecolor="white",
            label=f"Train — {pct_area.get('train', 0):.1f}% del área · "
                  f"{int(total_patches.get('train', 0)):,} parches "
                  f"({int((grid['split']=='train').sum())} celdas)"
        ),
        mpatches.Patch(
            facecolor=COLORS["val"], edgecolor="white",
            label=f"Validation — {pct_area.get('val', 0):.1f}% del área · "
                  f"{int(total_patches.get('val', 0)):,} parches "
                  f"({int((grid['split']=='val').sum())} celdas)"
        ),
        mpatches.Patch(
            facecolor=COLORS["test"], edgecolor="white",
            label=f"Test — {pct_area.get('test', 0):.1f}% del área · "
                  f"{int(total_patches.get('test', 0)):,} parches "
                  f"({int((grid['split']=='test').sum())} celdas)"
        ),
        Line2D([0], [0], color="black", lw=1.5,
               label="Extent del municipio"),
    ]

    fig.legend(handles=legend_elems, loc="lower center",
               ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Figura 3.3 — Partición espacial por grid de 1 km × 1 km\n"
        "Asignación por cuadrículas completas (no por parches) para garantizar "
        "independencia espacial entre Train, Validation y Test",
        fontsize=13, fontweight="bold", y=0.965,
    )

    n_total_celdas = len(grid)
    n_total_patches = int(total_patches.sum())
    fig.text(
        0.5, 0.045,
        f"Total: {n_total_celdas} celdas válidas (1 km²) · "
        f"{n_total_patches:,} parches de 256×256 px · "
        f"Estratificación por municipio · Semilla = 42",
        ha="center", fontsize=10, style="italic", color="#333333",
    )

    out_path = OUT_DIR / "figura_3_3_spatial_split.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"OK figura guardada: {out_path}")
    print(f"  Celdas válidas: {n_total_celdas}")
    print(f"  Área por split (km²): "
          f"train={by_split_area.get('train', 0):.1f}, "
          f"val={by_split_area.get('val', 0):.1f}, "
          f"test={by_split_area.get('test', 0):.1f}")
    print(f"  Parches por split: "
          f"train={int(total_patches.get('train', 0))}, "
          f"val={int(total_patches.get('val', 0))}, "
          f"test={int(total_patches.get('test', 0))}")


if __name__ == "__main__":
    main()
