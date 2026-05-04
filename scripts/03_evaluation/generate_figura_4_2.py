"""
Genera la Figura 4.2 — Vectores generados — Municipio de Tocancipá
(Construcción + Bosque sobre ortofotografía RGB).

Fuentes:
  - data/raw/ortofotos/tocancipa/Orto10_25817_20231116.tif  (RGBA 0.1 m/px)
  - results/vectores/construccion_tocancipa.gpkg
  - results/vectores/bosque_tocancipa.gpkg
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
ORTO = ROOT / "data" / "raw" / "ortofotos" / "tocancipa" / "Orto10_25817_20231116.tif"
GPKG_CONS = ROOT / "results" / "vectores" / "construccion_tocancipa.gpkg"
GPKG_BOSQ = ROOT / "results" / "vectores" / "bosque_tocancipa.gpkg"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE = 30  # 0.1 m * 30 = 3 m/px de salida (suficiente para una figura grande)


def read_orthophoto_downsampled(path, factor):
    with rasterio.open(path) as src:
        new_h = src.height // factor
        new_w = src.width // factor
        rgb = src.read(
            indexes=[1, 2, 3],
            out_shape=(3, new_h, new_w),
            resampling=Resampling.average,
        )
        bounds = src.bounds
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    # Estiramiento percentil por banda para mejorar contraste
    out = np.zeros_like(rgb)
    for b in range(3):
        band = rgb[..., b]
        valid = band[band > 0]
        if len(valid) == 0:
            out[..., b] = band
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        stretched = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        out[..., b] = stretched
    return out, bounds


def main():
    print(f"Leyendo ortofotografía (downsample x{DOWNSAMPLE})...")
    rgb, bounds = read_orthophoto_downsampled(ORTO, DOWNSAMPLE)
    print(f"  shape salida: {rgb.shape} (~{(bounds.right - bounds.left) / rgb.shape[1]:.1f} m/px)")

    cons = gpd.read_file(GPKG_CONS)
    bosq = gpd.read_file(GPKG_BOSQ)
    print(f"  Construcción: {len(cons)} polígonos · {cons.geometry.area.sum() / 1e4:.1f} ha")
    print(f"  Bosque:       {len(bosq)} polígonos · {bosq.geometry.area.sum() / 1e4:.1f} ha")

    fig, ax = plt.subplots(figsize=(15, 14))

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear")

    bosq.plot(ax=ax, facecolor="#16A34A", edgecolor="#064E1A",
              linewidth=0.2, alpha=0.70, zorder=2)
    cons.plot(ax=ax, facecolor="#EF1A2D", edgecolor="#660A12",
              linewidth=0.2, alpha=0.80, zorder=3)

    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_aspect("equal")
    ax.set_xlabel("Este (m) · EPSG:9377", fontsize=11)
    ax.set_ylabel("Norte (m) · EPSG:9377", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
    width = maxx - minx
    height = maxy - miny

    bar_len = 1000  # 1 km
    bar_x0 = minx + 0.04 * width
    bar_y0 = miny + 0.05 * height
    bar_h = 0.008 * height
    ax.add_patch(plt.Rectangle((bar_x0, bar_y0), bar_len / 2, bar_h,
                                facecolor="black", edgecolor="black", zorder=10))
    ax.add_patch(plt.Rectangle((bar_x0 + bar_len / 2, bar_y0), bar_len / 2, bar_h,
                                facecolor="white", edgecolor="black", zorder=10))
    ax.text(bar_x0, bar_y0 - 0.018 * height, "0", fontsize=9,
            ha="center", va="top", fontweight="bold", zorder=10)
    ax.text(bar_x0 + bar_len / 2, bar_y0 - 0.018 * height, "0.5", fontsize=9,
            ha="center", va="top", fontweight="bold", zorder=10)
    ax.text(bar_x0 + bar_len, bar_y0 - 0.018 * height, "1 km", fontsize=9,
            ha="center", va="top", fontweight="bold", zorder=10)

    arrow_x = maxx - 0.07 * width
    arrow_y = miny + 0.06 * height
    arrow_len = 0.06 * height
    # Fondo blanco para la rosa
    ax.add_patch(plt.Rectangle(
        (arrow_x - 0.04 * width, arrow_y - 0.015 * height),
        0.08 * width, arrow_len + 0.06 * height,
        facecolor="white", edgecolor="black",
        linewidth=1, alpha=0.85, zorder=9,
    ))
    ax.annotate(
        "", xy=(arrow_x, arrow_y + arrow_len), xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.5",
                        color="black", lw=2.5), zorder=11,
    )
    ax.text(arrow_x, arrow_y + arrow_len + 0.005 * height, "N",
            ha="center", va="bottom", fontsize=14, fontweight="bold", zorder=11)

    legend_elems = [
        mpatches.Patch(facecolor="#E63946", edgecolor="#7A0F1A",
                       alpha=0.65,
                       label=f"Construcción — {len(cons):,} polígonos · "
                             f"{cons.geometry.area.sum() / 1e4:.1f} ha"),
        mpatches.Patch(facecolor="#1B7A33", edgecolor="#0F4D1F",
                       alpha=0.55,
                       label=f"Bosque — {len(bosq):,} polígonos · "
                             f"{bosq.geometry.area.sum() / 1e4:.1f} ha"),
    ]
    leg = ax.legend(handles=legend_elems, loc="upper left",
                    fontsize=10, framealpha=0.92, edgecolor="black",
                    title="Capas vectoriales (DeepLabv3+ → RDP 0.5 m)",
                    title_fontsize=10)
    leg.get_title().set_fontweight("bold")

    info = (
        "Fuente: ortofotografía RGB IGAC (10 cm/px)\n"
        "Sistema de referencia: MAGNA-SIRGAS · Origen Nacional (EPSG:9377)\n"
        "Escala referencial: 1:25 000  ·  Datum y proyección: Transversa de Mercator"
    )
    ax.text(0.985, 0.015, info, transform=ax.transAxes,
            fontsize=8.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9), zorder=10)

    fig.suptitle(
        "Figura 4.2 — Vectores generados · Municipio de Tocancipá\n"
        "Construcción y Bosque sobre ortofotografía RGB · DeepLabv3+ (ResNet-50)",
        fontsize=13, fontweight="bold", y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "figura_4_2_vectores_tocancipa.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
