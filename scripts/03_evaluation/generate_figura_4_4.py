"""
Genera la Figura 4.4 — Comparación metodológica · Mosquera.

Tres paneles horizontales sobre la misma ventana 1.5 km × 1.5 km:
  (a) Ortofotografía RGB
  (b) Máscara multiclase del DeepLabv3+ (paleta por clase)
  (c) Polígonos finales (Construcción + Bosque) sobre la ortofotografía
Cada panel marca una caja con la zona de detalle ampliado (panel inferior).
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
ORTO = ROOT / "data" / "raw" / "ortofotos" / "mosquera" / "Orto10_25473000_20241211.tif"
INFER = ROOT / "results" / "inferencia_arcgis" / "inferencia_mosquera.tif"
GPKG_CONS = ROOT / "results" / "vectores" / "construccion_mosquera.gpkg"
GPKG_BOSQ = ROOT / "results" / "vectores" / "bosque_mosquera.gpkg"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Centro y tamaño de la ventana principal y de la ventana de detalle
WIN_CX, WIN_CY = 4_863_958, 2_079_940     # zona urbana densa de Mosquera
WIN_HALF = 750                            # 1.5 km × 1.5 km
WIN_PIXEL_M = 0.4                         # ~3700 px de salida

DETAIL_CX, DETAIL_CY = 4_864_120, 2_080_080
DETAIL_HALF = 150                         # 300 m × 300 m
DETAIL_PIXEL_M = 0.15

CLASS_NAMES = [
    "Background", "Construcción", "Bosque", "Depósito Agua",
    "Zona Verde", "Zona Dura", "Vía", "Cerca", "Cabecera Municipal",
]
CLASS_COLORS = [
    (1, 1, 1, 0),         # 0 background transparente
    "#EF1A2D",             # 1 Construcción
    "#16A34A",             # 2 Bosque
    "#1F77B4",             # 3 Depósito Agua
    "#7CFC00",             # 4 Zona Verde
    "#A0A0A0",             # 5 Zona Dura
    "#404040",             # 6 Vía
    "#8B4513",             # 7 Cerca
    "#F4D03F",             # 8 Cabecera Municipal
]

COLOR_CONS = "#EF1A2D"
COLOR_BOSQ = "#16A34A"


def stretch_rgb(rgb):
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb)
    for b in range(3):
        band = rgb[..., b]
        valid = band[band > 0]
        if len(valid) == 0:
            out[..., b] = band / 255
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        out[..., b] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def read_window_rgb(path, cx, cy, half_m, pixel_m):
    bx = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    with rasterio.open(path) as src:
        win = from_bounds(*bx, transform=src.transform)
        out_h = int(2 * half_m / pixel_m)
        rgb = src.read(indexes=[1, 2, 3], window=win,
                       out_shape=(3, out_h, out_h),
                       resampling=Resampling.cubic)
    return stretch_rgb(np.transpose(rgb, (1, 2, 0))), bx


def read_window_class(path, cx, cy, half_m, pixel_m):
    bx = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    with rasterio.open(path) as src:
        win = from_bounds(*bx, transform=src.transform)
        out_h = int(2 * half_m / pixel_m)
        arr = src.read(1, window=win, out_shape=(out_h, out_h),
                       resampling=Resampling.nearest)
    return arr, bx


def add_scalebar(ax, bx, length_m, label):
    minx, maxx = bx[0], bx[2]
    miny, maxy = bx[1], bx[3]
    x0 = minx + 0.05 * (maxx - minx)
    y0 = miny + 0.05 * (maxy - miny)
    h = 0.018 * (maxy - miny)
    ax.add_patch(plt.Rectangle((x0, y0), length_m, h,
                                facecolor="black", edgecolor="black", zorder=10))
    ax.text(x0 + length_m / 2, y0 + h + 0.012 * (maxy - miny), label,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                      edgecolor="none"), zorder=11)


def add_north_arrow(ax, bx):
    minx, maxx = bx[0], bx[2]
    miny, maxy = bx[1], bx[3]
    width = maxx - minx
    height = maxy - miny
    arrow_x = maxx - 0.07 * width
    arrow_y = miny + 0.07 * height
    arrow_len = 0.06 * height
    ax.add_patch(plt.Rectangle(
        (arrow_x - 0.045 * width, arrow_y - 0.012 * height),
        0.09 * width, arrow_len + 0.05 * height,
        facecolor="white", edgecolor="black", linewidth=1, alpha=0.9, zorder=9,
    ))
    ax.annotate("", xy=(arrow_x, arrow_y + arrow_len),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.5",
                                color="black", lw=2.5), zorder=11)
    ax.text(arrow_x, arrow_y + arrow_len + 0.005 * height, "N",
            ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=11)


def main():
    print("Leyendo ventana principal...")
    rgb, bx_main = read_window_rgb(ORTO, WIN_CX, WIN_CY, WIN_HALF, WIN_PIXEL_M)
    cls, _ = read_window_class(INFER, WIN_CX, WIN_CY, WIN_HALF, WIN_PIXEL_M)
    print(f"  RGB: {rgb.shape}, clases: {cls.shape}, únicos: {sorted(np.unique(cls).tolist())}")

    print("Leyendo ventana de detalle...")
    rgb_d, bx_det = read_window_rgb(ORTO, DETAIL_CX, DETAIL_CY, DETAIL_HALF, DETAIL_PIXEL_M)
    cls_d, _ = read_window_class(INFER, DETAIL_CX, DETAIL_CY, DETAIL_HALF, DETAIL_PIXEL_M)

    print("Cargando vectores...")
    cons = gpd.read_file(GPKG_CONS)
    bosq = gpd.read_file(GPKG_BOSQ)
    clip_main = box(*bx_main)
    clip_det = box(*bx_det)
    cons_w = cons[cons.geometry.intersects(clip_main)].copy()
    bosq_w = bosq[bosq.geometry.intersects(clip_main)].copy()
    cons_w["geometry"] = cons_w.geometry.intersection(clip_main)
    bosq_w["geometry"] = bosq_w.geometry.intersection(clip_main)
    cons_d = cons[cons.geometry.intersects(clip_det)].copy()
    bosq_d = bosq[bosq.geometry.intersects(clip_det)].copy()
    cons_d["geometry"] = cons_d.geometry.intersection(clip_det)
    bosq_d["geometry"] = bosq_d.geometry.intersection(clip_det)
    print(f"  En ventana: {len(cons_w)} cons / {len(bosq_w)} bosq")
    print(f"  En detalle: {len(cons_d)} cons / {len(bosq_d)} bosq")

    cmap = ListedColormap(CLASS_COLORS)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.05],
                          hspace=0.18, wspace=0.06,
                          left=0.04, right=0.98, top=0.92, bottom=0.05)

    extent_main = (bx_main[0], bx_main[2], bx_main[1], bx_main[3])
    extent_det = (bx_det[0], bx_det[2], bx_det[1], bx_det[3])

    # (a) Ortofoto
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(rgb, extent=extent_main, origin="upper", interpolation="bilinear")
    ax_a.set_title("(a) Ortofotografía RGB",
                   fontsize=13, fontweight="bold", pad=6)
    ax_a.set_xlabel("Este (m) · EPSG:9377", fontsize=10)
    ax_a.set_ylabel("Norte (m) · EPSG:9377", fontsize=10)

    # (b) Predicción ráster multiclase
    ax_b = fig.add_subplot(gs[0, 1], sharex=ax_a, sharey=ax_a)
    ax_b.imshow(cls, extent=extent_main, origin="upper",
                cmap=cmap, vmin=0, vmax=8, interpolation="nearest")
    ax_b.set_title("(b) Predicción ráster · DeepLabv3+ (8 clases)",
                   fontsize=13, fontweight="bold", pad=6)
    ax_b.set_xlabel("Este (m) · EPSG:9377", fontsize=10)
    ax_b.tick_params(labelleft=False)

    # (c) Vectores finales sobre la ortofoto
    ax_c = fig.add_subplot(gs[0, 2], sharex=ax_a, sharey=ax_a)
    ax_c.imshow(rgb, extent=extent_main, origin="upper", interpolation="bilinear")
    if len(bosq_w):
        bosq_w.plot(ax=ax_c, facecolor=COLOR_BOSQ, edgecolor="#064E1A",
                    linewidth=0.5, alpha=0.65, zorder=2)
    if len(cons_w):
        cons_w.plot(ax=ax_c, facecolor=COLOR_CONS, edgecolor="#660A12",
                    linewidth=0.5, alpha=0.7, zorder=3)
    ax_c.set_title(f"(c) Polígonos finales tras pipeline vectorial\n"
                   f"({len(cons_w)} cons. · {len(bosq_w)} bosq. en la ventana)",
                   fontsize=12, fontweight="bold", pad=6)
    ax_c.set_xlabel("Este (m) · EPSG:9377", fontsize=10)
    ax_c.tick_params(labelleft=False)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(bx_main[0], bx_main[2])
        ax.set_ylim(bx_main[1], bx_main[3])
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=8)
        # Marca de la ventana de detalle
        rect = plt.Rectangle(
            (bx_det[0], bx_det[1]),
            bx_det[2] - bx_det[0], bx_det[3] - bx_det[1],
            fill=False, edgecolor="#FFD700", linewidth=2.5, zorder=15,
        )
        ax.add_patch(rect)

    add_scalebar(ax_a, bx_main, 200, "200 m")
    add_north_arrow(ax_a, bx_main)

    # Leyenda de clases para panel (b)
    class_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[i],
                       edgecolor="black", linewidth=0.4,
                       label=f"{i} · {CLASS_NAMES[i]}")
        for i in range(1, 9)
    ]
    ax_b.legend(handles=class_handles, loc="upper left",
                fontsize=8, framealpha=0.92, ncol=2,
                title="Clases", title_fontsize=9)

    # Leyenda de vectores para panel (c)
    vec_handles = [
        mpatches.Patch(facecolor=COLOR_CONS, edgecolor="#660A12", alpha=0.7,
                       label="Construcción"),
        mpatches.Patch(facecolor=COLOR_BOSQ, edgecolor="#064E1A", alpha=0.65,
                       label="Bosque"),
        mpatches.Patch(facecolor="none", edgecolor="#FFD700", linewidth=2.5,
                       label="Detalle ampliado (panel inferior)"),
    ]
    ax_c.legend(handles=vec_handles, loc="upper left",
                fontsize=8.5, framealpha=0.92)

    # ====== Fila de detalle (zoom 300 m × 300 m) ======
    ax_da = fig.add_subplot(gs[1, 0])
    ax_da.imshow(rgb_d, extent=extent_det, origin="upper", interpolation="bilinear")
    ax_da.set_title(f"Detalle (a) — ortofoto · {2 * DETAIL_HALF} m × {2 * DETAIL_HALF} m",
                    fontsize=11, fontweight="bold")

    ax_db = fig.add_subplot(gs[1, 1], sharex=ax_da, sharey=ax_da)
    ax_db.imshow(cls_d, extent=extent_det, origin="upper",
                 cmap=cmap, vmin=0, vmax=8, interpolation="nearest")
    ax_db.set_title("Detalle (b) — predicción ráster",
                    fontsize=11, fontweight="bold")
    ax_db.tick_params(labelleft=False)

    ax_dc = fig.add_subplot(gs[1, 2], sharex=ax_da, sharey=ax_da)
    ax_dc.imshow(rgb_d, extent=extent_det, origin="upper", interpolation="bilinear")
    if len(bosq_d):
        bosq_d.plot(ax=ax_dc, facecolor=COLOR_BOSQ, edgecolor="#064E1A",
                    linewidth=1.0, alpha=0.55, zorder=2)
    if len(cons_d):
        cons_d.plot(ax=ax_dc, facecolor=COLOR_CONS, edgecolor="#660A12",
                    linewidth=1.0, alpha=0.65, zorder=3)
    ax_dc.set_title(f"Detalle (c) — polígonos vectoriales\n"
                    f"({len(cons_d)} cons. · {len(bosq_d)} bosq.)",
                    fontsize=11, fontweight="bold")
    ax_dc.tick_params(labelleft=False)

    for ax in (ax_da, ax_db, ax_dc):
        ax.set_xlim(bx_det[0], bx_det[2])
        ax.set_ylim(bx_det[1], bx_det[3])
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFD700")
            spine.set_linewidth(2.0)

    add_scalebar(ax_da, bx_det, 50, "50 m")

    fig.suptitle(
        "Figura 4.4 — Comparación metodológica · Mosquera\n"
        "Predicción ráster vs. vectores finales del pipeline (DeepLabv3+ + RDP 0.5 m)",
        fontsize=14, fontweight="bold", y=0.985
    )

    out = OUT_DIR / "figura_4_4_comparacion_mosquera.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
