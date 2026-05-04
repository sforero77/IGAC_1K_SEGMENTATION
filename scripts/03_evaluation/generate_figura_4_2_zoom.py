"""
Genera la Figura 4.2b — Vectores generados — Tocancipá CON ZOOMS DE DETALLE.

Layout:
  - Mapa global (izquierda, grande) con marcas de las zonas de zoom
  - 3 paneles de detalle (derecha, apilados) a 1 km × 1 km cada uno
      Z1: Cabecera urbana (mayor densidad de Construcción)
      Z2: Zona industrial
      Z3: Mancha forestal continua
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
ORTO = ROOT / "data" / "raw" / "ortofotos" / "tocancipa" / "Orto10_25817_20231116.tif"
GPKG_CONS = ROOT / "results" / "vectores" / "construccion_tocancipa.gpkg"
GPKG_BOSQ = ROOT / "results" / "vectores" / "bosque_tocancipa.gpkg"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_GLOBAL = 30      # 3 m/px para mapa global
ZOOM_PIXELSIZE_M = 0.5      # 0.5 m/px en zooms (mucho detalle)
ZOOM_HALF_M = 500           # ventana 1 km × 1 km

ZOOMS = [
    {"id": "Z1", "label": "Cabecera urbana", "cx": 4_895_991, "cy": 2_111_242},
    {"id": "Z2", "label": "Zona industrial", "cx": 4_897_991, "cy": 2_104_742},
    {"id": "Z3", "label": "Mancha forestal", "cx": 4_899_991, "cy": 2_104_242},
]

COLOR_CONS = "#EF1A2D"
COLOR_BOSQ = "#16A34A"


def stretch_rgb(rgb):
    out = np.zeros_like(rgb, dtype=np.float32)
    rgb = rgb.astype(np.float32)
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


def read_global_orthophoto(factor):
    with rasterio.open(ORTO) as src:
        new_h = src.height // factor
        new_w = src.width // factor
        rgb = src.read(indexes=[1, 2, 3], out_shape=(3, new_h, new_w),
                       resampling=Resampling.average)
        bounds = src.bounds
    return stretch_rgb(np.transpose(rgb, (1, 2, 0))), bounds


def read_zoom_orthophoto(cx, cy, half_m, target_pixel_m):
    bx = (cx - half_m, cy - half_m, cx + half_m, cy + half_m)
    with rasterio.open(ORTO) as src:
        win = from_bounds(*bx, transform=src.transform)
        out_h = int(2 * half_m / target_pixel_m)
        out_w = out_h
        rgb = src.read(indexes=[1, 2, 3], window=win,
                       out_shape=(3, out_h, out_w),
                       resampling=Resampling.cubic)
    return stretch_rgb(np.transpose(rgb, (1, 2, 0))), bx


def main():
    print("Cargando vectores...")
    cons = gpd.read_file(GPKG_CONS)
    bosq = gpd.read_file(GPKG_BOSQ)
    print(f"  Construcción: {len(cons):,} polígonos · {cons.geometry.area.sum() / 1e4:.1f} ha")
    print(f"  Bosque:       {len(bosq):,} polígonos · {bosq.geometry.area.sum() / 1e4:.1f} ha")

    print("Leyendo ortofotografía global...")
    rgb_g, bounds_g = read_global_orthophoto(DOWNSAMPLE_GLOBAL)

    print("Leyendo zooms...")
    zooms_data = []
    for z in ZOOMS:
        rgb, bx = read_zoom_orthophoto(z["cx"], z["cy"], ZOOM_HALF_M, ZOOM_PIXELSIZE_M)
        clip_geom = box(*bx)
        c_clip = cons[cons.geometry.intersects(clip_geom)].copy()
        b_clip = bosq[bosq.geometry.intersects(clip_geom)].copy()
        c_clip["geometry"] = c_clip.geometry.intersection(clip_geom)
        b_clip["geometry"] = b_clip.geometry.intersection(clip_geom)
        zooms_data.append({"meta": z, "rgb": rgb, "bx": bx, "cons": c_clip, "bosq": b_clip})
        print(f"  {z['id']} {z['label']}: {len(c_clip)} cons / {len(b_clip)} bosq")

    # ============== Plot ==============
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], wspace=0.10, hspace=0.18,
                          left=0.04, right=0.98, top=0.93, bottom=0.05)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_zooms = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    extent_g = (bounds_g.left, bounds_g.right, bounds_g.bottom, bounds_g.top)
    ax_main.imshow(rgb_g, extent=extent_g, origin="upper", interpolation="bilinear")
    bosq.plot(ax=ax_main, facecolor=COLOR_BOSQ, edgecolor="#064E1A",
              linewidth=0.2, alpha=0.65, zorder=2)
    cons.plot(ax=ax_main, facecolor=COLOR_CONS, edgecolor="#660A12",
              linewidth=0.2, alpha=0.78, zorder=3)
    ax_main.set_xlim(bounds_g.left, bounds_g.right)
    ax_main.set_ylim(bounds_g.bottom, bounds_g.top)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("Este (m) · EPSG:9377", fontsize=11)
    ax_main.set_ylabel("Norte (m) · EPSG:9377", fontsize=11)
    ax_main.tick_params(axis="both", labelsize=9)
    ax_main.set_title("Mapa global · Tocancipá", fontsize=12, fontweight="bold", pad=6)

    # Marcas de zooms en el mapa global
    for z in ZOOMS:
        rect = plt.Rectangle(
            (z["cx"] - ZOOM_HALF_M, z["cy"] - ZOOM_HALF_M),
            2 * ZOOM_HALF_M, 2 * ZOOM_HALF_M,
            fill=False, edgecolor="#FFD700", linewidth=2.5, zorder=10,
        )
        ax_main.add_patch(rect)
        ax_main.annotate(
            z["id"],
            xy=(z["cx"], z["cy"] + ZOOM_HALF_M),
            xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#000",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="#FFD700",
                      edgecolor="black", linewidth=1.5),
            zorder=11,
        )

    # Barra de escala global
    minx, miny = bounds_g.left, bounds_g.bottom
    maxx, maxy = bounds_g.right, bounds_g.top
    width = maxx - minx
    height = maxy - miny
    bar_len = 1000
    bar_x0 = minx + 0.04 * width
    bar_y0 = miny + 0.04 * height
    bar_h = 0.008 * height
    ax_main.add_patch(plt.Rectangle((bar_x0, bar_y0), bar_len / 2, bar_h,
                                     facecolor="black", edgecolor="black", zorder=10))
    ax_main.add_patch(plt.Rectangle((bar_x0 + bar_len / 2, bar_y0), bar_len / 2, bar_h,
                                     facecolor="white", edgecolor="black", zorder=10))
    for tick_x, tick_l in [(bar_x0, "0"), (bar_x0 + bar_len / 2, "0.5"),
                             (bar_x0 + bar_len, "1 km")]:
        ax_main.text(tick_x, bar_y0 - 0.014 * height, tick_l, fontsize=9,
                      ha="center", va="top", fontweight="bold", zorder=10)

    # Rosa de los vientos
    arrow_x = maxx - 0.07 * width
    arrow_y = miny + 0.05 * height
    arrow_len = 0.05 * height
    ax_main.add_patch(plt.Rectangle(
        (arrow_x - 0.04 * width, arrow_y - 0.012 * height),
        0.08 * width, arrow_len + 0.05 * height,
        facecolor="white", edgecolor="black", linewidth=1, alpha=0.9, zorder=9,
    ))
    ax_main.annotate(
        "", xy=(arrow_x, arrow_y + arrow_len), xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle="-|>,head_length=0.8,head_width=0.5",
                        color="black", lw=2.5), zorder=11,
    )
    ax_main.text(arrow_x, arrow_y + arrow_len + 0.004 * height, "N",
                 ha="center", va="bottom", fontsize=14, fontweight="bold", zorder=11)

    legend_elems = [
        mpatches.Patch(facecolor=COLOR_CONS, edgecolor="#660A12", alpha=0.78,
                       label=f"Construcción — {len(cons):,} polígonos · "
                             f"{cons.geometry.area.sum() / 1e4:.1f} ha"),
        mpatches.Patch(facecolor=COLOR_BOSQ, edgecolor="#064E1A", alpha=0.65,
                       label=f"Bosque — {len(bosq):,} polígonos · "
                             f"{bosq.geometry.area.sum() / 1e4:.1f} ha"),
        mpatches.Patch(facecolor="none", edgecolor="#FFD700", linewidth=2.5,
                       label="Ventana de zoom (1 km × 1 km)"),
    ]
    leg = ax_main.legend(
        handles=legend_elems, loc="upper left", fontsize=10,
        framealpha=0.92, edgecolor="black",
        title="Capas vectoriales (DeepLabv3+ → RDP 0.5 m)",
        title_fontsize=10,
    )
    leg.get_title().set_fontweight("bold")

    # ============ Zooms ============
    for ax_z, zd in zip(ax_zooms, zooms_data):
        z = zd["meta"]
        bx = zd["bx"]
        ax_z.imshow(zd["rgb"], extent=(bx[0], bx[2], bx[1], bx[3]),
                    origin="upper", interpolation="bilinear")
        if len(zd["bosq"]):
            zd["bosq"].plot(ax=ax_z, facecolor=COLOR_BOSQ, edgecolor="#064E1A",
                              linewidth=0.4, alpha=0.55, zorder=2)
        if len(zd["cons"]):
            zd["cons"].plot(ax=ax_z, facecolor=COLOR_CONS, edgecolor="#660A12",
                              linewidth=0.4, alpha=0.65, zorder=3)
        ax_z.set_xlim(bx[0], bx[2])
        ax_z.set_ylim(bx[1], bx[3])
        ax_z.set_aspect("equal")
        ax_z.tick_params(axis="both", labelsize=7)
        n_c = len(zd["cons"])
        n_b = len(zd["bosq"])
        ax_z.set_title(
            f"{z['id']} — {z['label']}  "
            f"({n_c} cons. / {n_b} bosq.)",
            fontsize=11, fontweight="bold",
        )
        # marco amarillo
        for spine in ax_z.spines.values():
            spine.set_edgecolor("#FFD700")
            spine.set_linewidth(2.5)
        # mini barra de escala (200 m) en cada zoom
        sx0 = bx[0] + 0.06 * (bx[2] - bx[0])
        sy0 = bx[1] + 0.06 * (bx[3] - bx[1])
        sl = 200
        sh = 0.018 * (bx[3] - bx[1])
        ax_z.add_patch(plt.Rectangle((sx0, sy0), sl, sh,
                                      facecolor="black", edgecolor="black", zorder=10))
        ax_z.text(sx0 + sl / 2, sy0 + sh + 0.01 * (bx[3] - bx[1]),
                  "200 m", ha="center", va="bottom", fontsize=8,
                  fontweight="bold", color="white",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                            edgecolor="none"), zorder=11)

    fig.suptitle(
        "Figura 4.2 — Vectores generados · Tocancipá (vista global + zooms de detalle)\n"
        "Construcción y Bosque sobre ortofotografía RGB IGAC · DeepLabv3+ + RDP 0.5 m",
        fontsize=14, fontweight="bold", y=0.985,
    )

    out = OUT_DIR / "figura_4_2_vectores_tocancipa_zoom.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
