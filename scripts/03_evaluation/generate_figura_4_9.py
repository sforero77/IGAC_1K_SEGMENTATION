"""
Genera la Figura 4.9 — Antes y después de la simplificación RDP en un polígono
ejemplo de Construcción.

Procedimiento:
  1. Tomar la región de un polígono de Construcción del GPKG ya simplificado de
     Madrid (que sirve como localizador).
  2. Recortar la ventana correspondiente del raster de inferencia (resolución
     0.1 m/px).
  3. Vectorizar la clase 1 SIN simplificar → polígono crudo (efecto escalera).
  4. Aplicar simplify(tolerance=0.5, preserve_topology=True) → polígono RDP.
  5. Plotear comparación lado a lado con vértices marcados, conteo de vértices,
     área y % de reducción.

Fuente:
  - results/inferencia_arcgis/inferencia_madrid.tif       (clase 1 = Construcción)
  - results/vectores/construccion_madrid.gpkg              (localizador)
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, shape

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
RASTER = ROOT / "results" / "inferencia_arcgis" / "inferencia_madrid.tif"
GPKG = ROOT / "results" / "vectores" / "construccion_madrid.gpkg"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASE_CONSTRUCCION = 1
RDP_TOL = 0.5  # metros


def count_vertices(geom):
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords) - 1  # último coord = primero
    return sum(len(p.exterior.coords) - 1 for p in geom.geoms)


def vectorize_class(arr, transform, target_class):
    mask = (arr == target_class).astype(np.uint8)
    polys = []
    for geom_dict, val in shapes(mask, mask=mask.astype(bool), transform=transform):
        if val == 1:
            polys.append(shape(geom_dict))
    return polys


def aspect_ratio(geom):
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    sides = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        sides.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    long_side = max(sides)
    short_side = min(sides)
    return long_side / max(short_side, 1e-6)


def select_candidate(gpkg_path, n_top=30):
    """Selecciona polígono compacto, claramente ortogonal y de tamaño medio."""
    g = gpd.read_file(gpkg_path)
    g["area"] = g.geometry.area
    candidates = g[(g["area"] > 300) & (g["area"] < 1200)].copy()
    candidates["mbr_ratio"] = candidates.geometry.apply(
        lambda p: p.area / p.minimum_rotated_rectangle.area
    )
    candidates["aspect"] = candidates.geometry.apply(aspect_ratio)
    candidates = candidates[
        (candidates["mbr_ratio"] > 0.85) & (candidates["aspect"] < 2.0)
    ]
    candidates = candidates.sort_values("area", ascending=False).head(n_top)
    return candidates


def main():
    cands = select_candidate(GPKG)
    print(f"Candidatos: {len(cands)}")

    best_result = None
    for idx, row in cands.iterrows():
        bounds = row.geometry.bounds
        pad = 5.0
        bx = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
        with rasterio.open(RASTER) as src:
            window = from_bounds(*bx, transform=src.transform)
            arr = src.read(1, window=window)
            win_transform = src.window_transform(window)

        polys_raw = vectorize_class(arr, win_transform, CLASE_CONSTRUCCION)
        if not polys_raw:
            continue

        # Quedarse con el polígono más cercano al centroide de la candidata
        target_centroid = row.geometry.centroid
        polys_raw.sort(key=lambda p: p.centroid.distance(target_centroid))
        raw = polys_raw[0]
        if not raw.is_valid or raw.area < 200:
            continue

        n_raw = count_vertices(raw)
        if n_raw < 80 or n_raw > 800:
            continue

        simplified = raw.simplify(RDP_TOL, preserve_topology=True)
        n_simp = count_vertices(simplified)
        if n_simp == 0:
            continue

        reduction = (1 - n_simp / n_raw) * 100
        area_pres = (simplified.area / raw.area) * 100

        # Preferir polígonos con n_raw cercano a 200 y reducción ~60% (lo
        # descrito en el caption original).
        score = -abs(n_raw - 200) - abs(reduction - 60) * 2

        if best_result is None or score > best_result.get("score", -1e9):
            best_result = {
                "score": score,
                "raw": raw,
                "simplified": simplified,
                "n_raw": n_raw,
                "n_simp": n_simp,
                "reduction": reduction,
                "area_raw": raw.area,
                "area_simp": simplified.area,
                "area_pres": area_pres,
                "row": row,
                "raster_window_arr": arr,
                "raster_window_transform": win_transform,
            }

    if best_result is None:
        raise RuntimeError("No se encontró candidato")

    r = best_result
    print(f"  Vértices raw: {r['n_raw']}")
    print(f"  Vértices simp: {r['n_simp']}")
    print(f"  Reducción: {r['reduction']:.1f}%")
    print(f"  Área raw: {r['area_raw']:.2f} m²")
    print(f"  Área simp: {r['area_simp']:.2f} m²")
    print(f"  Área preservada: {r['area_pres']:.2f}%")

    # ============== Plot ==============
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    minx, miny, maxx, maxy = r["raw"].bounds
    pad = 2
    lim = (minx - pad, maxx + pad, miny - pad, maxy + pad)

    raster_arr = (r["raster_window_arr"] == CLASE_CONSTRUCCION).astype(np.uint8)
    rt = r["raster_window_transform"]
    raster_extent = (rt.c, rt.c + rt.a * raster_arr.shape[1],
                     rt.f + rt.e * raster_arr.shape[0], rt.f)

    for ax in (ax1, ax2):
        ax.imshow(raster_arr, extent=raster_extent, cmap="Greys",
                  alpha=0.25, origin="upper", interpolation="nearest")
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal")
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        ax.grid(True, alpha=0.3, linestyle="--")

    raw_x, raw_y = r["raw"].exterior.xy
    ax1.fill(raw_x, raw_y, facecolor="#FFB347", alpha=0.5, zorder=2)
    ax1.plot(raw_x, raw_y, color="#D2691E", linewidth=1.5, zorder=3)
    ax1.scatter(raw_x, raw_y, c="#8B0000", s=18, zorder=4,
                edgecolor="white", linewidth=0.5)

    simp_x, simp_y = r["simplified"].exterior.xy
    ax2.fill(simp_x, simp_y, facecolor="#90EE90", alpha=0.5, zorder=2)
    ax2.plot(simp_x, simp_y, color="#006400", linewidth=2.0, zorder=3)
    ax2.scatter(simp_x, simp_y, c="#003300", s=45, zorder=4,
                edgecolor="white", linewidth=1.0)

    ax1.set_title(
        f"(a) Polígono crudo (sin simplificar)\n"
        f"{r['n_raw']} vértices · área = {r['area_raw']:.1f} m²",
        fontsize=11, fontweight="bold"
    )
    ax2.set_title(
        f"(b) Tras RDP con tolerancia {RDP_TOL} m\n"
        f"{r['n_simp']} vértices · área = {r['area_simp']:.1f} m²",
        fontsize=11, fontweight="bold"
    )

    # ===== Panel (c): zoom de detalle anclado en una esquina real =====
    # Elegir el vértice del polígono simplificado más al noreste (esquina sup-der)
    simp_coords = list(r["simplified"].exterior.coords)
    cx, cy = max(simp_coords, key=lambda p: p[0] + p[1])
    zoom_size = 4.0  # 4 m de lado
    zoom_lim = (cx - zoom_size / 2, cx + zoom_size / 2,
                cy - zoom_size / 2, cy + zoom_size / 2)

    ax3.imshow(raster_arr, extent=raster_extent, cmap="Greys",
               alpha=0.5, origin="upper", interpolation="nearest")
    ax3.fill(raw_x, raw_y, facecolor="#FFB347", alpha=0.35, zorder=2)
    ax3.plot(raw_x, raw_y, color="#D2691E", linewidth=2,
             zorder=3, label=f"Crudo ({r['n_raw']} vért.)")
    ax3.scatter(raw_x, raw_y, c="#8B0000", s=40, zorder=4,
                edgecolor="white", linewidth=0.6)
    ax3.plot(simp_x, simp_y, color="#006400", linewidth=2.5,
             zorder=5, label=f"RDP {RDP_TOL} m ({r['n_simp']} vért.)")
    ax3.scatter(simp_x, simp_y, c="#003300", s=80, zorder=6,
                edgecolor="white", linewidth=1.2, marker="s")
    ax3.set_xlim(zoom_lim[0], zoom_lim[1])
    ax3.set_ylim(zoom_lim[2], zoom_lim[3])
    ax3.set_aspect("equal")
    ax3.set_xlabel("Este (m)")
    ax3.set_ylabel("Norte (m)")
    ax3.set_title(
        "(c) Zoom de detalle (4 m × 4 m)\n"
        "comparación de bordes raw vs RDP",
        fontsize=11, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Indicar la zona del zoom en (a) y (b)
    for ax in (ax1, ax2):
        rect = plt.Rectangle((zoom_lim[0], zoom_lim[2]), zoom_size, zoom_size,
                             fill=False, edgecolor="#FF1493",
                             linewidth=1.8, linestyle="-", zorder=10)
        ax.add_patch(rect)

    info_box = (
        f"Reducción de vértices: {r['n_raw']} → {r['n_simp']}  ({r['reduction']:.1f}%)\n"
        f"Área preservada: {r['area_pres']:.2f}%  "
        f"(diferencia: {abs(r['area_raw'] - r['area_simp']):.2f} m²)\n"
        f"Tolerancia RDP: {RDP_TOL} m  ·  Resolución del raster: 0.1 m/px"
    )
    fig.text(0.5, 0.01, info_box, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#999999"))

    fig.suptitle(
        "Figura 4.9 — Simplificación RDP (Ramer–Douglas–Peucker) sobre un polígono\n"
        "ejemplo de Construcción · DeepLabv3+ · Madrid",
        fontsize=13, fontweight="bold", y=0.995
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    out = OUT_DIR / "figura_4_9_rdp_simplification.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
