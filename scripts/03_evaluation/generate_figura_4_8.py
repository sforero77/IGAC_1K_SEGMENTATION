"""
Genera la Figura 4.8 — Histograma de áreas de polígonos generados por clase
(Construcción y Bosque, agregando los 4 municipios).

Layout 1×3:
  (a) Áreas de Construcción (n = 10 789)
  (b) Áreas de Bosque       (n =  7 412)
  (c) Densidad superpuesta (KDE log-normal) para comparar las dos distribuciones

En cada histograma se marcan los percentiles 5, 50 (mediana) y 95.
Eje X en escala logarítmica (m²).

Fuentes: results/vectores/construccion_{ciudad}.gpkg, bosque_{ciudad}.gpkg
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CIUDADES = ["madrid", "mosquera", "tabio", "tocancipa"]

COLOR_CONS = "#EF1A2D"
COLOR_BOSQ = "#16A34A"
PCTL_COLORS = {5: "#1F77B4", 50: "#FF8C00", 95: "#8B0000"}


def load_areas(clase):
    out = []
    for c in CIUDADES:
        gdf = gpd.read_file(ROOT / "results" / "vectores" / f"{clase}_{c}.gpkg")
        out.append(gdf.geometry.area.values)
    return np.concatenate(out)


def plot_hist(ax, areas, color, name):
    p5 = np.percentile(areas, 5)
    p50 = np.percentile(areas, 50)
    p95 = np.percentile(areas, 95)

    bins = np.logspace(np.log10(max(areas.min(), 1)),
                       np.log10(areas.max()), 60)
    ax.hist(areas, bins=bins, color=color, alpha=0.75,
            edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")

    for p, val, color_p in [
        (5, p5, PCTL_COLORS[5]),
        (50, p50, PCTL_COLORS[50]),
        (95, p95, PCTL_COLORS[95]),
    ]:
        ax.axvline(val, color=color_p, linestyle="--", linewidth=2, zorder=5)
        ax.text(val, ax.get_ylim()[1] * 0.95 if False else 0,
                f"P{p}\n{val:,.0f} m²",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color_p,
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color_p, alpha=0.95))

    ax.set_xlabel("Área del polígono (m²) — escala log", fontsize=11)
    ax.set_ylabel("Frecuencia (n.º de polígonos)", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    stats = (
        f"n = {len(areas):,}\n"
        f"min = {areas.min():.0f} m²\n"
        f"P5 = {p5:.0f} m²\n"
        f"mediana = {p50:.0f} m²\n"
        f"media = {areas.mean():.0f} m²\n"
        f"P95 = {p95:,.0f} m²\n"
        f"max = {areas.max():,.0f} m²\n"
        f"Σárea = {areas.sum() / 1e4:,.1f} ha"
    )
    ax.text(0.99, 0.99, stats, transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="black", alpha=0.95))


def plot_overlay(ax, cons, bosq):
    bins = np.logspace(
        np.log10(max(min(cons.min(), bosq.min()), 1)),
        np.log10(max(cons.max(), bosq.max())),
        60,
    )
    ax.hist(cons, bins=bins, color=COLOR_CONS, alpha=0.55,
            label=f"Construcción (n={len(cons):,})",
            density=True, edgecolor="#660A12", linewidth=0.4)
    ax.hist(bosq, bins=bins, color=COLOR_BOSQ, alpha=0.55,
            label=f"Bosque (n={len(bosq):,})",
            density=True, edgecolor="#064E1A", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Área del polígono (m²) — escala log", fontsize=11)
    ax.set_ylabel("Densidad (normalizada)", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.set_title("(c) Densidad superpuesta — distribución log-normal por clase",
                 fontsize=12, fontweight="bold")


def main():
    cons = load_areas("construccion")
    bosq = load_areas("bosque")
    print(f"Construcción: n={len(cons):,}, mediana={np.median(cons):.1f} m², "
          f"P5={np.percentile(cons, 5):.1f}, P95={np.percentile(cons, 95):.1f}, "
          f"max={cons.max():.1f}")
    print(f"Bosque:       n={len(bosq):,}, mediana={np.median(bosq):.1f} m², "
          f"P5={np.percentile(bosq, 5):.1f}, P95={np.percentile(bosq, 95):.1f}, "
          f"max={bosq.max():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    plot_hist(axes[0], cons, COLOR_CONS, "Construcción")
    axes[0].set_title(f"(a) Áreas de Construcción · n = {len(cons):,}",
                      fontsize=13, fontweight="bold")

    plot_hist(axes[1], bosq, COLOR_BOSQ, "Bosque")
    axes[1].set_title(f"(b) Áreas de Bosque · n = {len(bosq):,}",
                      fontsize=13, fontweight="bold")

    plot_overlay(axes[2], cons, bosq)

    fig.suptitle(
        "Figura 4.8 — Histograma de áreas de polígonos generados por clase\n"
        "Cuatro municipios agregados (Madrid, Mosquera, Tabio, Tocancipá) · "
        "DeepLabv3+ + RDP 0.5 m  ·  Eje X en escala logarítmica",
        fontsize=14, fontweight="bold", y=1.005,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "figura_4_8_histograma_areas.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
