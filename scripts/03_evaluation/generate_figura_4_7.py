"""
Genera la Figura 4.7 — Matriz de confusión normalizada (DeepLabv3+ sobre test).

Heatmap 9×9, normalizado por fila (recall): cada fila suma 1, la diagonal
muestra el recall por clase. Anotaciones numéricas en cada celda; celdas con
confusión > 10% (fuera de diagonal) resaltadas con borde grueso. Pares
problemáticos destacados al lado del título.

Fuente:
  - results/test_evaluation/deeplabv3_resnet50_baseline/confusion_matrix.npy
  - results/test_evaluation/deeplabv3_resnet50_baseline/test_metrics.json
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
EVAL_DIR = ROOT / "results" / "test_evaluation" / "deeplabv3_resnet50_baseline"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Background", "Construcción", "Bosque", "Depósito\nAgua",
    "Zona\nVerde", "Zona\nDura", "Vía", "Cerca", "Cabecera\nMunicipal",
]
HIGHLIGHT_PAIRS = [
    (6, 5, "Vía → ZonaDura"),
    (4, 2, "ZonaVerde → Bosque"),
    (8, 5, "CabeceraMunicipal → ZonaDura"),
]


def main():
    cm = np.load(EVAL_DIR / "confusion_matrix.npy").astype(np.float64)
    with open(EVAL_DIR / "test_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    valid_mask = (row_sums.flatten() > 0)

    fig, ax = plt.subplots(figsize=(13, 11))

    cmap = plt.get_cmap("YlOrRd")
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    n = len(CLASSES)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel("Clase predicha", fontsize=12, fontweight="bold")
    ax.set_ylabel("Clase verdadera", fontsize=12, fontweight="bold")

    # Anotar valores en cada celda
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if not valid_mask[i]:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#999999")
                continue
            if v >= 0.001:
                txt = f"{v*100:.1f}"
            else:
                txt = "·"
            color = "white" if v > 0.55 else "black"
            weight = "bold" if (i == j or v > 0.10) else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    # Resaltar diagonal
    for i in range(n):
        if not valid_mask[i]:
            continue
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="#0B3D91",
                                    linewidth=2.2, zorder=3))

    # Resaltar celdas con confusión > 10% (fuera de la diagonal)
    for i in range(n):
        if not valid_mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if cm_norm[i, j] > 0.10:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="#8B0000",
                                            linewidth=2.0, linestyle="--",
                                            zorder=3))

    # Resaltar pares específicos del enunciado con flechas
    for (i, j, label) in HIGHLIGHT_PAIRS:
        if not valid_mask[i]:
            continue
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="#FF1493",
                                    linewidth=2.5, zorder=4))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proporción normalizada por fila (recall)", fontsize=11)

    miou = metrics["metrics"]["mIoU"]
    pix_acc = metrics["metrics"]["pixel_accuracy"]
    n_test = metrics["test_patches"]
    epoch = metrics["checkpoint_epoch"]

    legend_elems = [
        mpatches.Patch(facecolor="none", edgecolor="#0B3D91", linewidth=2.2,
                       label="Diagonal — recall por clase"),
        mpatches.Patch(facecolor="none", edgecolor="#8B0000", linewidth=2.0,
                       linestyle="--", label="Confusión > 10%"),
        mpatches.Patch(facecolor="none", edgecolor="#FF1493", linewidth=2.5,
                       label="Confusiones señaladas (Vía/ZonaDura, ZonaVerde/Bosque, CM/ZonaDura)"),
        mpatches.Patch(facecolor="#999999",
                       label="— : clase sin píxeles verdaderos en test (Background)"),
    ]
    ax.legend(handles=legend_elems, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=True)

    fig.suptitle(
        "Figura 4.7 — Matriz de confusión normalizada · DeepLabv3+ (ResNet-50) · test set\n"
        f"Checkpoint época {epoch} · mIoU₈ = {miou:.4f} · "
        f"pixel-accuracy = {pix_acc:.4f} · {n_test:,} parches de test\n"
        "Valores en % normalizados por fila (la fila suma 100). "
        "Diagonal = recall por clase.",
        fontsize=12, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out = OUT_DIR / "figura_4_7_confusion_matrix_deeplabv3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"OK guardada: {out}")
    print()
    print("Recall por clase (diagonal):")
    for i, c in enumerate(CLASSES):
        if valid_mask[i]:
            cname = c.replace("\n", " ")
            print(f"  {cname:<20} {cm_norm[i, i]*100:6.2f}%")
    print()
    print("Confusiones > 10% (fuera de diagonal):")
    for i in range(n):
        if not valid_mask[i]:
            continue
        for j in range(n):
            if i != j and cm_norm[i, j] > 0.10:
                ci = CLASSES[i].replace("\n", " ")
                cj = CLASSES[j].replace("\n", " ")
                print(f"  {ci:<20} -> {cj:<20} {cm_norm[i, j]*100:6.2f}%")


if __name__ == "__main__":
    main()
