"""
Genera la Figura 4.10 — Mosaico de parches con RGB, GT y predicciones de los
tres modelos (U-Net, DeepLabv3+, U-Net++) sobre el split de test.

Layout: 5 filas × 5 columnas
  Filas: 1 parche representativo de Madrid, Mosquera, Tabio, Tocancipá
         + 1 parche "difícil" (alta diversidad de clases)
  Columnas: (1) RGB · (2) Ground truth · (3) U-Net · (4) DeepLabv3+ · (5) U-Net++

Paleta consistente para 9 clases.
"""

from pathlib import Path

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from matplotlib.colors import ListedColormap

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
PATCHES_INDEX = ROOT / "data" / "processed" / "patches" / "patches_index.csv"
NORM_STATS = ROOT / "data" / "interim" / "normalization_stats.json"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "U-Net": ROOT / "models" / "checkpoints" / "unet_resnet34_baseline" / "best_checkpoint.pth",
    "DeepLabv3+": ROOT / "models" / "checkpoints" / "deeplabv3_resnet50_baseline" / "best_checkpoint.pth",
    "U-Net++": ROOT / "models" / "checkpoints" / "unetplusplus_resnet50_baseline" / "best_checkpoint.pth",
}
N_CLASSES = 9

CLASS_NAMES = [
    "Background", "Construcción", "Bosque", "Depósito Agua",
    "Zona Verde", "Zona Dura", "Vía", "Cerca", "Cabecera Municipal",
]
CLASS_COLORS = [
    "#FFFFFF",   # 0 background blanco
    "#EF1A2D",   # 1 Construcción
    "#16A34A",   # 2 Bosque
    "#1F77B4",   # 3 Depósito Agua
    "#7CFC00",   # 4 Zona Verde
    "#A0A0A0",   # 5 Zona Dura
    "#404040",   # 6 Vía
    "#8B4513",   # 7 Cerca
    "#F4D03F",   # 8 Cabecera Municipal
]
CMAP = ListedColormap(CLASS_COLORS)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(name):
    if name == "U-Net":
        return smp.Unet(encoder_name="resnet34", encoder_weights=None,
                        in_channels=3, classes=N_CLASSES)
    if name == "DeepLabv3+":
        return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None,
                                 in_channels=3, classes=N_CLASSES)
    if name == "U-Net++":
        return smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None,
                                in_channels=3, classes=N_CLASSES)
    raise ValueError(name)


def load_models():
    models = {}
    for name, path in CHECKPOINTS.items():
        m = build_model(name)
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        m.eval().to(DEVICE)
        models[name] = m
        print(f"  loaded {name} (epoch {ck['epoch']}, val_mIoU={ck['best_val_miou']:.4f})")
    return models


def select_patches():
    df = pd.read_csv(PATCHES_INDEX)
    test = df[df["split"] == "test"].copy()
    test["n_classes"] = test["classes_present"].apply(lambda s: len(eval(s)))

    selections = []
    for ciudad in ["madrid", "mosquera", "tabio", "tocancipa"]:
        sub = test[test["ciudad"] == ciudad]
        sub = sub[(sub["nodata_percent"] < 5) & (sub["n_classes"] >= 4)]
        if len(sub) == 0:
            sub = test[test["ciudad"] == ciudad]
        sub = sub.sort_values("n_classes", ascending=False)
        chosen = sub.iloc[len(sub) // 4]  # ni el más raro ni el más común
        selections.append({"label": f"{ciudad.capitalize()} — repres.",
                            "row": chosen})

    hard = test[(test["nodata_percent"] < 10) & (test["n_classes"] >= 6)]
    hard = hard.sort_values("n_classes", ascending=False)
    if len(hard) > 0:
        chosen = hard.iloc[len(hard) // 8]
        selections.append({"label": f"{chosen['ciudad'].capitalize()} — difícil "
                                     f"({chosen['n_classes']} clases)",
                            "row": chosen})

    return selections


def denorm_rgb(img_norm):
    """img_norm: (H, W, 3) float32 normalizado → (H, W, 3) uint8 visible."""
    with open(NORM_STATS) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    rgb = img_norm * std + mean
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


@torch.no_grad()
def predict(model, img_norm):
    """img_norm: (H,W,3) float32 → mask (H,W) uint8."""
    x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    logits = model(x)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def main():
    print(f"Device: {DEVICE}")
    print("Cargando modelos...")
    models = load_models()

    print("\nSeleccionando parches representativos...")
    selections = select_patches()
    for s in selections:
        r = s["row"]
        print(f"  {s['label']}: {r['patch_id']} (clases={r['n_classes']}, "
              f"nodata={r['nodata_percent']:.1f}%)")

    n_rows = len(selections)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.2 * n_rows))
    col_titles = ["(1) Imagen RGB", "(2) Ground truth",
                  "(3) U-Net", "(4) DeepLabv3+", "(5) U-Net++"]

    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=13, fontweight="bold", pad=10)

    def fix_path(p):
        # El CSV tiene rutas absolutas con la base D:\IGAC_1K_SEGMENTATION\data\...
        # pero el proyecto está en D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION\data\...
        p = Path(p)
        rel = Path(*p.parts[p.parts.index("data"):])
        return ROOT / rel

    for i, sel in enumerate(selections):
        r = sel["row"]
        img = np.load(fix_path(r["image_path"])).astype(np.float32)
        gt = np.load(fix_path(r["mask_path"])).astype(np.uint8)

        rgb = denorm_rgb(img)
        preds = {name: predict(m, img) for name, m in models.items()}

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt, cmap=CMAP, vmin=0, vmax=8, interpolation="nearest")
        axes[i, 2].imshow(preds["U-Net"], cmap=CMAP, vmin=0, vmax=8,
                          interpolation="nearest")
        axes[i, 3].imshow(preds["DeepLabv3+"], cmap=CMAP, vmin=0, vmax=8,
                          interpolation="nearest")
        axes[i, 4].imshow(preds["U-Net++"], cmap=CMAP, vmin=0, vmax=8,
                          interpolation="nearest")

        axes[i, 0].set_ylabel(sel["label"], fontsize=11, fontweight="bold")

        # IoU por modelo (clases 1-8) anotado en la esquina
        for j, mname in enumerate(["U-Net", "DeepLabv3+", "U-Net++"]):
            pred = preds[mname]
            ious = []
            for c in range(1, 9):
                gt_c = gt == c
                pr_c = pred == c
                if gt_c.sum() == 0 and pr_c.sum() == 0:
                    continue
                inter = (gt_c & pr_c).sum()
                union = (gt_c | pr_c).sum()
                if union > 0:
                    ious.append(inter / union)
            miou = np.mean(ious) if ious else 0.0
            axes[i, 2 + j].text(
                0.02, 0.98, f"mIoU = {miou:.3f}",
                transform=axes[i, 2 + j].transAxes,
                fontsize=10, fontweight="bold", color="white",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", edgecolor="none", alpha=0.7),
            )

        for j in range(n_cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[c], edgecolor="black",
                       linewidth=0.5, label=f"{c} · {CLASS_NAMES[c]}")
        for c in range(1, 9)
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=8, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.005),
               title="Paleta de clases (Background no se muestra)",
               title_fontsize=11)

    fig.suptitle(
        "Figura 4.10 — Ejemplos de parches: RGB, ground truth y predicciones por modelo\n"
        "Parches 512×512 del split de test · 4 municipios + 1 caso difícil · "
        "checkpoints best (U-Net ép. 29, DeepLabv3+ ép. 27, U-Net++ ép. 21)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out = OUT_DIR / "figura_4_10_ejemplos_parches.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nOK guardada: {out}")


if __name__ == "__main__":
    main()
