"""
Genera la Figura 4.6 — Curvas de pérdida y mIoU por época para los tres modelos.

Layout 2×2:
  (a) Train loss vs época
  (b) Learning-rate schedule (CosineAnnealingWarmRestarts T0=10, T_mult=2)
      *NOTA: val_loss no fue registrado durante el entrenamiento (NaN en los
      training_history.json y en los TensorBoard logs). En su lugar se grafica
      el LR schedule para ilustrar la dinámica de optimización mencionada en
      el pie de figura.*
  (c) Train mIoU vs época
  (d) Validation mIoU vs época con línea horizontal en mejor valor por modelo
      y marcador en la época de selección (early-stopping).

Fuentes:
  - logs/tensorboard/unet_resnet34_baseline/20251115-123240
  - logs/tensorboard/deeplabv3_resnet50_baseline/20251117-134811
  - logs/tensorboard/unetplusplus_resnet50_baseline/20251119-220445
  - models/checkpoints/*/best_checkpoint.pth (history + lr)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(r"D:\IGAC_1K_SEGMENTATION\IGAC_1K_SEGMENTATION")
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "U-Net (ResNet-34)": {
        "tb": ROOT / "logs/tensorboard/unet_resnet34_baseline/20251115-123240",
        "ckpt": ROOT / "models/checkpoints/unet_resnet34_baseline/best_checkpoint.pth",
        "best_epoch": 29,
        "color": "#2E86C1",
    },
    "DeepLabv3+ (ResNet-50)": {
        "tb": ROOT / "logs/tensorboard/deeplabv3_resnet50_baseline/20251117-134811",
        "ckpt": ROOT / "models/checkpoints/deeplabv3_resnet50_baseline/best_checkpoint.pth",
        "best_epoch": 27,
        "color": "#E67E22",
    },
    "U-Net++ (ResNet-50)": {
        "tb": ROOT / "logs/tensorboard/unetplusplus_resnet50_baseline/20251119-220445",
        "ckpt": ROOT / "models/checkpoints/unetplusplus_resnet50_baseline/best_checkpoint.pth",
        "best_epoch": 21,
        "color": "#27AE60",
    },
}


def read_tb_scalar(run_dir, tag):
    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def normalize_epochs(steps):
    """Reindexa steps a 1..N (los TB de U-Net empiezan en 16 porque se reanudó)."""
    return np.arange(1, len(steps) + 1)


def load_lr(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return np.array(c["history"]["lr"])


def main():
    data = {}
    for name, cfg in MODELS.items():
        _, train_loss = read_tb_scalar(cfg["tb"], "Loss/train")
        _, train_miou = read_tb_scalar(cfg["tb"], "mIoU/train")
        _, val_miou = read_tb_scalar(cfg["tb"], "mIoU/val")
        _, lr = read_tb_scalar(cfg["tb"], "Learning_rate")
        epochs = normalize_epochs(train_loss)
        # Alinear longitudes (todas vienen de la misma run)
        n = min(len(train_loss), len(train_miou), len(val_miou), len(lr))
        train_loss = train_loss[:n]
        train_miou = train_miou[:n]
        val_miou = val_miou[:n]
        lr = lr[:n]
        epochs = epochs[:n]
        data[name] = {
            "epochs": epochs,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_miou": val_miou,
            "lr": lr,
            "best_epoch": cfg["best_epoch"],
            "color": cfg["color"],
        }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    for name, d in data.items():
        ax_a.plot(d["epochs"], d["train_loss"], color=d["color"],
                  linewidth=2, label=name, alpha=0.9)
    ax_a.set_title("(a) Pérdida en entrenamiento", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Época")
    ax_a.set_ylabel("Loss (CE + Dice ponderada)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=10, loc="upper right")

    for name, d in data.items():
        ax_b.plot(d["epochs"], d["lr"], color=d["color"],
                  linewidth=2, label=name, alpha=0.9)
    # Marcas de reinicio del cosine annealing (T0=10, T_mult=2 → reinicios en 10, 30, 70...)
    for restart in [10, 30]:
        ax_b.axvline(restart, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax_b.text(restart, 1.05, f"reinicio (ép. {restart})", transform=ax_b.get_xaxis_transform(),
                  fontsize=8, ha="center", color="gray", style="italic")
    ax_b.set_title("(b) Learning-rate schedule — CosineAnnealingWarmRestarts (T₀=10, T_mult=2)",
                   fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Época")
    ax_b.set_ylabel("Learning rate")
    ax_b.set_yscale("log")
    ax_b.grid(True, alpha=0.3, which="both")
    ax_b.legend(fontsize=10, loc="upper right")
    note = ("Sustituye Validation loss: no fue registrado durante el\n"
            "entrenamiento (val_loss = NaN en histories y TB logs).")
    ax_b.text(0.98, 0.02, note, transform=ax_b.transAxes,
              fontsize=8.5, ha="right", va="bottom", style="italic",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8DC",
                        edgecolor="#B8860B", alpha=0.9))

    for name, d in data.items():
        ax_c.plot(d["epochs"], d["train_miou"], color=d["color"],
                  linewidth=2, label=name, alpha=0.9)
    ax_c.set_title("(c) mIoU en entrenamiento", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Época")
    ax_c.set_ylabel("mIoU (8 clases)")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=10, loc="lower right")

    annotation_offsets = {
        "U-Net (ResNet-34)": (10, -45),
        "DeepLabv3+ (ResNet-50)": (10, 18),
        "U-Net++ (ResNet-50)": (-95, -45),
    }
    for name, d in data.items():
        ax_d.plot(d["epochs"], d["val_miou"], color=d["color"],
                  linewidth=2, label=name, alpha=0.9)
        be = d["best_epoch"]
        # Encontrar el valor en la época "best" (val_miou en el índice be-1)
        if be <= len(d["val_miou"]):
            best_val = float(d["val_miou"][be - 1])
            ax_d.axhline(best_val, color=d["color"], linestyle="--",
                         linewidth=1, alpha=0.5)
            ax_d.scatter([be], [best_val], color=d["color"], s=140,
                         zorder=5, edgecolor="black", linewidth=1.4)
            dx, dy = annotation_offsets.get(name, (10, 12))
            ax_d.annotate(
                f"{name.split(' (')[0]}\nép. {be} · mIoU = {best_val:.3f}",
                xy=(be, best_val),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight="bold", color=d["color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=d["color"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=d["color"], lw=1),
            )
    ax_d.set_title("(d) mIoU en validación con época de selección (early stopping)",
                   fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Época")
    ax_d.set_ylabel("mIoU₈ (validación)")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        "Figura 4.6 — Curvas de pérdida y mIoU por época para los tres modelos\n"
        "Optimización con CosineAnnealingWarmRestarts (T₀=10, T_mult=2). "
        "Épocas seleccionadas por early stopping: U-Net = 29, DeepLabv3+ = 27, U-Net++ = 21.",
        fontsize=13, fontweight="bold", y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "figura_4_6_training_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK guardada: {out}")
    for name, d in data.items():
        print(f"  {name}: épocas={len(d['epochs'])}, "
              f"best_epoch={d['best_epoch']}, "
              f"max_val_miou={d['val_miou'].max():.4f}")


if __name__ == "__main__":
    main()
