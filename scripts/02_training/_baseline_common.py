"""
Utilidades comunes para los baselines clásicos (RF, SVM, OBIA).

Incluye:
  - resolución de paths (el patches_index.csv tiene rutas absolutas con el
    prefijo legacy D:\\IGAC_1K_SEGMENTATION\\data\\... mientras el repo
    actual está en D:\\IGAC_1K_SEGMENTATION\\IGAC_1K_SEGMENTATION\\data\\...)
  - lista de parches por split
  - sampling estratificado de píxeles
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PATCHES_INDEX = ROOT / "data" / "processed" / "patches" / "patches_index.csv"

CLASS_NAMES_FULL = [
    "Background",
    "Construccion",
    "Bosque",
    "DepositoAgua",
    "ZonaVerde",
    "ZonaDura",
    "Via",
    "Cerca",
    "CabeceraMunicipal",
]
N_CLASSES = len(CLASS_NAMES_FULL)
IGNORE_INDEX = 0


def fix_path(p: str) -> Path:
    """Reescribe rutas legacy a la ubicación actual del repositorio."""
    p = Path(p)
    parts = p.parts
    if "data" in parts:
        rel = Path(*parts[parts.index("data"):])
        return ROOT / rel
    return p


def load_index(split: str) -> pd.DataFrame:
    df = pd.read_csv(PATCHES_INDEX)
    df = df[df["split"] == split].reset_index(drop=True).copy()
    df["image_path"] = df["image_path"].apply(fix_path)
    df["mask_path"] = df["mask_path"].apply(fix_path)
    return df


def load_patch(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = np.load(image_path).astype(np.float32)   # (H, W, 3) normalizado
    mask = np.load(mask_path).astype(np.uint8)     # (H, W)
    return img, mask


def sample_pixels_stratified(
    df: pd.DataFrame,
    feature_fn,
    n_patches: int = 600,
    n_per_class_per_patch: int = 80,
    rng_seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Toma `n_patches` parches aleatorios; en cada uno muestrea hasta
    `n_per_class_per_patch` píxeles por clase (excluyendo background).

    feature_fn: callable que recibe (H, W, 3) y devuelve (H*W, F).
    """
    rng = np.random.default_rng(rng_seed)
    df_shuffled = df.sample(frac=1.0, random_state=rng_seed).iloc[:n_patches]

    X_chunks = []
    y_chunks = []
    skipped = 0
    for i, row in enumerate(df_shuffled.itertuples(index=False), start=1):
        try:
            img, mask = load_patch(row.image_path, row.mask_path)
        except FileNotFoundError:
            skipped += 1
            continue
        feats = feature_fn(img)  # (H*W, F)
        labels = mask.reshape(-1).astype(np.int64)
        for cls in range(1, N_CLASSES):
            cls_idx = np.where(labels == cls)[0]
            if cls_idx.size == 0:
                continue
            n_take = min(n_per_class_per_patch, cls_idx.size)
            chosen = rng.choice(cls_idx, size=n_take, replace=False)
            X_chunks.append(feats[chosen])
            y_chunks.append(labels[chosen])
        if verbose and i % 50 == 0:
            print(f"  Procesados {i}/{len(df_shuffled)} parches...")

    if skipped:
        print(f"  AVISO: {skipped} parches no encontrados (rutas legacy?)")

    X = np.vstack(X_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.int64)
    return X, y
