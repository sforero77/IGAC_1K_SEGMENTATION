"""
Entrena un Random Forest (RF) baseline píxel-a-píxel sobre el MISMO split
de entrenamiento usado para los modelos de Deep Learning.

  Features: 15 features handcrafted por píxel (RGB + estadísticas locales 5x5
  + Sobel + Laplaciano)  → ver scripts/utils/handcrafted_features.py

  Sampling: ~480 K píxeles (600 parches × 80 píxeles/clase × 8 clases).
  Class balance perfecto (mismo número de muestras por clase).

  Salida: models/checkpoints/rf_baseline/rf.joblib
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scripts.utils.handcrafted_features import extract_pixel_features
from scripts.utils import N_CLASSES
from scripts.utils.config import CHECKPOINTS_DIR
from scripts.utils.metrics import SegmentationMetrics
from _baseline_common import (
    CLASS_NAMES_FULL, load_index, sample_pixels_stratified,
)


N_PATCHES = 600
N_PER_CLASS_PER_PATCH = 80
N_ESTIMATORS = 200
MAX_DEPTH = 25
N_JOBS = -1
SEED = 42

OUT_DIR = Path(CHECKPOINTS_DIR) / "rf_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Random Forest baseline — entrenamiento")
    print("=" * 70)

    print("\n[1/3] Cargando índice de parches (split=train)...")
    df_train = load_index("train")
    print(f"  Parches train disponibles: {len(df_train):,}")

    print(f"\n[2/3] Sampling estratificado de píxeles "
          f"({N_PATCHES} parches × {N_PER_CLASS_PER_PATCH} px/clase × 8 clases)...")
    t0 = time.time()
    X, y = sample_pixels_stratified(
        df_train,
        feature_fn=extract_pixel_features,
        n_patches=N_PATCHES,
        n_per_class_per_patch=N_PER_CLASS_PER_PATCH,
        rng_seed=SEED,
    )
    print(f"  X shape: {X.shape}   y shape: {y.shape}")
    print(f"  Distribución por clase:")
    for cls in range(N_CLASSES):
        n = int((y == cls).sum())
        if n > 0:
            print(f"    {CLASS_NAMES_FULL[cls]:20s}  {n:>8,}")
    print(f"  Sampling completado en {time.time() - t0:.1f}s")

    print(f"\n[3/3] Entrenando RandomForestClassifier "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=1,
    )
    rf.fit(X, y)
    print(f"  Entrenamiento completado en {time.time() - t0:.1f}s")
    print(f"  Train accuracy: {rf.score(X, y):.4f}")

    out = OUT_DIR / "rf.joblib"
    joblib.dump(
        {
            "model": rf,
            "feature_extractor": "extract_pixel_features",
            "n_features": X.shape[1],
            "class_names": CLASS_NAMES_FULL,
            "n_classes": N_CLASSES,
            "config": {
                "n_patches": N_PATCHES,
                "n_per_class_per_patch": N_PER_CLASS_PER_PATCH,
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "seed": SEED,
            },
        },
        out,
        compress=3,
    )
    print(f"\nOK modelo guardado: {out}")


if __name__ == "__main__":
    main()
