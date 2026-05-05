"""
Entrena un SVM baseline (LinearSVC con SGD para escalabilidad) píxel-a-píxel
sobre el mismo split de entrenamiento usado para Deep Learning y RF.

Notas:
  - SVM con kernel RBF NO escala a cientos de miles de píxeles. Usamos
    SGDClassifier(loss='hinge') que aproxima un Linear SVM y soporta
    millones de muestras.
  - Mismas features handcrafted que RF (15 features por píxel).
  - Estandarización (StandardScaler) antes del entrenamiento.

Salida: models/checkpoints/svm_baseline/svm.joblib
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.utils.handcrafted_features import extract_pixel_features
from scripts.utils import N_CLASSES
from scripts.utils.config import CHECKPOINTS_DIR
from _baseline_common import (
    CLASS_NAMES_FULL, load_index, sample_pixels_stratified,
)


N_PATCHES = 600
N_PER_CLASS_PER_PATCH = 80
SEED = 42
MAX_ITER = 200

OUT_DIR = Path(CHECKPOINTS_DIR) / "svm_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Linear SVM baseline (SGD-hinge) — entrenamiento")
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
    print(f"  Sampling completado en {time.time() - t0:.1f}s")

    print(f"\n[3/3] Entrenando Pipeline(StandardScaler + SGDClassifier(hinge))...")
    t0 = time.time()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            max_iter=MAX_ITER,
            tol=1e-4,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced",
            verbose=1,
        )),
    ])
    pipe.fit(X, y)
    print(f"  Entrenamiento completado en {time.time() - t0:.1f}s")
    print(f"  Train accuracy: {pipe.score(X, y):.4f}")

    out = OUT_DIR / "svm.joblib"
    joblib.dump(
        {
            "model": pipe,
            "feature_extractor": "extract_pixel_features",
            "n_features": X.shape[1],
            "class_names": CLASS_NAMES_FULL,
            "n_classes": N_CLASSES,
            "config": {
                "n_patches": N_PATCHES,
                "n_per_class_per_patch": N_PER_CLASS_PER_PATCH,
                "max_iter": MAX_ITER,
                "seed": SEED,
                "model_type": "SGDClassifier(loss=hinge)",
            },
        },
        out,
        compress=3,
    )
    print(f"\nOK modelo guardado: {out}")


if __name__ == "__main__":
    main()
