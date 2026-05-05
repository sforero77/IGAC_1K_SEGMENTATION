"""
Entrena un baseline OBIA (Object-Based Image Analysis):

  1. Por cada parche de train, segmentar con SLIC (~150 superpíxeles/parche).
  2. Calcular features por segmento (color RGB mean/std, gradiente promedio,
     área, perímetro, compactness, posición normalizada del centroide).
  3. Asignar clase = clase mayoritaria de los píxeles del segmento.
  4. Entrenar Random Forest sobre estos vectores de segmento.

Salida: models/checkpoints/obia_baseline/obia.joblib
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from skimage.segmentation import slic
from sklearn.ensemble import RandomForestClassifier

from scripts.utils.handcrafted_features import (
    extract_segment_features, majority_label_per_segment, N_SEGMENT_FEATURES,
)
from scripts.utils import N_CLASSES
from scripts.utils.config import CHECKPOINTS_DIR
from _baseline_common import (
    CLASS_NAMES_FULL, load_index, load_patch,
)


N_PATCHES = 600
SLIC_N_SEGMENTS = 150
SLIC_COMPACTNESS = 10.0
N_ESTIMATORS = 200
MAX_DEPTH = 25
SEED = 42

OUT_DIR = Path(CHECKPOINTS_DIR) / "obia_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("OBIA baseline (SLIC + RandomForest) — entrenamiento")
    print("=" * 70)

    print("\n[1/3] Cargando índice de parches (split=train)...")
    df_train = load_index("train")
    df_train = df_train.sample(frac=1.0, random_state=SEED).iloc[:N_PATCHES]
    print(f"  Parches a procesar: {len(df_train):,}")

    print(f"\n[2/3] Segmentación SLIC + extracción de features por segmento...")
    t0 = time.time()
    X_chunks = []
    y_chunks = []
    skipped = 0
    for i, row in enumerate(df_train.itertuples(index=False), start=1):
        try:
            img, mask = load_patch(row.image_path, row.mask_path)
        except FileNotFoundError:
            skipped += 1
            continue
        # SLIC trabaja mejor con datos en rango [0, 1]
        # Tomamos la imagen normalizada y la mapeamos a [0, 1] vía min-max
        img_for_slic = img.copy()
        lo, hi = np.percentile(img_for_slic, (1, 99))
        img_for_slic = np.clip((img_for_slic - lo) / max(hi - lo, 1e-6), 0, 1)
        segments = slic(
            img_for_slic,
            n_segments=SLIC_N_SEGMENTS,
            compactness=SLIC_COMPACTNESS,
            start_label=0,
            channel_axis=-1,
        )
        X, seg_ids = extract_segment_features(img, segments)
        if X.size == 0:
            continue
        labels_all = majority_label_per_segment(mask, segments, n_classes=N_CLASSES)
        y = labels_all[seg_ids]
        # filtrar segmentos cuya clase mayoritaria es Background
        keep = y != 0
        if not keep.any():
            continue
        X_chunks.append(X[keep])
        y_chunks.append(y[keep])
        if i % 50 == 0:
            print(f"  Procesados {i}/{len(df_train)} parches...")

    if skipped:
        print(f"  AVISO: {skipped} parches no encontrados")
    X = np.vstack(X_chunks).astype(np.float32)
    y = np.concatenate(y_chunks).astype(np.int64)
    print(f"  X shape: {X.shape}   y shape: {y.shape}")
    print(f"  Distribución por clase:")
    for cls in range(N_CLASSES):
        n = int((y == cls).sum())
        if n > 0:
            print(f"    {CLASS_NAMES_FULL[cls]:20s}  {n:>8,}")
    print(f"  Segmentación + features completado en {time.time() - t0:.1f}s")

    print(f"\n[3/3] Entrenando RandomForestClassifier sobre segmentos...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",
        verbose=1,
    )
    rf.fit(X, y)
    print(f"  Entrenamiento completado en {time.time() - t0:.1f}s")
    print(f"  Train accuracy (sobre segmentos): {rf.score(X, y):.4f}")

    out = OUT_DIR / "obia.joblib"
    joblib.dump(
        {
            "model": rf,
            "feature_extractor": "extract_segment_features",
            "n_features": X.shape[1],
            "class_names": CLASS_NAMES_FULL,
            "n_classes": N_CLASSES,
            "slic_params": {
                "n_segments": SLIC_N_SEGMENTS,
                "compactness": SLIC_COMPACTNESS,
                "start_label": 0,
            },
            "config": {
                "n_patches": N_PATCHES,
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
