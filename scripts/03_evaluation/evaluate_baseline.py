"""
Evalúa cualquiera de los baselines clásicos (RF, SVM, OBIA) sobre el split
de test del proyecto IGAC 1K, usando exactamente el mismo dataset y la
misma clase de métricas que los modelos de Deep Learning.

Genera un test_metrics.json con la MISMA estructura que produjo el notebook
05_inference_test_set.ipynb para los modelos DL, de modo que la tabla
comparativa del README/Capítulo 4 sea homogénea.

Uso:
  python scripts/03_evaluation/evaluate_baseline.py --model rf
  python scripts/03_evaluation/evaluate_baseline.py --model svm
  python scripts/03_evaluation/evaluate_baseline.py --model obia
  python scripts/03_evaluation/evaluate_baseline.py --model rf --max-patches 500   # debug rápido
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "02_training"))

import joblib
import numpy as np
import torch
from skimage.segmentation import slic

from scripts.utils.handcrafted_features import (
    extract_pixel_features, extract_segment_features,
)
from scripts.utils import N_CLASSES, CLASS_NAMES
from scripts.utils.metrics import SegmentationMetrics
from _baseline_common import CLASS_NAMES_FULL, load_index, load_patch


CHECKPOINT_PATHS = {
    "rf":   "rf_baseline/rf.joblib",
    "svm":  "svm_baseline/svm.joblib",
    "obia": "obia_baseline/obia.joblib",
}


def predict_pixelwise(model, img_norm: np.ndarray) -> np.ndarray:
    feats = extract_pixel_features(img_norm)
    pred_flat = model.predict(feats).astype(np.uint8)
    return pred_flat.reshape(img_norm.shape[:2])


def predict_obia(model, img_norm: np.ndarray, slic_params: dict) -> np.ndarray:
    img_for_slic = img_norm.copy()
    lo, hi = np.percentile(img_for_slic, (1, 99))
    img_for_slic = np.clip((img_for_slic - lo) / max(hi - lo, 1e-6), 0, 1)
    segments = slic(
        img_for_slic,
        n_segments=slic_params["n_segments"],
        compactness=slic_params["compactness"],
        start_label=slic_params.get("start_label", 0),
        channel_axis=-1,
    )
    X, seg_ids = extract_segment_features(img_norm, segments)
    h, w = img_norm.shape[:2]
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    if X.size == 0:
        return pred_mask
    seg_preds = model.predict(X).astype(np.uint8)
    seg_to_class = np.zeros(int(segments.max()) + 1, dtype=np.uint8)
    for seg_id, cls in zip(seg_ids, seg_preds):
        seg_to_class[seg_id] = cls
    pred_mask = seg_to_class[segments]
    return pred_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["rf", "svm", "obia"])
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Limitar el número de parches de test (debug)")
    args = parser.parse_args()

    from scripts.utils.config import CHECKPOINTS_DIR
    ckpt_path = Path(CHECKPOINTS_DIR) / CHECKPOINT_PATHS[args.model]
    print(f"Cargando modelo: {ckpt_path}")
    bundle = joblib.load(ckpt_path)
    model = bundle["model"]
    slic_params = bundle.get("slic_params")

    print("Cargando índice (split=test)...")
    df_test = load_index("test")
    if args.max_patches:
        df_test = df_test.iloc[:args.max_patches]
    print(f"  Parches a evaluar: {len(df_test):,}")

    metrics = SegmentationMetrics(
        num_classes=N_CLASSES,
        ignore_index=0,
        class_names={i: CLASS_NAMES_FULL[i] for i in range(N_CLASSES)},
    )

    t0 = time.time()
    skipped = 0
    for i, row in enumerate(df_test.itertuples(index=False), start=1):
        try:
            img, mask = load_patch(row.image_path, row.mask_path)
        except FileNotFoundError:
            skipped += 1
            continue
        if args.model == "obia":
            pred = predict_obia(model, img, slic_params)
        else:
            pred = predict_pixelwise(model, img)
        metrics.update(
            torch.from_numpy(pred[None]).long(),
            torch.from_numpy(mask[None]).long(),
        )
        if i % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(df_test) - i)
            print(f"  [{i:>5d}/{len(df_test)}]  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    if skipped:
        print(f"AVISO: {skipped} parches no encontrados")
    print(f"\nEvaluación completada en {(time.time() - t0) / 60:.1f} min")

    summary = metrics.get_summary(detailed=True)
    iou = metrics.get_iou_per_class()
    dice = metrics.get_dice_per_class()
    precision = metrics.get_precision_per_class()
    f1 = metrics.get_f1_per_class()

    per_class = {}
    for c in range(N_CLASSES):
        per_class[CLASS_NAMES_FULL[c]] = {
            "IoU": float(iou[c]) if not np.isnan(iou[c]) else 0.0,
            "Dice": float(dice[c]) if not np.isnan(dice[c]) else 0.0,
            "Precision": float(precision[c]) if not np.isnan(precision[c]) else 0.0,
            "Recall": float(metrics.get_class_accuracy()[c]) if not np.isnan(metrics.get_class_accuracy()[c]) else 0.0,
            "F1": float(f1[c]) if not np.isnan(f1[c]) else 0.0,
        }

    out_dir = ROOT / "results" / "test_evaluation" / f"{args.model}_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix.npy", metrics.confusion_matrix)

    payload = {
        "model_name": f"{args.model}_baseline",
        "feature_extractor": bundle.get("feature_extractor"),
        "test_date": datetime.utcnow().isoformat(),
        "test_patches": len(df_test) - skipped,
        "metrics": {
            "mIoU": float(metrics.get_miou(ignore_background=True)),
            "mIoU_with_bg": float(metrics.get_miou(ignore_background=False)),
            "pixel_accuracy": float(metrics.get_pixel_accuracy()),
        },
        "per_class": per_class,
        "config": bundle.get("config", {}),
    }
    if slic_params:
        payload["slic_params"] = slic_params

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {out_dir}/")
    print(f"  mIoU = {payload['metrics']['mIoU']:.4f}")
    print(f"  Pixel acc = {payload['metrics']['pixel_accuracy']:.4f}")
    print("\n  IoU por clase:")
    for c in range(1, N_CLASSES):
        print(f"    {CLASS_NAMES_FULL[c]:20s}  IoU={per_class[CLASS_NAMES_FULL[c]]['IoU']:.4f}")


if __name__ == "__main__":
    main()
