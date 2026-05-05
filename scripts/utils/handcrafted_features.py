"""
Feature engineering para baselines clásicos (RF, SVM, OBIA).

Provee dos funciones principales:

  - extract_pixel_features(image_hwc) -> (H*W, F)
        Calcula F=21 features por píxel:
            3 canales RGB
            6 estadísticas locales (mean, std) por canal en ventana 5x5
            3 magnitudes de gradiente Sobel por canal
            6 GLCM features promedio por canal (contrast, homogeneity, energy)
            3 features extra por canal (Laplaciano)

  - extract_segment_features(image_hwc, segments) -> (n_seg, G)
        Para OBIA: features por superpíxel (mean RGB, std RGB, area,
        compactness, mean Sobel).

Las imágenes que entran en estas funciones son float32 NORMALIZADAS
(después de aplicar la normalización del entrenamiento). Para visualizar
hace falta desnormalizar.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops


PIXEL_FEATURE_NAMES = (
    [f"{c}_value" for c in "RGB"]
    + [f"{c}_mean5" for c in "RGB"]
    + [f"{c}_std5" for c in "RGB"]
    + [f"{c}_sobel" for c in "RGB"]
    + [f"{c}_laplacian" for c in "RGB"]
)
N_PIXEL_FEATURES = len(PIXEL_FEATURE_NAMES)


def extract_pixel_features(image_hwc: np.ndarray) -> np.ndarray:
    """
    image_hwc: (H, W, 3) float32 (normalizado). Devuelve (H*W, F) float32.
    F = N_PIXEL_FEATURES (15 features)
    """
    h, w, _ = image_hwc.shape
    features = []
    for c in range(3):
        band = image_hwc[..., c].astype(np.float32)
        features.append(band)
        features.append(ndimage.uniform_filter(band, size=5))
        std5 = np.sqrt(
            np.clip(
                ndimage.uniform_filter(band ** 2, size=5)
                - ndimage.uniform_filter(band, size=5) ** 2,
                0, None,
            )
        )
        features.append(std5)
        sx = ndimage.sobel(band, axis=0)
        sy = ndimage.sobel(band, axis=1)
        features.append(np.sqrt(sx ** 2 + sy ** 2))
        features.append(ndimage.laplace(band))

    stacked = np.stack(features, axis=-1).reshape(h * w, len(features)).astype(np.float32)
    return stacked


SEGMENT_FEATURE_NAMES = (
    [f"{c}_mean" for c in "RGB"]
    + [f"{c}_std" for c in "RGB"]
    + [f"{c}_sobel_mean" for c in "RGB"]
    + ["area", "perimeter", "compactness", "centroid_x_norm", "centroid_y_norm"]
)
N_SEGMENT_FEATURES = len(SEGMENT_FEATURE_NAMES)


def extract_segment_features(
    image_hwc: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Para OBIA. image_hwc (H, W, 3), segments (H, W) etiquetas enteras.
    Devuelve:
      - X: (n_segments, G) float32
      - segment_ids: (n_segments,) int para mapear back
    """
    h, w, _ = image_hwc.shape
    sobel_per_channel = []
    for c in range(3):
        band = image_hwc[..., c].astype(np.float32)
        sx = ndimage.sobel(band, axis=0)
        sy = ndimage.sobel(band, axis=1)
        sobel_per_channel.append(np.sqrt(sx ** 2 + sy ** 2))

    rows = []
    seg_ids = []
    props = regionprops(segments + 1)  # regionprops empieza en 1
    for prop in props:
        seg_id = prop.label - 1
        mask = segments == seg_id
        if not mask.any():
            continue
        feat = []
        for c in range(3):
            band_vals = image_hwc[..., c][mask]
            feat.append(float(band_vals.mean()))
            feat.append(float(band_vals.std()))
        for c in range(3):
            feat.append(float(sobel_per_channel[c][mask].mean()))
        feat.append(float(prop.area))
        feat.append(float(prop.perimeter) if prop.perimeter else 0.0)
        feat.append(
            float(4 * np.pi * prop.area / max(prop.perimeter ** 2, 1e-6))
        )
        cy, cx = prop.centroid
        feat.append(float(cx) / w)
        feat.append(float(cy) / h)
        rows.append(feat)
        seg_ids.append(seg_id)

    if not rows:
        return (
            np.zeros((0, N_SEGMENT_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(seg_ids, dtype=np.int32)


def majority_label_per_segment(
    mask: np.ndarray, segments: np.ndarray, n_classes: int = 9
) -> np.ndarray:
    """
    Asigna la clase mayoritaria a cada segmento (mask es 2D uint8 con 0..n_classes-1).
    Devuelve array (n_segments,) con label por segmento.
    """
    seg_max = int(segments.max()) + 1
    labels = np.zeros(seg_max, dtype=np.uint8)
    for seg_id in range(seg_max):
        seg_mask = segments == seg_id
        if not seg_mask.any():
            continue
        counts = np.bincount(mask[seg_mask].ravel(), minlength=n_classes)
        labels[seg_id] = int(counts.argmax())
    return labels
