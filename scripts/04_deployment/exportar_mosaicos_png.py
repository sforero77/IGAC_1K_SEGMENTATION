"""
Exporta los mosaicos .npy como imágenes PNG con colores por clase.
Genera: gt.png, pred.png, pred_pp.png, comparativo.png para cada ciudad.

Uso:
    python scripts/exportar_mosaicos_png.py
    python scripts/exportar_mosaicos_png.py --ciudad tabio
    python scripts/exportar_mosaicos_png.py --max_px 6000
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import argparse

# ── Configuración ────────────────────────────────────────────────────────────
from scripts.utils import N_CLASSES, CLASS_NAMES, CLASS_COLORS

MOSAICOS_DIR = Path('D:/IGAC_1K_SEGMENTATION/IGAC_1K_SEGMENTATION/results/mosaicos')

# Colormap: array (N_CLASSES, 3) uint8
CMAP = np.zeros((N_CLASSES, 3), dtype=np.uint8)
for cls_id, rgb in CLASS_COLORS.items():
    if cls_id < N_CLASSES:
        CMAP[cls_id] = rgb

def mask_to_rgb(mask):
    """Máscara (H,W) int → RGB (H,W,3) uint8."""
    return CMAP[mask.clip(0, N_CLASSES - 1)]

def resize_nearest(arr, max_px=4000):
    """Reduce el array al tamaño máximo preservando aspecto con interpolación nearest."""
    h, w = arr.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale >= 1.0:
        return arr
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    # Nearest neighbor manual (evita dependencia de cv2)
    row_idx = (np.arange(nh) * h / nh).astype(int)
    col_idx = (np.arange(nw) * w / nw).astype(int)
    return arr[np.ix_(row_idx, col_idx)]

def save_png(mask, path, max_px=4000):
    """Guarda máscara como PNG con colormap de clases."""
    small = resize_nearest(mask, max_px)
    rgb   = mask_to_rgb(small)
    Image.fromarray(rgb).save(path)
    print(f'  Guardado: {path}  ({rgb.shape[1]}x{rgb.shape[0]}px)')

def legend_patches():
    return [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_NAMES[i])
        for i in range(1, N_CLASSES)
    ]

def export_ciudad(ciudad, max_px=4000, save_comparativo=True):
    city_dir = MOSAICOS_DIR / ciudad
    if not city_dir.exists():
        print(f'  [{ciudad}] no encontrado en {city_dir}')
        return

    files = {
        'gt':      city_dir / 'gt_mosaic.npy',
        'pred':    city_dir / 'pred_mosaic.npy',
        'pred_pp': city_dir / 'pred_pp_mosaic.npy',
    }

    mosaics = {}
    for key, fpath in files.items():
        if fpath.exists():
            mosaics[key] = np.load(fpath)
            print(f'  Cargado {key}: {mosaics[key].shape}')
        else:
            print(f'  FALTA: {fpath.name}')

    # ── PNGs individuales ───────────────────────────────────────────────────
    labels = {'gt': 'GroundTruth', 'pred': 'Prediccion', 'pred_pp': 'Prediccion_PP'}
    for key, mask in mosaics.items():
        out_path = city_dir / f'{labels[key]}.png'
        save_png(mask, out_path, max_px)

    # ── Comparativo 3 paneles ───────────────────────────────────────────────
    if save_comparativo and len(mosaics) == 3:
        titles = {
            'gt':      'Ground Truth',
            'pred':    'DeepLabv3+ (sin PP)',
            'pred_pp': 'DeepLabv3+ (con PP)',
        }
        keys = ['gt', 'pred', 'pred_pp']
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)

        for ax, key in zip(axes, keys):
            small = resize_nearest(mosaics[key], max_px)
            ax.imshow(mask_to_rgb(small))
            ax.set_title(titles[key], fontsize=13, fontweight='bold')
            ax.axis('off')

        h, w = mosaics['gt'].shape
        fig.suptitle(
            f'{ciudad.capitalize()} — Mosaico completo ({w}×{h} px  |  {w*0.1:.0f}m × {h*0.1:.0f}m)',
            fontsize=13, fontweight='bold', y=1.01
        )
        fig.legend(handles=legend_patches(), loc='lower center',
                   ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout()
        comp_path = city_dir / 'comparativo_3panel.png'
        plt.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Guardado comparativo: {comp_path}')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ciudad', default=None,
                        help='Ciudad específica (tabio/madrid/mosquera/tocancipa). Default: todas')
    parser.add_argument('--max_px', type=int, default=4000,
                        help='Píxeles máximos en el lado mayor del PNG (default: 4000)')
    args = parser.parse_args()

    ciudades = [args.ciudad] if args.ciudad else ['tabio', 'madrid', 'mosquera', 'tocancipa']

    for ciudad in ciudades:
        print(f'\n=== {ciudad.upper()} ===')
        export_ciudad(ciudad, max_px=args.max_px)

    print('\nListo.')
