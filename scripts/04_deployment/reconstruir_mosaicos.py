"""
Reconstruye mosaicos de predicción y GT para cada ciudad del test set.
Guarda: gt_mosaic.npy, pred_mosaic.npy y PNGs comparativos.

Uso:
    python scripts/reconstruir_mosaicos.py
    python scripts/reconstruir_mosaicos.py --ciudad tabio
"""

import sys, os, argparse
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from scripts.utils import (
    get_device, N_CLASSES, CLASS_NAMES, CLASS_COLORS,
    PATCHES_INDEX, CHECKPOINTS_DIR, RESULTS_DIR
)

# ── Constantes ────────────────────────────────────────────────────────────────
PATCHES_BASE = Path('D:/IGAC_1K_SEGMENTATION/IGAC_1K_SEGMENTATION/data/processed/patches')
OUTPUT_DIR   = RESULTS_DIR / 'mosaicos'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_PX  = 512    # tamaño del parche en píxeles
OVERLAP   = 128    # overlap en píxeles
STEP_PX   = PATCH_PX - OVERLAP   # 384 px = paso en píxeles entre parches

def make_idx_map(vals):
    """Mapea coordenadas geográficas a índices de grilla de forma incremental.
    Evita acumulación de error con pasos alternantes 38/39m (38.4m real).
    """
    vals = np.sort(np.unique(vals))
    if len(vals) == 1:
        return {int(vals[0]): 0}
    min_step = int(round(np.min(np.diff(vals))))
    idx_map = {int(vals[0]): 0}
    col = 0
    prev = vals[0]
    for v in vals[1:]:
        col += round((v - prev) / min_step)
        idx_map[int(v)] = col
        prev = v
    return idx_map

# Colormap
CMAP = np.zeros((N_CLASSES, 3), dtype=np.uint8)
for cls_id, rgb in CLASS_COLORS.items():
    if cls_id < N_CLASSES:
        CMAP[cls_id] = rgb

def mask_to_rgb(mask):
    return CMAP[np.clip(mask, 0, N_CLASSES - 1)]

def resize_nearest(arr, max_px=4000):
    h, w = arr.shape[:2]
    scale = min(max_px / max(h, w), 1.0)
    if scale >= 1.0:
        return arr
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    row_idx = (np.arange(nh) * h / nh).astype(int)
    col_idx = (np.arange(nw) * w / nw).astype(int)
    return arr[np.ix_(row_idx, col_idx)]

# ── Inferencia ────────────────────────────────────────────────────────────────
def load_model():
    device = get_device()
    ckpt_path = CHECKPOINTS_DIR / 'deeplabv3_resnet50_baseline' / 'best_checkpoint.pth'
    model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                              in_channels=3, classes=N_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f'Modelo listo: epoch {ckpt["epoch"]}, val mIoU={ckpt["best_val_miou"]:.4f}')
    return model, device

def run_inference(model, device, patch_ids, df_all):
    """Inferencia sobre lista de patch_ids → dict {pid: pred uint8 (512,512)}."""
    preds = {}
    BS = 16
    for start in tqdm(range(0, len(patch_ids), BS), desc='  Inferencia', leave=False):
        batch = patch_ids[start:start+BS]
        imgs  = []
        for pid in batch:
            split = df_all.loc[df_all['patch_id'] == pid, 'split'].values[0]
            img   = np.load(PATCHES_BASE / split / 'images' / f'{pid}.npy')  # float32 normalizado
            imgs.append(img)
        t = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            out = torch.argmax(model(t), dim=1).cpu().numpy().astype(np.uint8)
        for i, pid in enumerate(batch):
            preds[pid] = out[i]
    return preds

# ── Construcción del mosaico ──────────────────────────────────────────────────
def build_mosaic(df_city, preds_dict=None, use_gt=False):
    """
    Construye el mosaico completo de una ciudad.
    Usa make_idx_map() para calcular índices de grilla sin acumular error
    con pasos alternantes 38/39m (38.4m real).
    """
    x_to_col = make_idx_map(df_city['center_x'].values)
    y_to_row  = make_idx_map(df_city['center_y'].values)

    canvas_w = max(x_to_col.values()) * STEP_PX + PATCH_PX
    canvas_h = max(y_to_row.values())  * STEP_PX + PATCH_PX

    canvas   = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    priority = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Peso por posición: zona central = 2, bordes de overlap = 1
    w_patch = np.ones((PATCH_PX, PATCH_PX), dtype=np.uint8)
    w_patch[OVERLAP:PATCH_PX-OVERLAP, OVERLAP:PATCH_PX-OVERLAP] = 2

    for _, row in df_city.iterrows():
        pid = row['patch_id']
        px  = x_to_col[int(row['center_x'])] * STEP_PX
        py  = y_to_row[int(row['center_y'])]  * STEP_PX

        if use_gt:
            split = row['split']
            tile  = np.load(PATCHES_BASE / split / 'masks' / f'{pid}.npy').astype(np.uint8)
        else:
            if preds_dict is None or pid not in preds_dict:
                continue
            tile = preds_dict[pid]

        # Escribir con prioridad (centro gana sobre borde)
        region_prio = priority[py:py+PATCH_PX, px:px+PATCH_PX]
        update      = w_patch >= region_prio
        canvas  [py:py+PATCH_PX, px:px+PATCH_PX][update] = tile[update]
        priority[py:py+PATCH_PX, px:px+PATCH_PX][update] = w_patch[update]

    return canvas, x_to_col, y_to_row

# ── Export PNG ────────────────────────────────────────────────────────────────
def save_png(mask, path, max_px=4000):
    small = resize_nearest(mask, max_px)
    Image.fromarray(mask_to_rgb(small)).save(path)
    print(f'    PNG guardado: {path.name}  ({small.shape[1]}×{small.shape[0]}px)')

def save_comparativo(gt, pred, city_dir, ciudad, max_px=4000):
    titles = ['Ground Truth', 'DeepLabv3+ Predicción']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    for ax, mask, title in zip(axes, [gt, pred], titles):
        ax.imshow(mask_to_rgb(resize_nearest(mask, max_px)))
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
    legend = [mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
              for i in range(1, N_CLASSES)]
    h, w = gt.shape
    fig.suptitle(f'{ciudad.capitalize()}  —  {w}×{h} px  ({w*0.1/1000:.1f} km × {h*0.1/1000:.1f} km)',
                 fontsize=12, fontweight='bold')
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    p = city_dir / 'comparativo.png'
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Comparativo guardado: {p.name}')

# ── Main ──────────────────────────────────────────────────────────────────────
def process_ciudad(ciudad, df_all, df_test, model, device, max_px=4000):
    print(f'\n=== {ciudad.upper()} ===')
    df_city   = df_test[df_test['ciudad'] == ciudad].copy()
    patch_ids = df_city['patch_id'].tolist()
    city_dir  = OUTPUT_DIR / ciudad
    city_dir.mkdir(exist_ok=True)

    # Inferencia
    print(f'  {len(patch_ids)} parches...')
    preds = run_inference(model, device, patch_ids, df_all)

    # Mosaicos
    print('  Construyendo mosaicos...')
    pred_mosaic, x_to_col, y_to_row = build_mosaic(df_city, preds, use_gt=False)
    gt_mosaic, _, _                  = build_mosaic(df_city, use_gt=True)

    print(f'  Canvas: {pred_mosaic.shape[1]}×{pred_mosaic.shape[0]} px  '
          f'({pred_mosaic.shape[1]*0.1/1000:.1f}km × {pred_mosaic.shape[0]*0.1/1000:.1f}km)')

    # Guardar numpy
    np.save(city_dir / 'pred_mosaic.npy', pred_mosaic)
    np.save(city_dir / 'gt_mosaic.npy',   gt_mosaic)

    # Guardar PNGs individuales
    save_png(gt_mosaic,   city_dir / 'GT.png',   max_px)
    save_png(pred_mosaic, city_dir / 'Pred.png',  max_px)

    # Comparativo
    save_comparativo(gt_mosaic, pred_mosaic, city_dir, ciudad, max_px)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ciudad', default=None,
                        help='tabio / madrid / mosquera / tocancipa (default: todas)')
    parser.add_argument('--max_px', type=int, default=4000)
    args = parser.parse_args()

    df_all  = pd.read_csv(PATCHES_INDEX)
    df_test = df_all[df_all['split'] == 'test'].copy()

    model, device = load_model()

    ciudades = [args.ciudad] if args.ciudad else ['tabio', 'madrid', 'mosquera']
    for c in ciudades:
        process_ciudad(c, df_all, df_test, model, device, args.max_px)

    print('\nListo. Archivos en:', OUTPUT_DIR)
