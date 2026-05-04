"""
Genera el paquete DLPK de ArcGIS Pro para el modelo DeepLabV3+ IGAC.

Exporta el modelo a TorchScript (.pt) — formato autocontenido que solo
necesita torch (ya incluido en ArcGIS Pro). No requiere instalar nada extra.

Uso:
    python scripts/exportar_dlpk.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import json, shutil, zipfile
import torch
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.utils import CHECKPOINTS_DIR, N_CLASSES, CLASS_NAMES, CLASS_COLORS

OUTPUT_DIR = Path('dlpk_output')
OUTPUT_DIR.mkdir(exist_ok=True)

DLPK_NAME = 'deeplabv3plus_igac'
CKPT_PATH = CHECKPOINTS_DIR / 'deeplabv3_resnet50_baseline' / 'best_checkpoint.pth'

# ── 1. Exportar a TorchScript ──────────────────────────────────────────────────
print('Exportando a TorchScript...')
device = torch.device('cpu')

model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                          in_channels=3, classes=N_CLASSES)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

dummy = torch.randn(1, 3, 512, 512)
traced = torch.jit.trace(model, dummy)

ts_path = OUTPUT_DIR / f'{DLPK_NAME}.pt'
traced.save(str(ts_path))
print(f'TorchScript guardado: {ts_path}  ({ts_path.stat().st_size / 1e6:.0f} MB)')

# ── 2. Generar EMD ─────────────────────────────────────────────────────────────
classes = []
for cls_id in sorted(CLASS_NAMES.keys()):
    classes.append({
        "Value": int(cls_id),
        "Name":  CLASS_NAMES[cls_id],
        "Color": list(CLASS_COLORS.get(cls_id, [0, 0, 0]))
    })

emd = {
    "Framework":         "PyTorch",
    "ModelType":         "ImageClassification",
    "InferenceFunction": ".\\ImageClassifier.py",
    "ModelFile":         f".\\{DLPK_NAME}.pt",
    "ModelName":         "DeepLabV3Plus IGAC 9 clases",
    "Version":           "2025.1",
    "ImageHeight":       512,
    "ImageWidth":        512,
    "BatchSize":         4,
    "ExtractBands":      [0, 1, 2],
    "ImageSpaceUsed":    "MAP_SPACE",
    "ModelPadding":      64,
    "DataRange":         [0.0, 255.0],
    "NormalizationStats": {
        "band_mean_1": 0.485, "band_mean_2": 0.456, "band_mean_3": 0.406,
        "band_std_1":  0.229, "band_std_2":  0.224, "band_std_3":  0.225
    },
    "AllTileStats": [
        {"Min": 0, "Max": 255, "Mean": 123.675, "StdDev": 58.395},
        {"Min": 0, "Max": 255, "Mean": 116.280, "StdDev": 57.120},
        {"Min": 0, "Max": 255, "Mean": 103.530, "StdDev": 57.375}
    ],
    "Classes": classes
}

emd_path = OUTPUT_DIR / f'{DLPK_NAME}.emd'
with open(emd_path, 'w', encoding='utf-8') as f:
    json.dump(emd, f, indent=2, ensure_ascii=False)
print(f'EMD generado: {emd_path}')

# ── 3. Generar ImageClassifier.py (solo usa torch, sin smp) ───────────────────
INFERENCE_CODE = r'''"""
ImageClassifier para ArcGIS Pro - DeepLabV3+ IGAC
Carga el modelo en formato TorchScript (.pt) - solo requiere torch.
"""
import os, json
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageClassifier:

    def __init__(self):
        self.name        = "DeepLabV3Plus IGAC Segmentation"
        self.description = "Segmentacion semantica 9 clases - IGAC Colombia"
        self.model       = None
        self.device      = None

    def initialize(self, **kwargs):
        model_arg = kwargs.get("model")
        try:
            with open(model_arg, "r", encoding="utf-8") as f:
                self.json_info = json.load(f)
            self.emd_dir = os.path.dirname(os.path.abspath(model_arg))
        except Exception:
            self.json_info = json.loads(model_arg)
            self.emd_dir   = ""

        model_file = self.json_info["ModelFile"].replace(".\\", "").replace("./", "")
        model_path = os.path.join(self.emd_dir, model_file)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"[IGAC] TorchScript cargado en {self.device}")

    def getParameterInfo(self):
        return [
            {"name": "raster",     "dataType": "raster",  "required": True,
             "displayName": "Raster de entrada"},
            {"name": "model",      "dataType": "string",  "required": True,
             "displayName": "Archivo EMD / DLPK"},
            {"name": "padding",    "dataType": "numeric", "required": False,
             "value": 64,  "displayName": "Padding (px)"},
            {"name": "batch_size", "dataType": "numeric", "required": False,
             "value": 4,   "displayName": "Batch size"},
        ]

    def getConfiguration(self, **scalars):
        self.padding    = int(scalars.get("padding",    64))
        self.batch_size = int(scalars.get("batch_size",  4))
        tile_h = self.json_info["ImageHeight"]
        tile_w = self.json_info["ImageWidth"]
        pad    = self.padding
        # Calcular rectangulo de tiles para el batch (raiz cuadrada)
        side = max(1, int(self.batch_size ** 0.5))
        self.rect_h = side
        self.rect_w = side
        return {
            "extractBands":      tuple(self.json_info["ExtractBands"]),
            "padding":           pad,
            "tx":                tile_w * self.rect_w,
            "ty":                tile_h * self.rect_h,
            "fixedTileSize":     1,
            "inheritProperties": 2 | 4 | 8,
            "inputMask":         True,
        }

    def updateRasterInfo(self, **kwargs):
        kwargs["output_info"]["bandCount"] = 1
        kwargs["output_info"]["pixelType"] = "i4"
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        px     = pixelBlocks["raster_pixels"].astype(np.float32)  # (C, H, W)
        mask   = pixelBlocks["raster_mask"]
        px[:, mask[0] == 0] = 0

        tile_h = self.json_info["ImageHeight"]
        tile_w = self.json_info["ImageWidth"]
        pad    = self.padding
        C, H, W = px.shape

        # Dividir super-tile en tiles individuales
        tiles, positions = [], []
        rh, rw = self.rect_h, self.rect_w
        for r in range(rh):
            for c in range(rw):
                y0 = r * (tile_h - 2 * pad)
                x0 = c * (tile_w - 2 * pad)
                tile = px[:, y0:y0+tile_h, x0:x0+tile_w]
                if tile.shape[1] < tile_h or tile.shape[2] < tile_w:
                    padded = np.zeros((C, tile_h, tile_w), dtype=np.float32)
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                tiles.append(tile)
                positions.append((y0, x0))

        # Normalizar
        batch = np.stack(tiles)           # (B, C, H, W)
        batch /= 255.0
        batch[:, 0] = (batch[:, 0] - MEAN[0]) / STD[0]
        batch[:, 1] = (batch[:, 1] - MEAN[1]) / STD[1]
        batch[:, 2] = (batch[:, 2] - MEAN[2]) / STD[2]

        # Inferencia
        t = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(t)                       # (B, 9, H, W)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32)  # (B, H, W)

        # Reensamblar resultado
        out_h = H - 2 * pad if pad > 0 else H
        out_w = W - 2 * pad if pad > 0 else W
        result = np.zeros((1, H, W), dtype=np.int32)

        for i, (y0, x0) in enumerate(positions):
            inner = preds[i]
            if pad > 0:
                inner = inner[pad:-pad, pad:-pad]
            ry0 = y0 + pad if pad > 0 else y0
            rx0 = x0 + pad if pad > 0 else x0
            ih, iw = inner.shape
            result[0, ry0:ry0+ih, rx0:rx0+iw] = inner

        pixelBlocks["output_pixels"] = result.astype(props["pixelType"], copy=False)
        return pixelBlocks
'''

inf_path = OUTPUT_DIR / 'ImageClassifier.py'
with open(inf_path, 'w', encoding='utf-8') as f:
    f.write(INFERENCE_CODE)
print(f'ImageClassifier.py generado: {inf_path}')

# ── 4. Empaquetar DLPK ────────────────────────────────────────────────────────
dlpk_path = Path(f'{DLPK_NAME}.dlpk')
with zipfile.ZipFile(dlpk_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(emd_path,  arcname=emd_path.name)
    zf.write(ts_path,   arcname=ts_path.name)
    zf.write(inf_path,  arcname=inf_path.name)

print(f'\nDLPK creado: {dlpk_path}  ({dlpk_path.stat().st_size / 1e6:.0f} MB)')
print('Contenido:')
with zipfile.ZipFile(dlpk_path) as zf:
    for f in zf.namelist():
        print(f'  {f}')
