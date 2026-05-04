"""
Empaqueta el modelo DeepLabV3+ como DLPK usando arcgis.learn.

Requisito previo:
    conda install -c esri arcgis -y

Uso:
    python scripts/exportar_dlpk_arcgislearn.py
"""

import sys, os
sys.path.insert(0, os.path.abspath('.'))

import torch
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.utils import CHECKPOINTS_DIR, N_CLASSES, CLASS_NAMES, CLASS_COLORS

# ── Cargar modelo ──────────────────────────────────────────────────────────────
print('Cargando modelo...')
device = torch.device('cpu')
ckpt_path = CHECKPOINTS_DIR / 'deeplabv3_resnet50_baseline' / 'best_checkpoint.pth'

model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                          in_channels=3, classes=N_CLASSES)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Modelo listo — epoch {ckpt["epoch"]}, val mIoU={ckpt["best_val_miou"]:.4f}')

# ── Empaquetar con arcgis.learn ────────────────────────────────────────────────
from arcgis.learn import PixelClassifier

# Clases y colores
class_names  = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
class_colors = [CLASS_COLORS.get(i, [0,0,0]) for i in sorted(CLASS_NAMES.keys())]

print('Creando PixelClassifier...')
learner = PixelClassifier.from_model(
    emd_path=None,
    data=None
)

# Reemplazar el backbone con nuestro modelo entrenado
learner.learn.model = model

# Exportar como DLPK
out_path = Path('deeplabv3plus_igac_arcgislearn')
learner.save(str(out_path))
print(f'DLPK guardado en: {out_path}.dlpk')
