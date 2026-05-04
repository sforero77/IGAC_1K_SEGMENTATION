# Scripts - Estructura Organizada

Esta carpeta contiene todos los scripts del proyecto organizados por proceso.

## 📂 Estructura

```
scripts/
├── 01_data_preparation/    # Preparación del dataset
├── 02_training/             # Entrenamiento de modelos
├── 03_inference/            # Inferencia y predicciones
├── 04_evaluation/           # Evaluación de modelos
└── utils/                   # Utilidades compartidas
```

---

## 📁 01_data_preparation/

Scripts para procesar datos y preparar el dataset.

### Scripts Principales:

| Script | Descripción | Outputs |
|--------|-------------|---------|
| `03_create_spatial_splits.py` | Crea grilla espacial 1km² y splits (train/val/test) | `data/processed/spatial_splits/*.gpkg` |
| `04_calculate_normalization_stats.py` | Calcula estadísticas de normalización (μ, σ) de TRAIN | `data/interim/normalization_stats.json` |
| `06_create_patches.py` | Genera patches 512×512 con normalización on-the-fly | `data/processed/patches/{split}/` |
| `07_analyze_balance.py` | Analiza balance de clases y calcula weights | `data/interim/class_weights.json` |
| `consolidar_y_rasterizar.py` | Rasteriza vectores a máscaras .tif | `data/processed/rasters_etiquetas/*.tif` |

### Orden de Ejecución:

```bash
# 1. Rasterizar vectores (si no está hecho)
python scripts/01_data_preparation/consolidar_y_rasterizar.py

# 2. Crear partición espacial
python scripts/01_data_preparation/03_create_spatial_splits.py

# 3. Calcular stats de normalización
python scripts/01_data_preparation/04_calculate_normalization_stats.py

# 4. Crear patches
python scripts/01_data_preparation/06_create_patches.py

# 5. Analizar balance
python scripts/01_data_preparation/07_analyze_balance.py
```

---

## 📁 02_training/

Scripts para entrenar modelos de segmentación.

### Scripts Disponibles:

| Script | Modelo | Encoder | Batch Size | LR |
|--------|--------|---------|------------|-----|
| `train_unet.py` | U-Net | ResNet34 | 16 | 1e-4 |

### Uso:

```bash
# Entrenar U-Net
python scripts/02_training/train_unet.py
```

### Características:

- **Loss:** CrossEntropyLoss con class weights
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Early Stopping:** Patience=15 epochs
- **Checkpointing:** Guarda mejor modelo según mIoU
- **Logging:** TensorBoard
- **Métricas:** mIoU, IoU por clase, Pixel Accuracy

### Próximos Modelos:

- `train_deeplabv3.py` - DeepLabv3+ (ResNet50)
- `train_segformer.py` - SegFormer (Transformer-based)

---

## 📁 03_inference/

Scripts para inferencia en imágenes nuevas.

### Próximos Scripts:

- `predict_patch.py` - Predicción en un patch
- `predict_full_image.py` - Predicción en ortofoto completa con tiled inference
- `batch_predict.py` - Predicción en batch de imágenes

---

## 📁 04_evaluation/

Scripts para evaluar modelos entrenados.

### Próximos Scripts:

- `evaluate_model.py` - Evaluar modelo en test set
- `compare_models.py` - Comparar múltiples modelos
- `visualize_predictions.py` - Visualizar predicciones vs ground truth
- `error_analysis.py` - Análisis de errores por clase

---

## 📁 utils/

Módulos de utilidad compartidos por todos los scripts.

### Módulos:

| Módulo | Descripción | Principales Funciones/Clases |
|--------|-------------|------------------------------|
| `config.py` | Configuración centralizada | `get_device()`, `set_seed()`, `load_class_weights()`, constantes (BATCH_SIZE, LR, etc.) |
| `dataset.py` | PyTorch Dataset y DataLoader | `SegmentationDataset`, `BalancedRandomSampler`, `create_dataloaders()` |
| `augmentation.py` | Pipelines de augmentation | `get_training_augmentation()`, `get_validation_augmentation()`, `get_test_time_augmentation()` |
| `metrics.py` | Métricas de segmentación | `SegmentationMetrics`, `calculate_iou()`, `calculate_miou()` |

### Uso:

```python
# Importar desde utils
from utils import (
    get_device, set_seed, load_class_weights,
    create_dataloaders, get_training_augmentation,
    SegmentationMetrics
)

# Configurar
device = get_device()
set_seed(42)
class_weights = load_class_weights()

# Crear dataloaders
train_transform = get_training_augmentation()
dataloaders = create_dataloaders(
    patches_dir='data/processed/patches',
    indices_csv='data/processed/patches/patches_index.csv',
    batch_size=16,
    train_transform=train_transform
)

# Métricas
metrics = SegmentationMetrics(num_classes=9, ignore_index=0)
metrics.update(preds, targets)
miou = metrics.get_miou()
```

---

## 🔧 Configuración

### Editar Hiperparámetros:

Modificar `utils/config.py`:

```python
# Batch sizes
BATCH_SIZE_UNET = 16          # Ajustar según VRAM
BATCH_SIZE_DEEPLABV3 = 12
BATCH_SIZE_SEGFORMER = 8

# Learning rates
LR_UNET = 1e-4
LR_DEEPLABV3 = 1e-4
LR_SEGFORMER = 6e-5

# Training
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Hardware
NUM_WORKERS = 4               # Ajustar según CPU
MIXED_PRECISION = True        # AMP para ahorrar VRAM
```

### Variables de Entorno:

```bash
# GPU a usar
export CUDA_VISIBLE_DEVICES=0

# Número de workers
export NUM_WORKERS=4
```

---

## 📊 Archivos Generados

### Data:

```
data/
├── processed/
│   ├── patches/
│   │   ├── train/ (38,324 patches)
│   │   ├── val/   (15,625 patches)
│   │   └── test/  (18,991 patches)
│   └── spatial_splits/
│       ├── train_zones.gpkg
│       ├── val_zones.gpkg
│       └── test_zones.gpkg
└── interim/
    ├── normalization_stats.json
    └── class_weights.json
```

### Modelos:

```
models/
└── checkpoints/
    └── unet_baseline/
        ├── best_checkpoint.pth
        ├── last_checkpoint.pth
        └── training_history.json
```

### Logs:

```
logs/
└── tensorboard/
    └── unet_baseline/
        └── 20251112-143022/
            └── events.out.tfevents...
```

---

## 🚀 Quick Start

### 1. Preparar Dataset:

```bash
# Ejecutar todos los scripts de preparación
cd scripts/01_data_preparation
python 03_create_spatial_splits.py
python 04_calculate_normalization_stats.py
python 06_create_patches.py
python 07_analyze_balance.py
```

### 2. Entrenar U-Net:

```bash
# Entrenar modelo baseline
python scripts/02_training/train_unet.py

# Monitorear con TensorBoard
tensorboard --logdir=logs/tensorboard
```

### 3. Evaluar:

```bash
# (Próximamente)
python scripts/04_evaluation/evaluate_model.py --checkpoint models/checkpoints/unet_baseline/best_checkpoint.pth
```

---

## 📚 Dependencias

```bash
# Instaladas en requirements.txt
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3
albumentations>=1.3.1
tensorboard>=2.14.0
rasterio>=1.3.8
geopandas>=0.13.2
tqdm>=4.66.0
```

---

## 🐛 Troubleshooting

### Error: CUDA Out of Memory

```python
# Reducir batch size en config.py
BATCH_SIZE_UNET = 8  # En lugar de 16

# O usar gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 2
```

### Error: No se encuentra módulo utils

```bash
# Asegurarse de ejecutar desde la raíz del proyecto
cd D:\IGAC_1K_SEGMENTATION
python scripts/02_training/train_unet.py
```

### Error: Class weights no encontrados

```bash
# Ejecutar análisis de balance primero
python scripts/01_data_preparation/07_analyze_balance.py
```

---

## 📝 Notas

### Convenciones de Nombres:

- **Checkpoints:** `{modelo}_{experimento}/best_checkpoint.pth`
- **Logs:** `{modelo}_{experimento}/YYYYMMDD-HHMMSS/`
- **Patches:** `{ciudad}_{x_coord}_{y_coord}.npy`

### Buenas Prácticas:

1. **Siempre fijar seed:** `set_seed(42)` para reproducibilidad
2. **Monitorear VRAM:** Ajustar batch size según disponibilidad
3. **Usar TensorBoard:** Para visualizar entrenamiento en tiempo real
4. **Guardar checkpoints:** Cada 10 epochs + mejor modelo
5. **Early stopping:** Prevenir overfitting en dataset pequeño

---

**Última actualización:** Noviembre 12, 2025
**Estado:** ✅ Estructura completa - Scripts de entrenamiento listos
