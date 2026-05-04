"""
Configuración centralizada del proyecto

Contiene todos los paths, hiperparámetros y constantes usadas en el proyecto
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent.parent

# Datos
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Patches
PATCHES_DIR = PROCESSED_DIR / "patches"
PATCHES_INDEX = PATCHES_DIR / "patches_index.csv"

# Archivos de configuración
NORMALIZATION_STATS = INTERIM_DIR / "normalization_stats.json"
CLASS_WEIGHTS_FILE = INTERIM_DIR / "class_weights.json"

# Modelos y resultados
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Logs
LOGS_DIR = ROOT_DIR / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

# Crear directorios si no existen
for dir_path in [MODELS_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR, TENSORBOARD_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

# Ciudades
CIUDADES = ['madrid', 'mosquera', 'tabio', 'tocancipa']

# Clases
N_CLASSES = 9  # 0-8

CLASS_NAMES = {
    0: 'Background',
    1: 'Construccion',
    2: 'Bosque',
    3: 'DepositoAgua',
    4: 'ZonaVerde',
    5: 'ZonaDura',
    6: 'Via',
    7: 'Cerca',
    8: 'CabeceraMunicipal'
}

# Mapeo inverso
CLASS_NAMES_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

# Colores para visualización (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),           # Background - Negro
    1: (255, 0, 0),         # Construccion - Rojo
    2: (0, 128, 0),         # Bosque - Verde oscuro
    3: (0, 0, 255),         # DepositoAgua - Azul
    4: (0, 255, 0),         # ZonaVerde - Verde claro
    5: (128, 128, 128),     # ZonaDura - Gris
    6: (64, 64, 64),        # Via - Gris oscuro
    7: (139, 69, 19),       # Cerca - Café
    8: (255, 255, 0),       # CabeceraMunicipal - Amarillo
}


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# General
RANDOM_SEED = 42
DEVICE = 'cuda'  # 'cuda' o 'cpu'

# Batch sizes (ajustar según GPU VRAM)
BATCH_SIZE_UNET = 16          # U-Net (8-24)
BATCH_SIZE_DEEPLABV3 = 12     # DeepLabv3+ (8-16)
BATCH_SIZE_SEGFORMER = 8      # SegFormer (4-8)

# Learning rates
LR_UNET = 1e-4
LR_DEEPLABV3 = 1e-4
LR_SEGFORMER = 6e-5

# Optimizer
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Training
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Scheduler
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', 'plateau'
SCHEDULER_PATIENCE = 5     # Para ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5     # Para ReduceLROnPlateau
T_0 = 10                   # Para CosineAnnealingWarmRestarts
T_MULT = 2                 # Para CosineAnnealingWarmRestarts

# DataLoader
NUM_WORKERS = 4            # Número de workers para DataLoader (ajustar según CPU)
PIN_MEMORY = True

# Checkpointing
SAVE_EVERY_N_EPOCHS = 10
SAVE_BEST_ONLY = True


# ============================================================================
# AUGMENTATION
# ============================================================================

# Probabilidades de augmentation
AUG_HORIZONTAL_FLIP_P = 0.5
AUG_VERTICAL_FLIP_P = 0.5
AUG_ROTATE90_P = 0.5
AUG_SHIFT_SCALE_ROTATE_P = 0.5
AUG_BRIGHTNESS_CONTRAST_P = 0.5
AUG_HUE_SAT_VAL_P = 0.3
AUG_GAUSS_NOISE_P = 0.2
AUG_GAUSSIAN_BLUR_P = 0.2

# Límites de augmentation
AUG_SHIFT_LIMIT = 0.1
AUG_SCALE_LIMIT = 0.1
AUG_ROTATE_LIMIT = 15
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2
AUG_HUE_SHIFT_LIMIT = 10
AUG_SAT_SHIFT_LIMIT = 20
AUG_VAL_SHIFT_LIMIT = 10


# ============================================================================
# METRICS
# ============================================================================

# IoU thresholds (para calcular métricas adicionales)
IOU_THRESHOLDS = [0.5, 0.75, 0.9]

# Clases a ignorar en métricas
IGNORE_INDEX = 0  # Background

# Clases críticas (para monitoreo especial)
CRITICAL_CLASSES = [4, 7]  # ZonaVerde, Cerca (minoritarias)


# ============================================================================
# INFERENCE
# ============================================================================

# Overlap para tiled inference
INFERENCE_OVERLAP = 0.25  # 25%

# Post-processing
MORPHOLOGICAL_OPENING_KERNEL = 2
MORPHOLOGICAL_CLOSING_KERNEL = 3
MIN_AREA_THRESHOLD = 100  # píxeles


# ============================================================================
# LOGGING
# ============================================================================

# TensorBoard
USE_TENSORBOARD = True

# Weights & Biases
USE_WANDB = False
WANDB_PROJECT = "igac-segmentation"
WANDB_ENTITY = None  # Tu username de W&B

# Logging frequency
LOG_EVERY_N_STEPS = 50
VISUALIZE_EVERY_N_EPOCHS = 10


# ============================================================================
# HARDWARE
# ============================================================================

# GPU settings
GPU_IDS = [0]  # Lista de GPUs a usar (ej: [0, 1] para multi-GPU)
MIXED_PRECISION = True  # Usar AMP (Automatic Mixed Precision) para ahorrar VRAM

# Memory optimization
GRADIENT_ACCUMULATION_STEPS = 1  # Aumentar si hay OOM


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_device():
    """Retorna el device apropiado (cuda o cpu)"""
    import torch
    if DEVICE == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed=RANDOM_SEED):
    """Fija todas las semillas aleatorias para reproducibilidad"""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_class_weights():
    """Carga los class weights desde JSON"""
    import json
    import torch

    with open(CLASS_WEIGHTS_FILE, 'r') as f:
        data = json.load(f)

    # Usar inverse_freq_normalized (recomendado)
    weights = data['weights']['inverse_freq_normalized']

    return torch.tensor(weights, dtype=torch.float32)


def load_normalization_stats():
    """Carga las estadísticas de normalización desde JSON"""
    import json
    import numpy as np

    with open(NORMALIZATION_STATS, 'r') as f:
        data = json.load(f)

    mean = np.array(data['mean'], dtype=np.float32)
    std = np.array(data['std'], dtype=np.float32)

    return mean, std


def get_model_config(model_name):
    """
    Retorna la configuración específica para un modelo

    Args:
        model_name: 'unet', 'deeplabv3', 'segformer'

    Returns:
        dict con configuración
    """
    configs = {
        'unet': {
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'batch_size': BATCH_SIZE_UNET,
            'lr': LR_UNET,
            'name': 'U-Net'
        },
        'deeplabv3': {
            'encoder_name': 'resnet50',
            'encoder_weights': 'imagenet',
            'batch_size': BATCH_SIZE_DEEPLABV3,
            'lr': LR_DEEPLABV3,
            'name': 'DeepLabv3+'
        },
        'segformer': {
            'model_name': 'nvidia/segformer-b3-finetuned-ade-512-512',
            'batch_size': BATCH_SIZE_SEGFORMER,
            'lr': LR_SEGFORMER,
            'name': 'SegFormer'
        }
    }

    return configs.get(model_name.lower(), None)


if __name__ == "__main__":
    # Test de configuración
    print("Configuración del proyecto:")
    print(f"  ROOT_DIR: {ROOT_DIR}")
    print(f"  PATCHES_DIR: {PATCHES_DIR}")
    print(f"  N_CLASSES: {N_CLASSES}")
    print(f"  BATCH_SIZE_UNET: {BATCH_SIZE_UNET}")
    print(f"  DEVICE: {get_device()}")

    # Test de carga de weights
    try:
        weights = load_class_weights()
        print(f"\n  Class weights cargados: {weights}")
    except:
        print("\n  Class weights no disponibles aún")
