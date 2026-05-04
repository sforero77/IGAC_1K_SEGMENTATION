"""
Utilities package for IGAC Segmentation Project

Contains:
- config: Centralized configuration
- dataset: PyTorch Dataset and DataLoader
- augmentation: Albumentations pipelines
- metrics: Segmentation metrics (IoU, mIoU, Dice, etc.)
- trainers: Training classes (BaseTrainer, UNetTrainer, DeepLabV3Trainer, UnetPlusPlusTrainer)
"""

from .config import (
    ROOT_DIR, DATA_DIR, PATCHES_DIR, PATCHES_INDEX, MODELS_DIR, LOGS_DIR,
    N_CLASSES, CLASS_NAMES, CLASS_COLORS,
    BATCH_SIZE_UNET, BATCH_SIZE_DEEPLABV3, BATCH_SIZE_SEGFORMER,
    LR_UNET, LR_DEEPLABV3, LR_SEGFORMER,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
    get_device, set_seed, load_class_weights, load_normalization_stats,
    get_model_config
)

from .dataset import (
    SegmentationDataset,
    BalancedRandomSampler,
    create_dataloaders
)

from .augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_light_augmentation,
    get_heavy_augmentation,
    get_test_time_augmentation
)

from .metrics import (
    SegmentationMetrics,
    calculate_iou,
    calculate_miou
)

from .trainers import (
    BaseTrainer,
    UNetTrainer,
    DeepLabV3Trainer,
    UnetPlusPlusTrainer
)

__all__ = [
    # Config
    'ROOT_DIR', 'DATA_DIR', 'PATCHES_DIR', 'PATCHES_INDEX', 'MODELS_DIR', 'LOGS_DIR',
    'N_CLASSES', 'CLASS_NAMES', 'CLASS_COLORS',
    'BATCH_SIZE_UNET', 'BATCH_SIZE_DEEPLABV3', 'BATCH_SIZE_SEGFORMER',
    'LR_UNET', 'LR_DEEPLABV3', 'LR_SEGFORMER',
    'MAX_EPOCHS', 'EARLY_STOPPING_PATIENCE',
    'get_device', 'set_seed', 'load_class_weights', 'load_normalization_stats',
    'get_model_config',
    # Dataset
    'SegmentationDataset', 'BalancedRandomSampler', 'create_dataloaders',
    # Augmentation
    'get_training_augmentation', 'get_validation_augmentation',
    'get_light_augmentation', 'get_heavy_augmentation',
    'get_test_time_augmentation',
    # Metrics
    'SegmentationMetrics', 'calculate_iou', 'calculate_miou',
    # Trainers
    'BaseTrainer', 'UNetTrainer', 'DeepLabV3Trainer', 'UnetPlusPlusTrainer',
]
