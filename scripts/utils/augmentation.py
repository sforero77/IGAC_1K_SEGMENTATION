"""
Data Augmentation pipelines usando Albumentations

Define transformaciones para entrenamiento y validación
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    """
    Pipeline de augmentation para entrenamiento

    Incluye:
    - Transformaciones geométricas (aplicadas a imagen Y máscara)
    - Transformaciones fotométricas (solo imagen)

    Returns:
        albumentations.Compose
    """
    train_transform = A.Compose([
        # ========== TRANSFORMACIONES GEOMÉTRICAS ==========
        # Aplicadas a imagen Y máscara

        # Flips
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotación de 90 grados
        A.RandomRotate90(p=0.5),

        # Shift, Scale, Rotate
        A.ShiftScaleRotate(
            shift_limit=0.1,      # Shift hasta 10%
            scale_limit=0.1,      # Scale ±10%
            rotate_limit=15,      # Rotación ±15 grados
            border_mode=0,        # Border mode constant
            p=0.5
        ),

        # ========== TRANSFORMACIONES FOTOMÉTRICAS ==========
        # Solo aplicadas a la imagen (no a la máscara)

        # Brillo y contraste
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # ±20%
            contrast_limit=0.2,    # ±20%
            p=0.5
        ),

        # Hue, Saturation, Value
        A.HueSaturationValue(
            hue_shift_limit=10,    # ±10 en Hue
            sat_shift_limit=20,    # ±20 en Saturation
            val_shift_limit=10,    # ±10 en Value
            p=0.3
        ),

        # Gaussian Noise
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.2
        ),

        # Gaussian Blur
        A.GaussianBlur(
            blur_limit=(3, 7),
            p=0.2
        ),

        # ========== CONVERSIÓN A TENSOR ==========
        ToTensorV2()
    ])

    return train_transform


def get_validation_augmentation():
    """
    Pipeline para validación y test (sin augmentation)

    Solo convierte a tensor

    Returns:
        albumentations.Compose
    """
    val_transform = A.Compose([
        ToTensorV2()
    ])

    return val_transform


def get_light_augmentation():
    """
    Augmentation ligera (para debugging o entrenamientos rápidos)

    Returns:
        albumentations.Compose
    """
    light_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])

    return light_transform


def get_heavy_augmentation():
    """
    Augmentation agresiva (para datasets muy pequeños)

    ⚠️ CUIDADO: Puede cambiar la semántica de la imagen

    Returns:
        albumentations.Compose
    """
    heavy_transform = A.Compose([
        # Geométricas
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.7
        ),
        A.Transpose(p=0.5),

        # Fotométricas
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 100.0)),
            A.ISONoise(),
            A.MultiplicativeNoise(),
        ], p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.3),

        # Color
        A.OneOf([
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            A.ChannelShuffle(),
        ], p=0.2),

        ToTensorV2()
    ])

    return heavy_transform


def get_test_time_augmentation():
    """
    Test-Time Augmentation (TTA)

    Retorna lista de transformaciones para aplicar en inferencia
    y promediar las predicciones

    Returns:
        list de albumentations.Compose
    """
    tta_transforms = [
        # Original
        A.Compose([ToTensorV2()]),

        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            ToTensorV2()
        ]),

        # Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0),
            ToTensorV2()
        ]),

        # Rotate 90
        A.Compose([
            A.RandomRotate90(p=1.0),
            ToTensorV2()
        ]),

        # H + V flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            ToTensorV2()
        ]),
    ]

    return tta_transforms


def visualize_augmentation(dataset, idx=0, n_augmentations=9):
    """
    Visualiza augmentations aplicadas a un sample

    Args:
        dataset: SegmentationDataset con transform
        idx: Índice del sample
        n_augmentations: Número de augmentations a visualizar

    Returns:
        None (muestra plot)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Cargar sample original (sin transform)
    dataset_no_aug = dataset
    dataset_no_aug.transform = None
    image_orig, mask_orig = dataset_no_aug[idx]

    # Si es tensor, convertir a numpy
    if isinstance(image_orig, np.ndarray):
        image_np = image_orig
        mask_np = mask_orig
    else:
        image_np = image_orig.permute(1, 2, 0).numpy()
        mask_np = mask_orig.numpy()

    # Aplicar augmentations
    fig, axes = plt.subplots(3, n_augmentations, figsize=(20, 8))

    for i in range(n_augmentations):
        # Aplicar transform
        if dataset.transform:
            transformed = dataset.transform(image=image_np.copy(), mask=mask_np.copy())
            img_aug = transformed['image']
            mask_aug = transformed['mask']

            # Convertir tensor a numpy para visualización
            if isinstance(img_aug, np.ndarray):
                img_plot = img_aug
                mask_plot = mask_aug
            else:
                img_plot = img_aug.permute(1, 2, 0).numpy()
                mask_plot = mask_aug.numpy()
        else:
            img_plot = image_np
            mask_plot = mask_np

        # Desnormalizar imagen para visualización (opcional)
        # img_plot = (img_plot * std) + mean

        # Clip valores
        img_plot = np.clip(img_plot, -3, 3)  # Si está normalizado con z-score
        img_plot = (img_plot - img_plot.min()) / (img_plot.max() - img_plot.min())  # Normalizar a [0,1]

        # Plot
        axes[0, i].imshow(img_plot)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Aug {i+1}')

        axes[1, i].imshow(mask_plot, cmap='tab10', vmin=0, vmax=8)
        axes[1, i].axis('off')

        # Overlay
        axes[2, i].imshow(img_plot)
        axes[2, i].imshow(mask_plot, cmap='tab10', alpha=0.3, vmin=0, vmax=8)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=12)
    axes[1, 0].set_ylabel('Mask', fontsize=12)
    axes[2, 0].set_ylabel('Overlay', fontsize=12)

    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Visualización guardada en: augmentation_visualization.png")


if __name__ == "__main__":
    # Test de augmentation
    print("Testing augmentation pipelines...")

    # Get transforms
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()

    print("\nTrain transform:")
    print(train_transform)

    print("\nValidation transform:")
    print(val_transform)

    # Test con imagen dummy
    import numpy as np

    dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
    dummy_mask = np.random.randint(0, 9, (512, 512)).astype(np.uint8)

    transformed = train_transform(image=dummy_image, mask=dummy_mask)

    print(f"\nOriginal image shape: {dummy_image.shape}")
    print(f"Transformed image shape: {transformed['image'].shape}")
    print(f"Transformed mask shape: {transformed['mask'].shape}")

    print("\nTest exitoso!")
