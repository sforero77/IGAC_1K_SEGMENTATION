"""
PyTorch Dataset para patches de segmentación semántica

Carga patches normalizados (.npy) de imágenes y máscaras
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SegmentationDataset(Dataset):
    """
    Dataset para segmentación semántica

    Args:
        patches_dir: Path al directorio de patches (ej: data/processed/patches)
        split: 'train', 'val', o 'test'
        transform: Transformaciones de Albumentations (opcional)
        indices_csv: Path al CSV con índice de patches (opcional)
    """

    def __init__(self, patches_dir, split='train', transform=None, indices_csv=None):
        self.patches_dir = Path(patches_dir)
        self.split = split
        self.transform = transform

        # Directorios de imágenes y máscaras
        self.images_dir = self.patches_dir / split / 'images'
        self.masks_dir = self.patches_dir / split / 'masks'

        # Verificar que existan
        if not self.images_dir.exists():
            raise ValueError(f"Directorio de imágenes no existe: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Directorio de máscaras no existe: {self.masks_dir}")

        # Cargar índice de patches
        if indices_csv and os.path.exists(indices_csv):
            self.index_df = pd.read_csv(indices_csv)
            self.index_df = self.index_df[self.index_df['split'] == split]
            self.patch_ids = self.index_df['patch_id'].tolist()
        else:
            # Listar archivos directamente
            self.patch_ids = [
                f.stem for f in sorted(self.images_dir.glob('*.npy'))
            ]

        print(f"  Dataset {split}: {len(self.patch_ids)} patches")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        """
        Retorna un par (imagen, máscara)

        Returns:
            image: torch.Tensor (C, H, W) float32
            mask: torch.Tensor (H, W) long
        """
        patch_id = self.patch_ids[idx]

        # Cargar imagen (ya normalizada, float32, shape: H x W x 3)
        image_path = self.images_dir / f"{patch_id}.npy"
        image = np.load(image_path)

        # Cargar máscara (uint8, shape: H x W)
        mask_path = self.masks_dir / f"{patch_id}.npy"
        mask = np.load(mask_path)

        # Aplicar transformaciones (Albumentations)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Si no hay transform, convertir manualmente a tensors
        if not isinstance(image, torch.Tensor):
            # Albumentations ToTensorV2 ya hace esto, pero por si acaso
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # H x W x C -> C x H x W
            mask = torch.from_numpy(mask).long()
        else:
            # Asegurar que la máscara sea long (int64) para CrossEntropyLoss
            # ToTensorV2 puede retornar uint8 (Byte)
            mask = mask.long()

        return image, mask

    def get_patch_info(self, idx):
        """Retorna información del patch (útil para debugging)"""
        patch_id = self.patch_ids[idx]

        if hasattr(self, 'index_df'):
            info = self.index_df[self.index_df['patch_id'] == patch_id].iloc[0].to_dict()
        else:
            info = {'patch_id': patch_id}

        return info


class BalancedRandomSampler(torch.utils.data.Sampler):
    """
    Sampler que sobre-muestrea patches con clases minoritarias

    Útil para balancear el dataset durante entrenamiento
    """

    def __init__(self, dataset, indices_csv, minority_classes=[4, 7], oversample_factor=2.0):
        """
        Args:
            dataset: SegmentationDataset
            indices_csv: Path al CSV con metadata
            minority_classes: IDs de clases a sobre-muestrear
            oversample_factor: Factor de sobre-muestreo (2.0 = duplicar)
        """
        self.dataset = dataset
        self.minority_classes = minority_classes
        self.oversample_factor = oversample_factor

        # Cargar metadata
        df = pd.read_csv(indices_csv)
        df = df[df['split'] == dataset.split]

        # Identificar patches con clases minoritarias
        self.minority_indices = []
        self.regular_indices = []

        for idx, row in df.iterrows():
            classes_present = eval(row['classes_present'])
            has_minority = any(c in self.minority_classes for c in classes_present)

            if has_minority:
                self.minority_indices.append(idx)
            else:
                self.regular_indices.append(idx)

        print(f"    BalancedSampler: {len(self.minority_indices)} patches con clases minoritarias")
        print(f"    BalancedSampler: {len(self.regular_indices)} patches regulares")

    def __iter__(self):
        # Regular indices (1 vez cada uno)
        indices = self.regular_indices.copy()

        # Minority indices (sobre-muestrear)
        n_oversample = int(len(self.minority_indices) * self.oversample_factor)
        oversampled = np.random.choice(
            self.minority_indices,
            size=n_oversample,
            replace=True
        ).tolist()

        indices.extend(oversampled)

        # Shuffle
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        regular_count = len(self.regular_indices)
        minority_count = int(len(self.minority_indices) * self.oversample_factor)
        return regular_count + minority_count


def create_dataloaders(patches_dir, indices_csv, batch_size, num_workers=4,
                       train_transform=None, val_transform=None,
                       use_balanced_sampler=False):
    """
    Crea DataLoaders para train, val y test

    Args:
        patches_dir: Path a directorio de patches
        indices_csv: Path a patches_index.csv
        batch_size: Tamaño del batch
        num_workers: Número de workers para DataLoader
        train_transform: Transformaciones para train
        val_transform: Transformaciones para val/test
        use_balanced_sampler: Usar BalancedRandomSampler en train

    Returns:
        dict con 'train', 'val', 'test' DataLoaders
    """
    from torch.utils.data import DataLoader

    # Datasets
    train_dataset = SegmentationDataset(
        patches_dir,
        split='train',
        transform=train_transform,
        indices_csv=indices_csv
    )

    val_dataset = SegmentationDataset(
        patches_dir,
        split='val',
        transform=val_transform,
        indices_csv=indices_csv
    )

    test_dataset = SegmentationDataset(
        patches_dir,
        split='test',
        transform=val_transform,
        indices_csv=indices_csv
    )

    # Sampler para train (opcional)
    train_sampler = None
    if use_balanced_sampler:
        train_sampler = BalancedRandomSampler(
            train_dataset,
            indices_csv,
            minority_classes=[4, 7],  # ZonaVerde, Cerca
            oversample_factor=2.0
        )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # No shuffle si usamos sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last para evitar batch size inconsistente
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # Test del dataset
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import PATCHES_DIR, PATCHES_INDEX

    print("Testing SegmentationDataset...")

    # Crear dataset
    dataset = SegmentationDataset(
        patches_dir=PATCHES_DIR,
        split='train',
        transform=None
    )

    print(f"  Tamaño del dataset: {len(dataset)}")

    # Cargar primer sample
    image, mask = dataset[0]

    print(f"  Shape de imagen: {image.shape}")
    print(f"  Shape de máscara: {mask.shape}")
    print(f"  Tipo de imagen: {image.dtype}")
    print(f"  Tipo de máscara: {mask.dtype}")
    print(f"  Rango de imagen: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Clases en máscara: {torch.unique(mask)}")

    print("\nTest exitoso!")
