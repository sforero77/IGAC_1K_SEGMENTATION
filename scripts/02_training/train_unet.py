"""
Script CLI para entrenar U-Net

Usa la clase UNetTrainer de utils.trainers

Uso:
    python scripts/02_training/train_unet.py
"""

import sys
from pathlib import Path

# Agregar utils al path
sys.path.append(str(Path(__file__).parent.parent))

import segmentation_models_pytorch as smp

from utils import (
    get_device, set_seed, load_class_weights,
    create_dataloaders, get_training_augmentation, get_validation_augmentation,
    UNetTrainer, N_CLASSES,
    PATCHES_DIR, PATCHES_INDEX, BATCH_SIZE_UNET, LR_UNET,
    MAX_EPOCHS, NUM_WORKERS
)


def main():
    """Función principal"""
    print("="*80)
    print("ENTRENAMIENTO U-NET - IGAC SEGMENTACIÓN")
    print("="*80)

    # Configuración
    set_seed(42)
    device = get_device()

    # Cargar class weights
    print("\n✓ Cargando class weights...")
    class_weights = load_class_weights()
    print(f"  Weights: {class_weights.numpy()}")

    # Crear dataloaders
    print("\n✓ Creando dataloaders...")
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()

    dataloaders = create_dataloaders(
        patches_dir=PATCHES_DIR,
        indices_csv=PATCHES_INDEX,
        batch_size=BATCH_SIZE_UNET,
        num_workers=NUM_WORKERS,
        train_transform=train_transform,
        val_transform=val_transform,
        use_balanced_sampler=False
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Crear modelo U-Net
    print("\n✓ Creando modelo U-Net...")
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=N_CLASSES
    )

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parámetros: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")

    # Crear trainer
    print("\n✓ Creando trainer...")
    trainer = UNetTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        lr=LR_UNET,
        experiment_name='unet_baseline'
    )

    # Entrenar
    history = trainer.train(train_loader, val_loader, num_epochs=MAX_EPOCHS)

    print("\n✓ Entrenamiento finalizado exitosamente!")
    print(f"\nMejor mIoU alcanzado: {trainer.best_val_miou:.4f}")
    print(f"Checkpoint guardado en: models/checkpoints/unet_baseline/")


if __name__ == "__main__":
    main()
