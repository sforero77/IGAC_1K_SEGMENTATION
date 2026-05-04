"""
Script de entrenamiento Unet++ optimizado para GPU

Características:
- Mixed Precision Training (reduce memoria 50%)
- Memory efficient optimizations
- Mejor manejo de errores CUDA
- Soporte para retomar desde checkpoint

Uso:
    python train_unetplusplus.py --epochs 100 --batch-size 8
    python train_unetplusplus.py --epochs 50 --batch-size 8 --resume
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.utils import (
    set_seed, get_device, load_class_weights, create_dataloaders,
    get_training_augmentation, get_validation_augmentation,
    UnetPlusPlusTrainer,
    N_CLASSES, PATCHES_DIR, PATCHES_INDEX, CHECKPOINTS_DIR
)


def main():
    # Argumentos
    parser = argparse.ArgumentParser(
        description='Entrenar Unet++ para segmentación de cartografía urbana'
    )
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8 para RTX 4060)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--resume', action='store_true',
                        help='Retomar desde último checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        help='Encoder backbone (default: resnet50)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers (default: 0 para Windows)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    print("\n" + "="*80)
    print("ENTRENAMIENTO UNET++ - CARTOGRAFÍA URBANA IGAC 1K")
    print("="*80)
    print(f"\n📊 Configuración:")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Patience: {args.patience}")
    print()

    # Cargar class weights
    print("📥 Cargando class weights...")
    class_weights = load_class_weights()
    print(f"✓ Class weights cargados: {N_CLASSES} clases")
    print()

    # Crear DataLoaders
    print("📥 Creando DataLoaders...")
    dataloaders = create_dataloaders(
        patches_dir=PATCHES_DIR,
        indices_csv=PATCHES_INDEX,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=get_training_augmentation(),
        val_transform=get_validation_augmentation(),
        use_balanced_sampler=False
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print(f"✓ DataLoaders creados:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print()

    # Crear modelo Unet++
    print("🏗️  Creando modelo Unet++...")
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=N_CLASSES
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Modelo Unet++ creado:")
    print(f"  Encoder: {args.encoder}")
    print(f"  Total parámetros: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    print(f"  Device: {device}")
    print()

    # Crear trainer
    print("⚙️  Creando trainer...")
    trainer = UnetPlusPlusTrainer(
        model=model,
        device=device,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        experiment_name='unetplusplus_resnet50_baseline',
        patience=args.patience,
        use_mixed_precision=True
    )
    print(f"✓ Trainer creado")
    print(f"  TensorBoard: {trainer.log_dir}")
    print()

    # Retomar desde checkpoint si existe
    start_epoch = 0
    checkpoint_path = CHECKPOINTS_DIR / 'unetplusplus_resnet50_baseline' / 'best_checkpoint.pth'
    if args.resume and checkpoint_path.exists():
        print("🔄 Retomando desde checkpoint...")
        start_epoch = trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint cargado (epoch {start_epoch + 1})")
        print(f"  Mejor mIoU: {trainer.best_val_miou:.4f}")
        print()

    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    print("="*80)

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )

        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"\n📊 Resultados Finales:")
        print(f"  Mejor mIoU: {trainer.best_val_miou:.4f}")
        print(f"  Mejor Loss: {min(history['val_loss']):.4f}")
        print()
        print(f"💾 Checkpoints guardados en:")
        print(f"  {CHECKPOINTS_DIR / 'unetplusplus_resnet50_baseline'}")
        print()

    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
        print(f"Mejor mIoU hasta ahora: {trainer.best_val_miou:.4f}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
