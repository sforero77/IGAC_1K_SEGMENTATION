"""
Script de diagnóstico: Entrenar DeepLabv3+ en CPU con batch_size reducido

Uso:
    python test_cpu_training.py --epochs 2 --batch-size 2

Propósito:
- Identificar si el problema es de GPU/CUDA o del código
- Si funciona en CPU: problema de hardware/memoria GPU
- Si falla en CPU: problema del código o de los datos
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import segmentation_models_pytorch as smp
from pathlib import Path

from scripts.utils import (
    set_seed, load_class_weights, create_dataloaders,
    get_training_augmentation, get_validation_augmentation,
    DeepLabV3Trainer, N_CLASSES,
    PATCHES_DIR, PATCHES_INDEX
)


def main(args):
    """Entrenar modelo en CPU para diagnóstico"""

    print("="*80)
    print("SCRIPT DE DIAGNÓSTICO: DeepLabv3+ en CPU")
    print("="*80)
    print()

    # 1. Configurar
    print("1. Configurando...")
    set_seed(42)
    device = torch.device('cpu')
    print(f"   Device: {device}")

    # 2. Cargar class weights
    print("\n2. Cargando class weights...")
    class_weights = load_class_weights()
    print(f"   ✓ Class weights cargados: {class_weights.shape}")

    # 3. Crear dataloaders
    print("\n3. Creando DataLoaders...")
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()

    dataloaders = create_dataloaders(
        patches_dir=PATCHES_DIR,
        indices_csv=PATCHES_INDEX,
        batch_size=args.batch_size,
        num_workers=0,
        train_transform=train_transform,
        val_transform=val_transform,
        use_balanced_sampler=False
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    print(f"   ✓ DataLoaders creados")
    print(f"     - Batch size: {args.batch_size}")
    print(f"     - Train batches: {len(train_loader)}")
    print(f"     - Val batches: {len(val_loader)}")

    # 4. Crear modelo
    print("\n4. Creando modelo DeepLabv3+...")
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=N_CLASSES
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Modelo creado: {total_params:,} parámetros")

    # 5. Crear trainer
    print("\n5. Creando Trainer...")
    experiment_name = f'deeplabv3_cpu_test_bs{args.batch_size}'
    trainer = DeepLabV3Trainer(
        model=model,
        device=device,
        class_weights=class_weights,
        lr=args.lr,
        weight_decay=1e-4,
        experiment_name=experiment_name,
        patience=args.epochs  # Sin early stopping para diagnóstico
    )
    print(f"   ✓ Trainer listo")

    # 6. Test rápido: forward pass en 1 batch
    print("\n6. Test rápido: Forward pass en 1 batch...")
    try:
        images, masks = next(iter(train_loader))
        images = images.to(device)
        masks = masks.to(device)

        print(f"   Batch size: {images.shape[0]}")
        print(f"   Image shape: {images.shape}")
        print(f"   Mask shape: {masks.shape}")

        with torch.no_grad():
            outputs = model(images)

        print(f"   ✓ Forward pass exitoso!")
        print(f"     - Output shape: {outputs.shape}")
    except Exception as e:
        print(f"   ✗ Error en forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Entrenar
    print("\n7. Iniciando entrenamiento de diagnóstico...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Device: {device}")
    print()

    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )

        print("\n✓ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"  Mejor mIoU: {trainer.best_val_miou:.4f}")
        print()
        print("CONCLUSIONES:")
        print("  Si llegaste aquí, el problema NO es del código.")
        print("  El problema es probablemente:")
        print("  - Memoria insuficiente en GPU (VRAM)")
        print("  - Problema de drivers CUDA")
        print("  - Problema de hardware GPU")
        print()
        print("SOLUCIONES:")
        print("  1. Reducir batch size en GPU (prueba con batch_size=4)")
        print("  2. Actualizar drivers NVIDIA")
        print("  3. Verificar que CUDA y cuDNN están correctamente instalados")
        print("  4. Probar con CUDA_LAUNCH_BLOCKING=1 para mejor diagnóstico")

        return True

    except Exception as e:
        print(f"\n✗ ERROR DURANTE ENTRENAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("CONCLUSIONES:")
        print("  El problema está en el CÓDIGO o en los DATOS.")
        print("  No es un problema de GPU/CUDA.")
        print()
        print("DEBUG:")
        print("  1. Revisar el traceback arriba")
        print("  2. Verificar que los datos están correctamente cargados")
        print("  3. Revisar scripts/utils/metrics.py línea 51")

        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diagnóstico de entrenamiento DeepLabv3+ en CPU'
    )
    parser.add_argument('--epochs', type=int, default=2,
                        help='Número de epochs a entrenar (default: 2)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')

    args = parser.parse_args()

    success = main(args)
    sys.exit(0 if success else 1)
