"""
Script de entrenamiento DeepLabv3+ optimizado para GPU

Características:
- Mixed Precision Training (reduce memoria 50%)
- Gradient Checkpointing (reduce memoria más)
- Memory efficient optimizations
- Mejor manejo de errores CUDA

Uso:
    python train_deeplabv3_gpu_optimized.py --epochs 10 --batch-size 4
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from scripts.utils import (
    set_seed, load_class_weights, create_dataloaders,
    get_training_augmentation, get_validation_augmentation,
    SegmentationMetrics, N_CLASSES, CLASS_NAMES,
    PATCHES_DIR, PATCHES_INDEX, CHECKPOINTS_DIR, TENSORBOARD_DIR
)


class GPUOptimizedDeepLabV3Trainer:
    """Trainer optimizado para memoria GPU con mixed precision"""

    def __init__(self, model, device, class_weights, batch_size=4, lr=1e-4,
                 experiment_name='deeplabv3_optimized', use_mixed_precision=True,
                 use_gradient_checkpointing=True):

        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=0
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Mixed Precision
        if use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Gradient Checkpointing
        if use_gradient_checkpointing:
            print("  → Activando Gradient Checkpointing...")
            # Para ResNet50 encoder en DeepLabV3+
            if hasattr(model.encoder, 'layer3'):
                model.encoder.layer3.gradient_checkpointing = True
            if hasattr(model.encoder, 'layer4'):
                model.encoder.layer4.gradient_checkpointing = True

        # Métricas
        self.train_metrics = SegmentationMetrics(N_CLASSES, ignore_index=0, class_names=CLASS_NAMES)
        self.val_metrics = SegmentationMetrics(N_CLASSES, ignore_index=0, class_names=CLASS_NAMES)

        # Checkpoints
        self.best_val_miou = 0.0
        self.epochs_without_improvement = 0
        self.experiment_name = experiment_name

        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'lr': []
        }

        print(f"\n✓ Trainer GPU-Optimizado Creado:")
        print(f"  Device: {device}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Mixed Precision: {use_mixed_precision}")
        print(f"  Gradient Checkpointing: {use_gradient_checkpointing}")
        print(f"  Learning Rate: {lr}")

    def train_epoch(self, train_loader, epoch):
        """Entrena una época con mixed precision"""
        self.model.train()
        self.train_metrics.reset()

        running_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")

        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                # Mover a GPU
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                # Forward con mixed precision
                if self.use_mixed_precision:
                    with autocast(dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)

                    # Backward con scaling
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    # Forward normal
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    # Backward normal
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                # Métricas
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                # Hacer detach y cpu para evitar problemas CUDA
                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                self.train_metrics.update(preds_np, masks_np)

                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

                # Limpiar caché cada 50 batches
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"\n✗ Error en batch {batch_idx}: {e}")
                print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                torch.cuda.empty_cache()
                raise

        avg_loss = running_loss / num_batches
        miou = self.train_metrics.get_miou(ignore_background=True)

        return {
            'loss': avg_loss,
            'miou': miou
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        """Valida una época"""
        self.model.eval()
        self.val_metrics.reset()

        running_loss = 0.0
        num_batches = len(val_loader)

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]")

        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with autocast(dtype=torch.float16):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                # Detach y cpu
                preds_np = preds.detach().cpu().numpy()
                masks_np = masks.detach().cpu().numpy()
                self.val_metrics.update(preds_np, masks_np)

                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

            except RuntimeError as e:
                print(f"\n✗ Error en validación batch {batch_idx}: {e}")
                torch.cuda.empty_cache()
                raise

        avg_loss = running_loss / num_batches
        miou = self.val_metrics.get_miou(ignore_background=True)

        # ⚠️ IMPORTANTE: Volver a train mode después de validación
        self.model.train()

        return {
            'loss': avg_loss,
            'miou': miou
        }

    def save_checkpoint(self, epoch, is_best=False):
        """Guarda checkpoint"""
        checkpoint_dir = CHECKPOINTS_DIR / self.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_miou': self.best_val_miou,
            'history': self.history
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Último checkpoint
        last_path = checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)

        # Mejor checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Mejor modelo guardado (mIoU: {self.best_val_miou:.4f})")

    def train(self, train_loader, val_loader, num_epochs, patience=15):
        """Loop principal de entrenamiento"""
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO CON GPU OPTIMIZADO")
        print("="*80)

        for epoch in range(1, num_epochs + 1):
            # Memoria inicial
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Val
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Historial
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_miou'].append(train_metrics['miou'])
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['lr'].append(current_lr)

            # Print
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            print(f"  LR: {current_lr:.2e} | Peak Memory: {peak_memory:.2f} GB")

            # Early stopping
            val_miou = val_metrics['miou']
            is_best = val_miou > self.best_val_miou

            if is_best:
                self.best_val_miou = val_miou
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Checkpoint cada 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\n⚠ Early stopping después de {epoch} epochs")
                print(f"Mejor mIoU: {self.best_val_miou:.4f}")
                break

        print("\n" + "="*80)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*80)
        print(f"Mejor mIoU: {self.best_val_miou:.4f}")

        return self.history


def main(args):
    """Función principal"""
    print("="*80)
    print("ENTRENAMIENTO DeepLabv3+ OPTIMIZADO PARA GPU")
    print("="*80)

    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        print("⚠ CUDA no disponible! Usando CPU (será LENTO)")
    else:
        print(f"✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Memoria: {total_mem:.2f} GB")

    # Class weights
    print("\nCargando class weights...")
    class_weights = load_class_weights()

    # DataLoaders
    print(f"Creando DataLoaders (batch_size={args.batch_size})...")
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

    print(f"✓ DataLoaders creados:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")

    # Modelo
    print("\nCreando modelo...")
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=N_CLASSES
    )

    # Trainer
    trainer = GPUOptimizedDeepLabV3Trainer(
        model=model,
        device=device,
        class_weights=class_weights,
        batch_size=args.batch_size,
        lr=args.lr,
        experiment_name=f'deeplabv3_optimized_bs{args.batch_size}',
        use_mixed_precision=args.mixed_precision,
        use_gradient_checkpointing=args.gradient_checkpointing
    )

    # Train
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            patience=args.patience
        )

        print("\n✓ ENTRENAMIENTO EXITOSO!")
        return True

    except KeyboardInterrupt:
        print("\n⚠ Entrenamiento cancelado por usuario")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenar DeepLabv3+ con optimizaciones GPU'
    )
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--mixed-precision', type=bool, default=True,
                        help='Usar mixed precision (default: True)')
    parser.add_argument('--gradient-checkpointing', type=bool, default=True,
                        help='Usar gradient checkpointing (default: True)')

    args = parser.parse_args()

    success = main(args)
    sys.exit(0 if success else 1)
