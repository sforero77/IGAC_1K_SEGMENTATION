"""
Clases Trainer para entrenamiento de modelos de segmentación

Incluye:
- BaseTrainer: Clase base con funcionalidades comunes
- UNetTrainer: Trainer específico para U-Net
- DeepLabV3Trainer: Trainer para DeepLabv3+
- UnetPlusPlusTrainer: Trainer para Unet++
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

from .metrics import SegmentationMetrics
from .config import (
    N_CLASSES, CLASS_NAMES, CHECKPOINTS_DIR, TENSORBOARD_DIR,
    EARLY_STOPPING_PATIENCE
)


class BaseTrainer:
    """
    Clase base para entrenamiento de modelos de segmentación

    Args:
        model: Modelo de PyTorch
        device: Device (cuda o cpu)
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (opcional)
        experiment_name: Nombre del experimento
        patience: Epochs sin mejora para early stopping
    """

    def __init__(self, model, device, criterion, optimizer,
                 scheduler=None, experiment_name='experiment',
                 patience=EARLY_STOPPING_PATIENCE,
                 use_mixed_precision=True, accumulation_steps=1):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_name = experiment_name
        self.patience = patience

        # Mixed Precision y Gradient Accumulation
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        self.accumulation_steps = accumulation_steps
        if self.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Métricas
        self.train_metrics = SegmentationMetrics(N_CLASSES, ignore_index=0, class_names=CLASS_NAMES)
        self.val_metrics = SegmentationMetrics(N_CLASSES, ignore_index=0, class_names=CLASS_NAMES)

        # TensorBoard
        log_dir = TENSORBOARD_DIR / experiment_name / datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        # Early stopping
        self.best_val_miou = 0.0
        self.epochs_without_improvement = 0

        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_miou': [],
            'val_miou': [],
            'lr': []
        }

        print(f"✓ Trainer inicializado:")
        print(f"  - Device: {device}")
        print(f"  - Patience: {self.patience}")
        print(f"  - Mixed Precision: {'✓ Enabled' if self.use_mixed_precision else '✗ Disabled'}")
        print(f"  - Gradient Accumulation: {self.accumulation_steps} steps")
        print(f"  - TensorBoard: {log_dir}")

    def train_epoch(self, train_loader, epoch):
        """
        Entrena una época con soporte para Mixed Precision y Gradient Accumulation

        Args:
            train_loader: DataLoader de entrenamiento
            epoch: Número de época actual

        Returns:
            Dict con métricas del epoch
        """
        self.model.train()
        self.train_metrics.reset()

        running_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")

        for batch_idx, (images, masks) in enumerate(pbar):
            # Mover a device
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass con Mixed Precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation: actualizar pesos cada N batches
            is_last_batch = (batch_idx + 1) == num_batches
            if (batch_idx + 1) % self.accumulation_steps == 0 or is_last_batch:
                if self.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Actualizar métricas (usar loss acumulado)
            running_loss += loss.item() * self.accumulation_steps
            preds = torch.argmax(outputs, dim=1)
            self.train_metrics.update(preds, masks)

            # Actualizar progress bar
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        # Métricas del epoch
        avg_loss = running_loss / num_batches
        miou = self.train_metrics.get_miou(ignore_background=True)

        return {
            'loss': avg_loss,
            'miou': miou
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        """
        Valida una época con soporte para Mixed Precision

        Args:
            val_loader: DataLoader de validación
            epoch: Número de época actual

        Returns:
            Dict con métricas del epoch
        """
        self.model.eval()
        self.val_metrics.reset()

        running_loss = 0.0
        num_batches = len(val_loader)

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]")

        for batch_idx, (images, masks) in enumerate(pbar):
            # Mover a device
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass con Mixed Precision
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Actualizar métricas
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            self.val_metrics.update(preds, masks)

            # Actualizar progress bar
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        # Métricas del epoch
        avg_loss = running_loss / num_batches
        miou = self.val_metrics.get_miou(ignore_background=True)

        # ⚠️ IMPORTANTE: Volver a train mode después de validación
        # para asegurar que el siguiente epoch funcione correctamente
        self.model.train()

        return {
            'loss': avg_loss,
            'miou': miou
        }

    def save_checkpoint(self, epoch, is_best=False):
        """
        Guarda checkpoint del modelo

        Args:
            epoch: Época actual
            is_best: Si es el mejor modelo hasta ahora
        """
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

        # Guardar último checkpoint
        last_path = checkpoint_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, last_path)

        # Guardar mejor checkpoint
        if is_best:
            best_path = checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Mejor modelo guardado (mIoU: {self.best_val_miou:.4f})")

    def load_checkpoint(self, checkpoint_path):
        """
        Carga checkpoint para continuar entrenamiento

        Args:
            checkpoint_path: Path al checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_val_miou = checkpoint.get('best_val_miou', 0.0)
        self.history = checkpoint.get('history', self.history)

        epoch = checkpoint.get('epoch', 0)
        print(f"✓ Checkpoint cargado desde epoch {epoch}")

        return epoch

    def log_metrics(self, epoch, train_metrics, val_metrics):
        """
        Registra métricas en TensorBoard

        Args:
            epoch: Época actual
            train_metrics: Dict con métricas de train
            val_metrics: Dict con métricas de val
        """
        # Loss
        self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)

        # mIoU
        self.writer.add_scalar('mIoU/train', train_metrics['miou'], epoch)
        self.writer.add_scalar('mIoU/val', val_metrics['miou'], epoch)

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_rate', current_lr, epoch)

        # IoU por clase (validación)
        iou_per_class = self.val_metrics.get_iou_per_class()
        for i in range(1, N_CLASSES):  # Excluir background
            class_name = CLASS_NAMES[i]
            self.writer.add_scalar(f'IoU_per_class/{class_name}', iou_per_class[i], epoch)

    def train(self, train_loader, val_loader, num_epochs):
        """
        Loop principal de entrenamiento

        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número de épocas a entrenar

        Returns:
            Dict con historial de entrenamiento
        """
        print("\n" + "="*80)
        print("INICIANDO ENTRENAMIENTO")
        print("="*80)

        for epoch in range(1, num_epochs + 1):
            # Entrenar
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validar
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Actualizar scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_miou'].append(train_metrics['miou'])
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['lr'].append(current_lr)

            self.log_metrics(epoch, train_metrics, val_metrics)

            # Imprimir resumen
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}")
            print(f"  LR: {current_lr:.2e}")

            # Early stopping y checkpoint
            val_miou = val_metrics['miou']
            is_best = val_miou > self.best_val_miou

            if is_best:
                self.best_val_miou = val_miou
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Guardar último checkpoint cada 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Verificar early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⚠ Early stopping activado después de {epoch} epochs")
                print(f"Mejor mIoU: {self.best_val_miou:.4f}")
                break

        # Guardar historial
        self.save_training_history()

        # Cerrar TensorBoard
        self.writer.close()

        print("\n" + "="*80)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*80)
        print(f"Mejor mIoU: {self.best_val_miou:.4f}")

        return self.history

    def save_training_history(self):
        """Guarda historial de entrenamiento en JSON"""
        history_dir = CHECKPOINTS_DIR / self.experiment_name
        history_path = history_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n✓ Historial guardado en: {history_path}")


class UNetTrainer(BaseTrainer):
    """
    Trainer específico para U-Net

    Args:
        model: Modelo U-Net
        device: Device (cuda o cpu)
        class_weights: Weights para loss function
        lr: Learning rate
        weight_decay: Weight decay para optimizer
        experiment_name: Nombre del experimento
        patience: Epochs sin mejora para early stopping
        use_mixed_precision: Usar Mixed Precision (FP16)
        accumulation_steps: Gradient accumulation steps
    """

    def __init__(self, model, device, class_weights, lr=1e-4,
                 weight_decay=1e-4, experiment_name='unet',
                 patience=EARLY_STOPPING_PATIENCE,
                 use_mixed_precision=True, accumulation_steps=1):

        # Loss function con class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=0  # Ignorar background
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Reiniciar cada 10 epochs
            T_mult=2,  # Duplicar período cada reinicio
            eta_min=1e-6
        )

        # Inicializar base trainer
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment_name=experiment_name,
            patience=patience,
            use_mixed_precision=use_mixed_precision,
            accumulation_steps=accumulation_steps
        )

        print(f"  - Arquitectura: U-Net")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler: CosineAnnealingWarmRestarts")


class DeepLabV3Trainer(BaseTrainer):
    """
    Trainer específico para DeepLabv3+

    Similar a UNetTrainer pero con configuraciones específicas

    Args:
        model: Modelo DeepLabv3+
        device: Device (cuda o cpu)
        class_weights: Weights para loss function
        lr: Learning rate
        weight_decay: Weight decay para optimizer
        experiment_name: Nombre del experimento
        patience: Epochs sin mejora para early stopping
        use_mixed_precision: Usar Mixed Precision (FP16)
        accumulation_steps: Gradient accumulation steps
    """

    def __init__(self, model, device, class_weights, lr=1e-4,
                 weight_decay=1e-4, experiment_name='deeplabv3',
                 patience=EARLY_STOPPING_PATIENCE,
                 use_mixed_precision=True, accumulation_steps=1):

        # Loss function con class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Inicializar base trainer
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment_name=experiment_name,
            patience=patience,
            use_mixed_precision=use_mixed_precision,
            accumulation_steps=accumulation_steps
        )

        print(f"  - Arquitectura: DeepLabv3+")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler: CosineAnnealingWarmRestarts")


class UnetPlusPlusTrainer(BaseTrainer):
    """
    Trainer específico para Unet++

    Unet++ es una versión mejorada de U-Net con conexiones anidadas
    que mejoran el flujo de gradientes y la precisión.

    Args:
        model: Modelo Unet++
        device: Device (cuda o cpu)
        class_weights: Weights para loss function
        lr: Learning rate
        weight_decay: Weight decay para optimizer
        experiment_name: Nombre del experimento
        patience: Epochs sin mejora para early stopping
        use_mixed_precision: Usar Mixed Precision (FP16)
        accumulation_steps: Gradient accumulation steps
    """

    def __init__(self, model, device, class_weights, lr=1e-4,
                 weight_decay=1e-4, experiment_name='unetplusplus',
                 patience=EARLY_STOPPING_PATIENCE,
                 use_mixed_precision=True, accumulation_steps=1):

        # Loss function con class weights
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            ignore_index=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Inicializar base trainer
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment_name=experiment_name,
            patience=patience,
            use_mixed_precision=use_mixed_precision,
            accumulation_steps=accumulation_steps
        )

        print(f"  - Arquitectura: Unet++")
        print(f"  - Learning rate: {lr}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Scheduler: CosineAnnealingWarmRestarts")


if __name__ == "__main__":
    print("Este módulo contiene clases Trainer para entrenamiento.")
    print("Importar y usar en scripts o notebooks.")
    print("\nEjemplo:")
    print("""
    from utils.trainers import UNetTrainer, DeepLabV3Trainer, UnetPlusPlusTrainer
    import segmentation_models_pytorch as smp

    # Crear modelo U-Net
    model_unet = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                          in_channels=3, classes=9)
    trainer_unet = UNetTrainer(model_unet, device, class_weights, lr=1e-4,
                               experiment_name='unet_baseline')

    # Crear modelo DeepLabv3+
    model_deeplab = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet',
                                      in_channels=3, classes=9)
    trainer_deeplab = DeepLabV3Trainer(model_deeplab, device, class_weights, lr=1e-4,
                                       experiment_name='deeplabv3_baseline')

    # Crear modelo Unet++
    model_unetplusplus = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet',
                                          in_channels=3, classes=9)
    trainer_unetplusplus = UnetPlusPlusTrainer(model_unetplusplus, device, class_weights, lr=1e-4,
                                               experiment_name='unetplusplus_baseline')

    # Entrenar
    history = trainer_unetplusplus.train(train_loader, val_loader, num_epochs=100)
    """)
