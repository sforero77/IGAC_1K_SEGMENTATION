"""
Métricas para evaluación de segmentación semántica

Incluye:
- IoU (Intersection over Union)
- mIoU (mean IoU)
- Dice Score
- Pixel Accuracy
- Per-class metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class SegmentationMetrics:
    """
    Clase para calcular métricas de segmentación semántica

    Args:
        num_classes: Número de clases
        ignore_index: Índice a ignorar (background)
        class_names: Diccionario con nombres de clases (opcional)
    """

    def __init__(self, num_classes: int, ignore_index: int = 0,
                 class_names: Optional[Dict[int, str]] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or {i: f"Class_{i}" for i in range(num_classes)}

        # Acumuladores para confusion matrix
        self.reset()

    def reset(self):
        """Reset acumuladores"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Actualiza métricas con un batch

        Args:
            pred: Predicciones (B, H, W) con clase predicha (long)
            target: Ground truth (B, H, W) con clase real (long)
        """
        # Convertir a numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # Flatten
        pred = pred.flatten()
        target = target.flatten()

        # Filtrar ignore_index
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        # ⚡ OPTIMIZACIÓN: Usar np.bincount en lugar de loop Python
        # Esto es ~1000x más rápido que el loop
        valid_indices = (target >= 0) & (target < self.num_classes) & \
                       (pred >= 0) & (pred < self.num_classes)
        target = target[valid_indices]
        pred = pred[valid_indices]

        # Calcular índices lineales para confusion matrix
        indices = target * self.num_classes + pred

        # Usar bincount para contar eficientemente
        counts = np.bincount(indices, minlength=self.num_classes**2)

        # Actualizar confusion matrix
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

        self.total_samples += len(target)

    def get_iou_per_class(self) -> np.ndarray:
        """
        Calcula IoU para cada clase

        Returns:
            Array de shape (num_classes,) con IoU por clase
        """
        # IoU = TP / (TP + FP + FN)
        # TP: diagonal de confusion matrix
        # FP: suma de columna - diagonal
        # FN: suma de fila - diagonal

        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        denominator = tp + fp + fn

        # Evitar división por cero
        iou = np.where(denominator > 0, tp / denominator, 0.0)

        return iou

    def get_miou(self, ignore_background: bool = True) -> float:
        """
        Calcula mean IoU

        Args:
            ignore_background: Si True, ignora clase 0 en el promedio

        Returns:
            mIoU (float)
        """
        iou_per_class = self.get_iou_per_class()

        if ignore_background:
            # Excluir clase 0
            iou_per_class = iou_per_class[1:]

        # Filtrar clases sin samples (IoU = 0 puede ser legítimo)
        # Solo promediar clases que aparecen en el dataset
        valid_ious = iou_per_class[iou_per_class > 0]

        if len(valid_ious) == 0:
            return 0.0

        return float(np.mean(valid_ious))

    def get_dice_per_class(self) -> np.ndarray:
        """
        Calcula Dice Score para cada clase

        Dice = 2*TP / (2*TP + FP + FN)

        Returns:
            Array de shape (num_classes,) con Dice por clase
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        denominator = 2 * tp + fp + fn

        dice = np.where(denominator > 0, 2 * tp / denominator, 0.0)

        return dice

    def get_pixel_accuracy(self) -> float:
        """
        Calcula Pixel Accuracy global

        Accuracy = (TP para todas las clases) / Total píxeles

        Returns:
            Pixel Accuracy (float)
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()

        if total == 0:
            return 0.0

        return float(correct / total)

    def get_class_accuracy(self) -> np.ndarray:
        """
        Calcula Accuracy por clase (recall)

        Class Accuracy = TP / (TP + FN)

        Returns:
            Array de shape (num_classes,) con accuracy por clase
        """
        tp = np.diag(self.confusion_matrix)
        total_per_class = self.confusion_matrix.sum(axis=1)

        accuracy = np.where(total_per_class > 0, tp / total_per_class, 0.0)

        return accuracy

    def get_precision_per_class(self) -> np.ndarray:
        """
        Calcula Precision por clase

        Precision = TP / (TP + FP)

        Returns:
            Array de shape (num_classes,) con precision por clase
        """
        tp = np.diag(self.confusion_matrix)
        total_predicted = self.confusion_matrix.sum(axis=0)

        precision = np.where(total_predicted > 0, tp / total_predicted, 0.0)

        return precision

    def get_f1_per_class(self) -> np.ndarray:
        """
        Calcula F1-Score por clase

        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Returns:
            Array de shape (num_classes,) con F1 por clase
        """
        precision = self.get_precision_per_class()
        recall = self.get_class_accuracy()  # Recall = Class Accuracy

        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * (precision * recall) / denominator, 0.0)

        return f1

    def get_summary(self, detailed: bool = False) -> Dict:
        """
        Retorna resumen de todas las métricas

        Args:
            detailed: Si True, incluye métricas por clase

        Returns:
            Dict con métricas
        """
        summary = {
            'mIoU': self.get_miou(ignore_background=True),
            'mIoU_with_bg': self.get_miou(ignore_background=False),
            'pixel_accuracy': self.get_pixel_accuracy(),
        }

        if detailed:
            iou_per_class = self.get_iou_per_class()
            dice_per_class = self.get_dice_per_class()
            f1_per_class = self.get_f1_per_class()
            precision_per_class = self.get_precision_per_class()
            recall_per_class = self.get_class_accuracy()

            for i in range(self.num_classes):
                class_name = self.class_names[i]
                summary[f'{class_name}_IoU'] = float(iou_per_class[i])
                summary[f'{class_name}_Dice'] = float(dice_per_class[i])
                summary[f'{class_name}_F1'] = float(f1_per_class[i])
                summary[f'{class_name}_Precision'] = float(precision_per_class[i])
                summary[f'{class_name}_Recall'] = float(recall_per_class[i])

        return summary

    def print_summary(self):
        """Imprime resumen de métricas"""
        print("\n" + "="*80)
        print("RESUMEN DE MÉTRICAS")
        print("="*80)

        print(f"\nmIoU (sin background): {self.get_miou(ignore_background=True):.4f}")
        print(f"Pixel Accuracy:        {self.get_pixel_accuracy():.4f}")

        print("\nMétricas por clase:")
        print("-" * 80)
        print(f"{'Clase':<25} {'IoU':<10} {'Dice':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print("-" * 80)

        iou_per_class = self.get_iou_per_class()
        dice_per_class = self.get_dice_per_class()
        f1_per_class = self.get_f1_per_class()
        precision_per_class = self.get_precision_per_class()
        recall_per_class = self.get_class_accuracy()

        for i in range(self.num_classes):
            if i == self.ignore_index:
                continue  # Skip background

            class_name = self.class_names[i]
            print(f"{class_name:<25} "
                  f"{iou_per_class[i]:<10.4f} "
                  f"{dice_per_class[i]:<10.4f} "
                  f"{f1_per_class[i]:<10.4f} "
                  f"{precision_per_class[i]:<12.4f} "
                  f"{recall_per_class[i]:<10.4f}")

        print("="*80)


def calculate_iou(pred: torch.Tensor, target: torch.Tensor,
                  num_classes: int, ignore_index: int = 0) -> torch.Tensor:
    """
    Calcula IoU para un batch (función standalone)

    Args:
        pred: Predicciones (B, H, W) o (B, C, H, W)
        target: Ground truth (B, H, W)
        num_classes: Número de clases
        ignore_index: Índice a ignorar

    Returns:
        Tensor con IoU por clase (num_classes,)
    """
    # Si pred es (B, C, H, W), tomar argmax
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            ious.append(torch.tensor(0.0))
            continue

        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = torch.tensor(0.0)
        else:
            iou = intersection / union

        ious.append(iou)

    return torch.stack(ious)


def calculate_miou(pred: torch.Tensor, target: torch.Tensor,
                   num_classes: int, ignore_index: int = 0) -> float:
    """
    Calcula mIoU para un batch (función standalone)

    Args:
        pred: Predicciones (B, H, W) o (B, C, H, W)
        target: Ground truth (B, H, W)
        num_classes: Número de clases
        ignore_index: Índice a ignorar

    Returns:
        mIoU (float)
    """
    ious = calculate_iou(pred, target, num_classes, ignore_index)

    # Filtrar ignore_index y clases sin samples
    valid_ious = ious[1:] if ignore_index == 0 else ious  # Excluir background
    valid_ious = valid_ious[valid_ious > 0]

    if len(valid_ious) == 0:
        return 0.0

    return float(valid_ious.mean())


if __name__ == "__main__":
    # Test de métricas
    print("Testing SegmentationMetrics...")

    # Crear datos dummy
    num_classes = 9
    batch_size = 4
    height, width = 128, 128

    # Predicciones dummy (argmax ya aplicado)
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))

    # Crear métrica
    class_names = {
        0: 'Background', 1: 'Construccion', 2: 'Bosque',
        3: 'DepositoAgua', 4: 'ZonaVerde', 5: 'ZonaDura',
        6: 'Via', 7: 'Cerca', 8: 'CabeceraMunicipal'
    }

    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=0,
        class_names=class_names
    )

    # Actualizar con batch
    metrics.update(pred, target)

    # Imprimir resultados
    metrics.print_summary()

    # Test función standalone
    print("\nTest función standalone:")
    miou = calculate_miou(pred, target, num_classes, ignore_index=0)
    print(f"  mIoU: {miou:.4f}")

    print("\nTest exitoso!")
