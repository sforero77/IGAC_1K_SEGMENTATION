"""
Script de diagnóstico detallado para GPU CUDA

Uso:
    python debug_gpu_detailed.py

Propósito:
- Identificar EXACTAMENTE dónde falla en GPU
- Probar cada componente por separado
- Sugerir soluciones específicas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path

from scripts.utils import (
    set_seed, load_class_weights, create_dataloaders,
    get_training_augmentation, get_validation_augmentation,
    N_CLASSES, PATCHES_DIR, PATCHES_INDEX
)


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_status(status, message):
    if status in ["✓", "OK"]:
        print(f"  [OK] {message}")
    elif status in ["✗", "ERROR", "X"]:
        print(f"  [ERROR] {message}")
    elif status in ["⚠", "WARNING"]:
        print(f"  [WARNING] {message}")
    elif status in ["->", ">>"]:
        print(f"  >> {message}")


def test_cuda_basics():
    """Probar configuración básica de CUDA"""
    print_section("1. VERIFICAR CONFIGURACIÓN CUDA")

    print_status("->", "Verificando CUDA...")
    if not torch.cuda.is_available():
        print_status("✗", "CUDA no disponible!")
        return False

    print_status("✓", "CUDA disponible")

    device_count = torch.cuda.device_count()
    print_status("✓", f"Número de GPUs: {device_count}")

    device_name = torch.cuda.get_device_name(0)
    print_status("✓", f"GPU: {device_name}")

    # Memoria
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free = total_memory - allocated

    print_status("✓", f"Memoria Total: {total_memory:.2f} GB")
    print_status("✓", f"Memoria Reservada: {reserved:.2f} GB")
    print_status("✓", f"Memoria Asignada: {allocated:.2f} GB")
    print_status("✓", f"Memoria Libre: {free:.2f} GB")

    return True


def test_tensor_operations():
    """Probar operaciones básicas en GPU"""
    print_section("2. PROBAR OPERACIONES BÁSICAS EN GPU")

    device = torch.device('cuda')

    try:
        # Crear tensores
        print_status("->", "Creando tensor en GPU...")
        x = torch.randn(1000, 1000, device=device)
        print_status("✓", "Tensor creado exitosamente")

        # Operaciones
        print_status("->", "Realizando operaciones...")
        y = x @ x.t()
        z = torch.nn.functional.relu(y)
        print_status("✓", "Operaciones básicas OK")

        # Sincronizar y limpiar
        torch.cuda.synchronize()
        del x, y, z
        torch.cuda.empty_cache()
        print_status("✓", "Sincronización y limpieza OK")

        return True

    except Exception as e:
        print_status("✗", f"Error en operaciones: {e}")
        return False


def test_model_creation():
    """Probar creación de modelo en GPU"""
    print_section("3. CREAR MODELO DeepLabv3+ EN GPU")

    try:
        device = torch.device('cuda')
        print_status("->", "Creando modelo...")

        model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=N_CLASSES
        )

        print_status("✓", "Modelo creado en CPU")

        print_status("->", "Moviendo modelo a GPU...")
        model = model.to(device)
        torch.cuda.synchronize()

        print_status("✓", "Modelo en GPU")

        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        print_status("✓", f"Parámetros: {total_params:,}")
        print_status("✓", f"Tamaño del modelo: {model_size:.2f} MB")

        # Verificar que está en GPU
        device_check = next(model.parameters()).device
        print_status("✓", f"Verificado en device: {device_check}")

        return model, device

    except Exception as e:
        print_status("✗", f"Error al crear modelo: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dataloader(device, batch_size=2):
    """Probar carga de datos en GPU"""
    print_section(f"4. CARGAR DATOS EN GPU (batch_size={batch_size})")

    try:
        print_status("->", "Creando DataLoaders...")

        train_transform = get_training_augmentation()
        val_transform = get_validation_augmentation()

        dataloaders = create_dataloaders(
            patches_dir=PATCHES_DIR,
            indices_csv=PATCHES_INDEX,
            batch_size=batch_size,
            num_workers=0,
            train_transform=train_transform,
            val_transform=val_transform,
            use_balanced_sampler=False
        )

        train_loader = dataloaders['train']
        print_status("✓", f"DataLoader creado: {len(train_loader)} batches")

        print_status("->", "Obteniendo primer batch...")
        images, masks = next(iter(train_loader))

        print_status("✓", f"Batch obtenido - Images shape: {images.shape}, Masks shape: {masks.shape}")

        print_status("->", "Moviendo batch a GPU...")
        images = images.to(device)
        masks = masks.to(device)

        torch.cuda.synchronize()
        print_status("✓", "Batch en GPU OK")

        return train_loader, images, masks, device

    except Exception as e:
        print_status("✗", f"Error al cargar datos: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_forward_pass(model, images, device):
    """Probar forward pass en GPU"""
    print_section("5. FORWARD PASS EN GPU")

    try:
        print_status("->", f"Forward pass con batch_size={images.shape[0]}...")

        model.eval()
        with torch.no_grad():
            outputs = model(images)

        torch.cuda.synchronize()

        print_status("✓", f"Forward pass exitoso!")
        print_status("✓", f"Output shape: {outputs.shape}")

        return outputs

    except Exception as e:
        print_status("✗", f"Error en forward pass: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_backward_pass(model, images, masks, device):
    """Probar backward pass en GPU"""
    print_section("6. BACKWARD PASS EN GPU")

    try:
        print_status("->", "Poniendo modelo en modo TRAIN...")
        model.train()  # IMPORTANTE: Debe estar en train mode para backward pass

        print_status("->", "Forward pass en TRAIN mode...")
        # IMPORTANTE: Forward pass debe hacerse en train mode para que los tensores tengan requires_grad
        outputs = model(images)

        print_status("->", "Creando loss function...")
        criterion = torch.nn.CrossEntropyLoss()

        print_status("->", "Calculando loss...")
        loss = criterion(outputs, masks)

        print_status("OK", f"Loss calculado: {loss.item():.4f}")

        print_status("->", "Backward pass...")
        loss.backward()

        torch.cuda.synchronize()
        print_status("OK", "Backward pass exitoso!")

        return loss

    except Exception as e:
        print_status("✗", f"Error en backward pass: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_metrics_update(outputs, masks, device):
    """Probar actualización de métricas (donde falla)"""
    print_section("7. ACTUALIZAR MÉTRICAS (punto crítico)")

    try:
        print_status("->", "Obteniendo predicciones...")
        preds = torch.argmax(outputs, dim=1)

        print_status("✓", f"Preds shape: {preds.shape}, dtype: {preds.dtype}, device: {preds.device}")

        print_status("->", "Detach...")
        preds_detached = preds.detach()
        print_status("✓", "Detach OK")

        print_status("->", "CPU...")
        preds_cpu = preds_detached.cpu()
        torch.cuda.synchronize()
        print_status("✓", "CPU OK")

        print_status("->", "NumPy...")
        preds_numpy = preds_cpu.numpy()
        print_status("✓", "NumPy OK")

        print_status("->", "Verificar masks...")
        masks_numpy = masks.detach().cpu().numpy()
        print_status("✓", "Masks a NumPy OK")

        # Aquí es donde falla normalmente
        print_status("->", "Simulando métrica update...")
        confusion = np.bincount(
            masks_numpy.ravel() * 9 + preds_numpy.ravel(),
            minlength=9 * 9
        )
        print_status("✓", "Métrica update OK!")

        return True

    except Exception as e:
        print_status("✗", f"Error en métricas: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_epoch_loop(model, train_loader, device):
    """Probar un epoch completo"""
    print_section("8. PROBAR EPOCH COMPLETO (2 batches)")

    try:
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for batch_idx, (images, masks) in enumerate(train_loader):
            if batch_idx >= 2:  # Solo 2 batches
                break

            print_status("->", f"Batch {batch_idx + 1}...")

            # Mover a GPU
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            print_status("✓", f"Batch {batch_idx + 1} OK - Loss: {loss.item():.4f}")

        print_status("✓", "Epoch completo OK!")
        return True

    except Exception as e:
        print_status("✗", f"Error en epoch: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_recommendations():
    """Imprimir recomendaciones finales"""
    print("\n" + "="*80)
    print("  RECOMENDACIONES Y SOLUCIONES")
    print("="*80)
    print()
    print("Si pasaste hasta aquí pero DeepLabv3Trainer sigue fallando:")
    print()
    print("1. USAR MIXED PRECISION (reduce memoria en 50%):")
    print("-" * 80)
    print("""
    import torch.cuda.amp as amp

    scaler = amp.GradScaler()

    with amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, masks)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    """)
    print()

    print("2. USAR GRADIENT CHECKPOINTING (reduce memoria):")
    print("-" * 80)
    print("""
    from torch.utils.checkpoint import checkpoint

    # En el modelo (para ResNet50 encoder)
    model.encoder.layer3 = torch.utils.checkpoint.checkpoint(model.encoder.layer3)
    """)
    print()

    print("3. LIMPIAR CACHÉ REGULARMENTE:")
    print("-" * 80)
    print("""
    if batch_idx % 100 == 0:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    """)
    print()

    print("4. USAR CUDA_LAUNCH_BLOCKING para mejor diagnóstico:")
    print("-" * 80)
    print("En PowerShell: $env:CUDA_LAUNCH_BLOCKING=1")
    print("En CMD: set CUDA_LAUNCH_BLOCKING=1")
    print()

    print("5. DESACTIVAR cuDNN BENCHMARK (puede causar problemas):")
    print("-" * 80)
    print("""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    """)
    print()

    print("6. REDUCIR BATCH SIZE MÁS (intenta 1 primero):")
    print("-" * 80)
    print("BATCH_SIZE = 1")
    print()

    print("7. ACTUALIZAR DRIVERS NVIDIA:")
    print("-" * 80)
    print("https://www.nvidia.com/Download/driverDetails.aspx")
    print()


def main():
    print("\n" + "="*80)
    print("  SCRIPT DE DIAGNÓSTICO DETALLADO GPU CUDA")
    print("="*80)

    set_seed(42)

    # Test 1: CUDA basics
    if not test_cuda_basics():
        return

    # Test 2: Tensor operations
    if not test_tensor_operations():
        print("\n⚠ Las operaciones básicas fallan. Problema grave de GPU/CUDA")
        return

    # Test 3: Model creation
    model, device = test_model_creation()
    if model is None:
        return

    # Test 4: DataLoader
    train_loader, images, masks, device = test_dataloader(device, batch_size=2)
    if train_loader is None:
        return

    # Test 5: Forward pass
    outputs = test_forward_pass(model, images, device)
    if outputs is None:
        return

    # Test 6: Backward pass
    loss = test_backward_pass(model, images, masks, device)
    if loss is None:
        return

    # Test 7: Metrics (el punto crítico)
    if not test_metrics_update(outputs, masks, device):
        print("\n⚠ Métricas fallan pero forward/backward OK")
        print("  Problema: Transfer GPU↔CPU o numpy conversion")
        return

    # Test 8: Full epoch
    if not test_epoch_loop(model, train_loader, device):
        return

    # All tests passed
    print("\n" + "="*80)
    print("  [OK] TODOS LOS TESTS PASARON")
    print("="*80)
    print()
    print("Si llegaste aquí, tu GPU funciona correctamente.")
    print("El problema está en DeepLabV3Trainer específicamente.")
    print()
    print("Intenta las soluciones en RECOMENDACIONES arriba.")
    print()

    print_recommendations()


if __name__ == '__main__':
    main()
