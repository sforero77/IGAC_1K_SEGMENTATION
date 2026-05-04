# Models (Modelos Entrenados)

Almacena checkpoints y configuraciones de todos los modelos del proyecto.

## 📂 Estructura por Modelo:

Cada carpeta de modelo contiene:
- `checkpoints/` - Pesos del modelo (.pth, .h5)
- `config.yaml` - Hiperparámetros y configuración usada

## 🏗️ Arquitecturas:

### `unet/`
**U-Net con ResNet-34 Encoder**
- Encoder: ResNet-34 preentrenado en ImageNet
- Decoder: Bloques convolucionales con skip connections
- Loss: CrossEntropy con class weights
- Optimizer: AdamW (lr=1e-4)

### `deeplabv3/`
**DeepLabv3+ con ResNet-50 Encoder**
- Encoder: ResNet-50 preentrenado
- ASPP: Dilations [6, 12, 18]
- Loss: CrossEntropy + Dice (híbrida 1:1)
- Optimizer: Adam (lr=1e-4)

### `segformer/`
**SegFormer (Transformer-based)**
- Encoder: MiT-B3 (Mix Transformer)
- Decoder: MLP ligero
- Loss: CrossEntropy
- Optimizer: AdamW (lr=6e-5)

### `pix2pix/`
**Pix2Pix (Conditional GAN)**
- Generator: U-Net architecture
- Discriminator: PatchGAN (70×70)
- Loss: L1 × 100 + Adversarial × 1
- Optimizers: Adam G(lr=2e-4) D(lr=1e-4)

### `sam_lora/`
**SAM con LoRA Fine-tuning (Experimental)**
- Base: Segment Anything Model (ViT-B)
- Adaptación: LoRA (rank=4-8)
- Loss: Focal + Dice
- Optimizer: AdamW (lr=1e-4 LoRA, lr=1e-5 resto)

### `traditional/`
**Métodos Tradicionales**
- Random Forest (.pkl)
- SVM (.pkl)
- OBIA model (.pkl)

## 📝 Convención de Nombres:

```
checkpoints/
├── {modelo}_epoch_010.pth      # Checkpoint regular
├── {modelo}_epoch_020.pth
├── {modelo}_best.pth           # Mejor según validación (mIoU)
└── {modelo}_final.pth          # Final del entrenamiento
```

## 💾 Gestión de Checkpoints:

- **Guardado**: Cada 5 epochs + mejor modelo
- **Criterio**: Mejor mIoU en validación
- **Formato**: PyTorch .pth (contiene model.state_dict())
- **NO versionar**: Archivos .pth muy pesados (ver .gitignore)
- **SÍ versionar**: Archivos config.yaml

## 📊 Información en Checkpoint:

Cada archivo .pth incluye:
```python
{
    'epoch': epoch_num,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': loss_train,
    'val_loss': loss_val,
    'val_miou': miou_val,
    'config': config_dict
}
```

## 🔄 Cargar Modelo:

```python
import torch
from src.models.unet import build_unet

# Cargar configuración
with open('models/unet/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Construir modelo
model = build_unet(**config['model_params'])

# Cargar pesos
checkpoint = torch.load('models/unet/checkpoints/unet_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Modelo cargado - Epoch {checkpoint['epoch']}, mIoU: {checkpoint['val_miou']:.4f}")
```

## 📈 Comparación de Modelos:

Ver notebook `notebooks/04_training_analysis.ipynb` para comparación de:
- Curvas de loss train/val
- Evolución de mIoU
- Tiempo de entrenamiento
- Uso de memoria

## 🔗 Scripts Relacionados:

- `scripts/08_train_unet.py`
- `scripts/09_train_deeplabv3.py`
- `scripts/10_train_segformer.py`
- `scripts/11_train_pix2pix.py`
- `scripts/12_train_traditional.py`
