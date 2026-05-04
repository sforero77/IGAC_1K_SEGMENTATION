# 🚀 Guía de Instalación - IGAC Segmentación

Instrucciones para configurar el ambiente de desarrollo.

---

## ✅ Requerimientos del Sistema

### Hardware Mínimo:
- **GPU:** NVIDIA con soporte CUDA (RTX 4060 o superior recomendado)
- **VRAM:** Mínimo 8GB (12GB recomendado)
- **RAM:** Mínimo 16GB (32GB recomendado)
- **Disco:** 50GB libres (para datos + modelos)

### Software:
- **OS:** Windows 10/11, Linux, o macOS
- **CUDA:** 12.1 (para RTX 40xx series)
- **Conda:** Miniconda o Anaconda
- **Git:** Para clonar el repositorio

---

## 📦 Opción 1: Instalación con Conda (Recomendada)

### Paso 1: Clonar el Repositorio

```bash
git clone <url-del-repo>
cd IGAC_1K_SEGMENTATION
```

### Paso 2: Crear Ambiente desde `environment.yml`

```bash
# Crear ambiente
conda env create -f environment.yml

# Activar ambiente
conda activate igac_segmentation
```

### Paso 3: Verificar Instalación

```bash
# Verificar PyTorch y CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verificar GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Verificar paquetes clave
python -c "import segmentation_models_pytorch as smp; import albumentations; import rasterio; print('✓ Todos los paquetes instalados correctamente')"
```

**Salida esperada:**
```
PyTorch: 2.3.0
CUDA disponible: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4060
✓ Todos los paquetes instalados correctamente
```

### Paso 4: Instalar Kernel de Jupyter (Opcional)

```bash
# Para usar el ambiente en Jupyter
python -m ipykernel install --user --name=igac_segmentation --display-name="IGAC Segmentation"
```

---

## 📦 Opción 2: Instalación con pip (Alternativa)

⚠️ **Nota:** Conda es preferible para manejar dependencias geoespaciales (GDAL, rasterio).

### Paso 1: Crear Entorno Virtual

```bash
# Con venv
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate
```

### Paso 2: Instalar PyTorch con CUDA

```bash
# Para CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Verificar
python -c "import torch; print(torch.cuda.is_available())"
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

⚠️ **Problemas con rasterio/GDAL:**
Si tienes errores instalando `rasterio` con pip:

**Windows:**
```bash
# Descargar wheel pre-compilado desde:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
pip install rasterio-X.X.X-cpXX-cpXX-win_amd64.whl
```

**Linux/Mac:**
```bash
# Instalar GDAL del sistema primero
sudo apt-get install gdal-bin libgdal-dev  # Ubuntu/Debian
brew install gdal  # macOS

pip install GDAL==<version-del-sistema>
pip install rasterio
```

---

## 🔧 Configuración Post-Instalación

### 1. Verificar Estructura del Proyecto

```bash
python -c "from pathlib import Path; print('✓ ROOT:', Path.cwd()); import sys; sys.path.append('scripts'); from utils import ROOT_DIR, PATCHES_DIR; print('✓ Imports OK')"
```

### 2. Configurar TensorBoard

```bash
# Crear directorio de logs si no existe
mkdir -p logs/tensorboard
```

### 3. Test Rápido del Dataset

```bash
# Ejecutar test del dataset (si ya tienes patches)
python scripts/utils/dataset.py
```

---

## 🧪 Test de Instalación Completo

### Script de Test:

Crea y ejecuta `test_installation.py`:

```python
"""Test de instalación completa"""
import sys
sys.path.append('scripts')

print("="*60)
print("TEST DE INSTALACIÓN - IGAC SEGMENTACIÓN")
print("="*60)

# 1. PyTorch y CUDA
print("\n1. PyTorch y CUDA:")
import torch
print(f"   ✓ PyTorch: {torch.__version__}")
print(f"   ✓ CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. Segmentation Models
print("\n2. Segmentation Models:")
import segmentation_models_pytorch as smp
print(f"   ✓ SMP: {smp.__version__}")

# 3. Albumentations
print("\n3. Albumentations:")
import albumentations as A
print(f"   ✓ Albumentations: {A.__version__}")

# 4. Geospatial
print("\n4. Geospatial:")
import rasterio
import geopandas as gpd
print(f"   ✓ Rasterio: {rasterio.__version__}")
print(f"   ✓ GeoPandas: {gpd.__version__}")

# 5. Utils del proyecto
print("\n5. Utils del Proyecto:")
from utils import (
    get_device, N_CLASSES, CLASS_NAMES,
    SegmentationDataset, UNetTrainer,
    SegmentationMetrics
)
print(f"   ✓ Clases: {N_CLASSES}")
print(f"   ✓ Device: {get_device()}")
print(f"   ✓ UNetTrainer: OK")
print(f"   ✓ SegmentationMetrics: OK")

# 6. Crear modelo de prueba
print("\n6. Test de Modelo:")
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=N_CLASSES
)
print(f"   ✓ U-Net creado: {sum(p.numel() for p in model.parameters()):,} parámetros")

# 7. Test de tensor en GPU
if torch.cuda.is_available():
    print("\n7. Test GPU:")
    x = torch.randn(2, 3, 512, 512).cuda()
    model = model.cuda()
    with torch.no_grad():
        y = model(x)
    print(f"   ✓ Forward pass GPU: Input {x.shape} -> Output {y.shape}")

print("\n" + "="*60)
print("✅ INSTALACIÓN COMPLETA Y FUNCIONAL")
print("="*60)
```

### Ejecutar Test:

```bash
python test_installation.py
```

**Salida esperada:**
```
============================================================
TEST DE INSTALACIÓN - IGAC SEGMENTACIÓN
============================================================

1. PyTorch y CUDA:
   ✓ PyTorch: 2.3.0
   ✓ CUDA disponible: True
   ✓ GPU: NVIDIA GeForce RTX 4060
   ✓ CUDA version: 12.1
   ✓ VRAM: 8.00 GB

2. Segmentation Models:
   ✓ SMP: 0.3.3

3. Albumentations:
   ✓ Albumentations: 1.3.1

4. Geospatial:
   ✓ Rasterio: 1.3.8
   ✓ GeoPandas: 0.13.2

5. Utils del Proyecto:
   ✓ Clases: 9
   ✓ Device: cuda
   ✓ UNetTrainer: OK
   ✓ SegmentationMetrics: OK

6. Test de Modelo:
   ✓ U-Net creado: 24,439,585 parámetros

7. Test GPU:
   ✓ Forward pass GPU: Input (2, 3, 512, 512) -> Output (2, 9, 512, 512)

============================================================
✅ INSTALACIÓN COMPLETA Y FUNCIONAL
============================================================
```

---

## 🔄 Actualizar Ambiente

### Con Conda:

```bash
# Actualizar desde environment.yml
conda env update -f environment.yml --prune

# O reinstalar completamente
conda env remove -n igac_segmentation
conda env create -f environment.yml
```

### Con pip:

```bash
pip install -r requirements.txt --upgrade
```

---

## 🐛 Troubleshooting

### Problema: CUDA Out of Memory durante test

**Solución:**
```python
# Reducir batch size en test
x = torch.randn(1, 3, 256, 256).cuda()  # En lugar de (2, 3, 512, 512)
```

### Problema: `torch.cuda.is_available()` retorna `False`

**Verificar:**
1. Drivers de NVIDIA actualizados
2. CUDA Toolkit instalado
3. Versión correcta de PyTorch para tu CUDA

**Reinstalar PyTorch:**
```bash
# Verificar versión CUDA del sistema
nvidia-smi

# Instalar PyTorch apropiado
# Para CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Problema: Error al importar `rasterio`

**Windows:**
- Descargar wheel pre-compilado de [Christoph Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)
- Instalar: `pip install rasterio-X.X.X.whl`

**Linux:**
```bash
sudo apt-get install libgdal-dev
pip install rasterio --no-binary rasterio
```

### Problema: Jupyter no encuentra el kernel

```bash
# Instalar kernel manualmente
python -m ipykernel install --user --name=igac_segmentation
```

---

## 📚 Recursos Adicionales

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **CUDA Compatibility:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **Conda Cheat Sheet:** https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- **Rasterio Documentation:** https://rasterio.readthedocs.io/

---

## ✅ Next Steps

Una vez instalado correctamente:

1. **Explorar el dataset:**
   ```bash
   jupyter lab
   # Abrir: notebooks/01_explore_dataset.ipynb
   ```

2. **Entrenar U-Net:**
   ```bash
   # Desde script
   python scripts/02_training/train_unet.py

   # O desde notebook
   jupyter lab notebooks/02_train_unet.ipynb
   ```

3. **Monitorear con TensorBoard:**
   ```bash
   tensorboard --logdir=logs/tensorboard
   # http://localhost:6006
   ```

---

**Última actualización:** Noviembre 12, 2025
**Contacto:** Si tienes problemas, revisa la sección Troubleshooting o abre un issue.
