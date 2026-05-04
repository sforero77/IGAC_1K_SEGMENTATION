# ⚡ Instalación Rápida con pip (5 minutos)

**Para los que no quieren esperar 30 minutos a que conda resuelva dependencias.**

---

## Opción 1: Script Automático (MÁS FÁCIL)

```bash
# Ejecutar script
install_pip.bat

# Esperar ~5 minutos

# Verificar
python test_installation.py
```

---

## Opción 2: Manual (Paso a Paso)

### 1. Crear y activar ambiente virtual (30 segundos)

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Actualizar pip (30 segundos)

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Instalar PyTorch con CUDA (2 minutos)

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Instalar paquetes core (2 minutos)

```bash
pip install -r requirements_core.txt
```

### 5. Instalar geoespaciales (opcional, puede fallar en Windows)

```bash
pip install rasterio geopandas fiona shapely pyproj
```

**Si falla en Windows:**
- Descargar wheels de: https://www.lfd.uci.edu/~gohlke/pythonlibs/
- Buscar: `rasterio`, `fiona`, `GDAL`
- Instalar: `pip install archivo.whl`

### 6. Verificar (10 segundos)

```bash
python test_installation.py
```

---

## ¿Qué pasa si no instalo las librerías geoespaciales?

**Puedes entrenar igual!**

Las librerías geoespaciales (rasterio, geopandas) solo se necesitan para:
- Scripts de preparación de datos (ya ejecutados)
- Leer ortofotos originales

**Para entrenar solo necesitas:**
- ✅ PyTorch
- ✅ Segmentation Models PyTorch
- ✅ Albumentations
- ✅ TensorBoard

Los **patches ya están generados** en `.npy`, así que NO necesitas rasterio para entrenar.

---

## Instalación Mínima para Entrenar YA (1 minuto)

Si solo quieres entrenar modelos ahora:

```bash
# Crear venv
python -m venv venv
venv\Scripts\activate

# Actualizar pip
pip install --upgrade pip

# PyTorch con CUDA
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Paquetes mínimos para entrenar
pip install segmentation-models-pytorch albumentations tensorboard torchmetrics tqdm numpy pandas matplotlib jupyterlab ipykernel
```

**¡Listo! Ya puedes entrenar:**
```bash
python scripts/02_training/train_unet.py
```

---

## Verificación Rápida

```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Debe imprimir: True

# Verificar GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Debe imprimir: NVIDIA GeForce RTX 4060

# Test completo
python test_installation.py
```

---

## Tiempos Estimados

| Método | Tiempo | Dificultad |
|--------|--------|------------|
| `install_pip.bat` | ~5 min | ⭐ Fácil |
| Manual con requirements_core.txt | ~5 min | ⭐⭐ Media |
| Instalación mínima | ~1 min | ⭐ Muy fácil |
| Conda (environment.yml) | ~30 min | ⭐⭐⭐ Difícil |

---

## Troubleshooting Express

### Error: "torch.cuda.is_available() = False"

```bash
# Verificar driver
nvidia-smi

# Reinstalar PyTorch con CUDA correcta
pip uninstall torch torchvision torchaudio
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### Error: "No module named 'segmentation_models_pytorch'"

```bash
pip install segmentation-models-pytorch
```

### Error al instalar rasterio

**Solución 1:** Ignorar (no es necesario para entrenar)

**Solución 2:** Descargar wheel pre-compilado
- Ir a: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
- Descargar: `rasterio-1.3.9-cp310-cp310-win_amd64.whl`
- Instalar: `pip install rasterio-1.3.9-cp310-cp310-win_amd64.whl`

---

## Siguiente Paso

Una vez instalado:

```bash
# Explorar dataset
jupyter lab notebooks/01_explore_dataset.ipynb

# O entrenar directamente
python scripts/02_training/train_unet.py
```

---

**Última actualización:** Noviembre 12, 2025
**Tiempo total estimado:** 5 minutos
