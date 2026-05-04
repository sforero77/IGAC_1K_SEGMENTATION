# 📓 Notebooks Interactivos

Jupyter notebooks para experimentación interactiva con el proyecto.

## 📋 Notebooks Disponibles

### `01_explore_dataset.ipynb` ✅
**Exploración del Dataset**

Explora y visualiza el dataset de patches:
- Cargar patches individuales
- Visualizar imágenes y máscaras
- Estadísticas del dataset
- Probar augmentations
- Verificar DataLoader
- Analizar class weights

**Cuándo usar:** Antes de entrenar, para familiarizarse con los datos.

---

### `02_train_unet.ipynb` ✅
**Entrenamiento Interactivo de U-Net**

Entrena U-Net de forma interactiva:
- Configuración de hiperparámetros
- Creación del modelo y trainer
- Entrenamiento con progress bars
- Visualización de curvas de entrenamiento
- Visualización de predicciones
- Métricas detalladas por clase

**Cuándo usar:** Para experimentar con diferentes configuraciones y ver resultados en tiempo real.

---

## 🚀 Quick Start

### 1. Instalar Jupyter

```bash
pip install jupyter notebook
# o
pip install jupyterlab
```

### 2. Iniciar Jupyter

```bash
# Desde la raíz del proyecto
cd D:\IGAC_1K_SEGMENTATION
jupyter notebook

# O JupyterLab
jupyter lab
```

### 3. Abrir Notebooks

Navega a la carpeta `notebooks/` y abre cualquier notebook.

---

## 💡 Ventajas de Usar Notebooks

### vs Scripts CLI:

| Aspecto | Scripts CLI | Notebooks |
|---------|-------------|-----------|
| **Experimentación** | Ejecutar todo de nuevo | Ejecución celda por celda |
| **Visualización** | Solo en TensorBoard | Inline + TensorBoard |
| **Debugging** | Print statements | Inspección de variables |
| **Iteración** | Relanzar script completo | Modificar y re-ejecutar |
| **Documentación** | Comentarios en código | Markdown + código |

### Casos de Uso:

**Usar Notebooks cuando:**
- Estás explorando datos
- Probando diferentes hiperparámetros
- Debugging de modelos
- Visualizando resultados
- Creando reportes interactivos

**Usar Scripts CLI cuando:**
- Entrenamientos largos (overnight)
- Producción / deployment
- Automatización con cron/scheduler
- Ejecución en servidor sin GUI

---

## 🛠️ Estructura de Código Reutilizable

Los notebooks importan clases de `scripts/utils/`:

```python
from utils import (
    # Dataset
    SegmentationDataset, create_dataloaders,

    # Augmentation
    get_training_augmentation, get_validation_augmentation,

    # Training
    UNetTrainer, DeepLabV3Trainer,

    # Metrics
    SegmentationMetrics,

    # Config
    get_device, set_seed, load_class_weights,
    N_CLASSES, CLASS_NAMES, PATCHES_DIR
)
```

**Ventajas:**
- ✅ Código centralizado
- ✅ Sin duplicación
- ✅ Fácil de mantener
- ✅ Importar y usar

---

## 📚 Ejemplos de Uso

### Ejemplo 1: Entrenar con Diferentes Encoders

```python
# En notebook 02_train_unet.ipynb, cambiar:
ENCODER = 'resnet50'  # Probar diferentes encoders
EXPERIMENT_NAME = 'unet_resnet50'

# Resto del código igual
model = smp.Unet(encoder_name=ENCODER, ...)
trainer = UNetTrainer(model, ..., experiment_name=EXPERIMENT_NAME)
history = trainer.train(train_loader, val_loader, num_epochs=100)
```

### Ejemplo 2: Experimentar con Learning Rate

```python
# Probar diferentes LRs
for lr in [1e-3, 1e-4, 1e-5]:
    experiment_name = f'unet_lr_{lr}'

    model = smp.Unet(...)
    trainer = UNetTrainer(model, device, class_weights, lr=lr,
                          experiment_name=experiment_name)

    history = trainer.train(train_loader, val_loader, num_epochs=20)
    print(f"LR {lr}: Best mIoU = {trainer.best_val_miou:.4f}")
```

### Ejemplo 3: Usar BalancedSampler

```python
# En creación de dataloaders
dataloaders = create_dataloaders(
    ...,
    use_balanced_sampler=True  # Sobre-muestrear minoritarias
)
```

---

## 🐛 Troubleshooting

### Error: Módulo utils no encontrado

```python
# Agregar al inicio del notebook
import sys
sys.path.append('../scripts')
```

### Error: CUDA Out of Memory

```python
# Reducir batch size
BATCH_SIZE = 8  # En lugar de 16
```

### Jupyter se congela en DataLoader

```python
# Reducir workers
NUM_WORKERS = 0  # Para debugging
```

### Restart del Kernel

Si cambias código en `utils/`, reinicia el kernel:
- Menú: Kernel → Restart & Run All

---

## 📊 Integración con TensorBoard

Los notebooks usan TensorBoard para logging. Para monitorear en tiempo real:

```bash
# Terminal separada
tensorboard --logdir=logs/tensorboard

# Abrir en navegador
# http://localhost:6006
```

---

## 🎯 Próximos Notebooks (Planeados)

- `03_evaluate_model.ipynb` - Evaluar en test set
- `04_compare_models.ipynb` - Comparar múltiples modelos
- `05_visualize_errors.ipynb` - Análisis de errores
- `06_inference_full_image.ipynb` - Inferencia en ortofoto completa
- `07_postprocess_predictions.ipynb` - Post-procesamiento morfológico

---

## 📊 Notebooks Antiguos (Análisis Exploratorio Inicial)

Los siguientes notebooks están planeados para análisis exploratorio:

### `02_preprocessing_validation.ipynb`
**Día 4-5 - Validación de Preprocesamiento**
- Verificación de normalización radiométrica
- Validación de alineación ortofoto-máscara
- Inspección visual de rasterización
- Verificación de resolución de conflictos

### `03_dataset_statistics.ipynb`
**Día 7 - Estadísticas del Dataset**
- Análisis de balance de clases
- Distribución de patches por split (train/val/test)
- Visualización de aumentación de datos
- Estadísticas de tamaño de objetos por clase

### `04_training_analysis.ipynb`
**Día 12 - Análisis de Entrenamiento**
- Curvas de loss train/val por modelo
- Evolución de mIoU durante entrenamiento
- Comparación de tiempos de entrenamiento
- Detección de overfitting/underfitting
- Visualización de predicciones en validación

### `05_results_visualization.ipynb`
**Día 22 - Visualización de Resultados**
- Predicciones sobre conjunto de test
- Mapas de confianza (softmax probabilities)
- Visualización ortofoto | ground truth | predicción
- Zoom en casos interesantes

### `06_comparative_analysis.ipynb`
**Día 26 - Análisis Comparativo**
- Comparación cuantitativa entre todos los modelos
- Gráficos radar multicriterio
- Heatmaps de desempeño por clase
- Scatter plots: exactitud vs eficiencia
- Tablas consolidadas de métricas

### `07_error_analysis.ipynb`
**Día 25 - Análisis de Errores**
- Mapas espaciales de error por clase
- Categorización de errores (borde, confusión, omisión)
- Análisis de errores por condición (sombra, densidad)
- Identificación de failure cases
- Ejemplos visuales de cada tipo de error

### `08_generalization_study.ipynb`
**Día 27 - Estudios de Generalización**
- Evaluación inter-ciudad (transferibilidad)
- Curvas de aprendizaje (data efficiency)
- Robustez a condiciones adversas
- Análisis de domain shift

## ⚠️ Importante:

### Propósito de los Notebooks:
- **Exploración**: Para entender datos y resultados
- **Validación**: Verificar calidad de procesamiento
- **Visualización**: Generar figuras para reportes
- **Análisis**: Interpretar métricas y patrones

### NO usar notebooks para:
- ❌ Código de producción
- ❌ Procesamiento de datos completo
- ❌ Entrenamiento de modelos
- ❌ Pipeline automatizado

### Código de producción debe estar en:
- ✅ `src/` (módulos reutilizables)
- ✅ `scripts/` (ejecutables CLI)

## 🎨 Configuración de Notebooks:

### Imports Estándar:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rasterio
from pathlib import Path

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# Imports del proyecto
from src.data.preprocessing import *
from src.visualization.plotting import *
```

### Estructura Recomendada:
```markdown
# Título del Notebook

## 1. Setup e Imports
## 2. Configuración de Paths
## 3. Carga de Datos
## 4. Análisis Exploratorio
## 5. Visualizaciones
## 6. Conclusiones
```

## 📊 Figuras para Reporte:

Guardar figuras de alta calidad en `reports/figures/`:
```python
fig.savefig(
    'reports/figures/nombre_figura.pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=True
)
```

## 🔗 Recursos:

- **Data**: `data/`
- **Models**: `models/`
- **Results**: `results/`
- **Utils**: `src/visualization/`, `src/utils/`

## 💡 Tips:

1. **Restart & Run All** antes de compartir
2. **Clear outputs** de notebooks experimentales antes de commit
3. **Documentar** conclusiones importantes en Markdown
4. **Versionar** solo notebooks finalizados y limpios
