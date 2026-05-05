# IGAC 1K — Segmentación semántica para cartografía base 1:1 000

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

Pipeline de segmentación semántica multiclase sobre ortofotografías RGB del IGAC (10 cm/px) para la generación automática de cartografía base a escala **1:1 000** sobre cuatro municipios de Cundinamarca: **Madrid, Mosquera, Tabio y Tocancipá**.

> Trabajo de grado · Maestría en Ciencias de la Información y las Comunicaciones (MCIC) · Universidad Distrital Francisco José de Caldas. Director: Prof. Rubén Medina.

---

## 📌 Resumen

Se entrenan y comparan **tres arquitecturas de Deep Learning** (**U-Net**, **DeepLabv3+**, **U-Net++**) y **tres baselines clásicos** (**Random Forest**, **Linear SVM**, **OBIA**) — todos sobre el mismo split de train/val/test, las mismas etiquetas y los mismos parches, evaluados con la misma métrica. El mejor modelo (DeepLabv3+ con encoder ResNet-50) se exporta a un pipeline vectorial automatizado (rasterizado → vectorización → simplificación RDP a 0.5 m) que produce capas listas para ArcGIS Pro.

**IoU por clase sobre el test set — modelos de Deep Learning** (18 991 parches, 8 clases temáticas):

| Clase | U-Net (ResNet-34) | **DeepLabv3+ (ResNet-50)** | U-Net++ (ResNet-50) |
|---|---:|---:|---:|
| Construcción       | 0.802 | **0.808** | 0.796 |
| Bosque             | 0.757 | **0.771** | 0.755 |
| Depósito de agua   | 0.422 | **0.516** | 0.333 |
| Zona verde         | 0.201 | **0.262** | 0.082 |
| Zona dura          | **0.302** | 0.299 | 0.243 |
| Vía                | 0.440 | 0.371 | **0.442** |
| Cerca              | 0.110 | **0.116** | 0.085 |
| Cabecera municipal | 0.671 | **0.682** | 0.654 |

> **Baselines clásicos (RF, SVM, OBIA)** — los scripts ya están disponibles en `scripts/02_training/train_{rf,svm,obia}_baseline.py`. Las métricas se completan al ejecutar los scripts (~6–10 h en CPU multinúcleo). Una vez generadas, se anexarán a esta tabla siguiendo el mismo formato.

Las clases con buena firma espectral en RGB (Construcción, Bosque, Cabecera municipal) alcanzan IoU > 0.65 en los tres modelos de DL. Las clases con respuesta espectral ambigua o lineal fina (Cerca, Zona verde, Zona dura, Vía) presentan IoU < 0.45, lo que confirma la limitación esperable del uso exclusivo de tres bandas visibles.

**Hiperparámetros y época de selección:**

| Modelo | Encoder | Best epoch (early stopping) |
|---|---|---:|
| U-Net | ResNet-34 | 29 |
| **DeepLabv3+** | **ResNet-50** | **27** |
| U-Net++ | ResNet-50 | 21 |

---

## 🧭 Metodología — CRISP-DM

El proyecto está organizado siguiendo las seis fases del estándar **CRISP-DM** (Cross-Industry Standard Process for Data Mining), adaptado a un problema de visión por computador:

### 1. Business Understanding · *Comprensión del problema*

**Pregunta de investigación:** ¿es viable automatizar la generación de cartografía base 1:1 000 mediante segmentación semántica profunda sobre ortofotografías RGB del IGAC?

Insumos disponibles: ortofotos RGB IGAC a 10 cm/px (4 municipios) + capas vectoriales catastrales del IGAC como ground truth (8 clases temáticas).

### 2. Data Understanding · *Comprensión de los datos*

| Recurso | Propósito |
|---|---|
| [`notebooks/01_explore_dataset.ipynb`](notebooks/01_explore_dataset.ipynb) | Carga, visualización, balance de clases, prueba de augmentations |
| [`scripts/01_data_preparation/explore_gdb_structure.py`](scripts/01_data_preparation/explore_gdb_structure.py) | Inspección de las geodatabases del IGAC |
| [`scripts/01_data_preparation/extract_vector_layers.py`](scripts/01_data_preparation/extract_vector_layers.py) | Extracción de capas vectoriales por clase |
| [`scripts/01_data_preparation/04_calculate_normalization_stats.py`](scripts/01_data_preparation/04_calculate_normalization_stats.py) | Estadísticas RGB (μ, σ) para normalización |
| [`scripts/01_data_preparation/07_analyze_balance.py`](scripts/01_data_preparation/07_analyze_balance.py) | Análisis de balance de clases y pesos |

### 3. Data Preparation · *Preparación de los datos*

Pipeline determinista de generación de parches con partición espacial estratificada por municipio:

| Recurso | Propósito |
|---|---|
| [`scripts/01_data_preparation/consolidar_y_rasterizar.py`](scripts/01_data_preparation/consolidar_y_rasterizar.py) | Consolida vectores y rasteriza a etiquetas multiclase |
| [`scripts/01_data_preparation/extract_cabecera_municipal.py`](scripts/01_data_preparation/extract_cabecera_municipal.py) | Extrae polígonos de cabeceras municipales |
| [`scripts/01_data_preparation/process_vias_buffer_cut.py`](scripts/01_data_preparation/process_vias_buffer_cut.py) · [`process_cercas_buffer.py`](scripts/01_data_preparation/process_cercas_buffer.py) | Buffer + corte para clases lineales (vías, cercas) |
| [`scripts/01_data_preparation/close_gaps_two_pass.py`](scripts/01_data_preparation/close_gaps_two_pass.py) · [`close_gaps_ciclorutas.py`](scripts/01_data_preparation/close_gaps_ciclorutas.py) | Cierre topológico de huecos |
| [`scripts/01_data_preparation/03_create_spatial_splits.py`](scripts/01_data_preparation/03_create_spatial_splits.py) | Partición espacial 1 km × 1 km estratificada por municipio (semilla 42) |
| [`scripts/01_data_preparation/06_create_patches.py`](scripts/01_data_preparation/06_create_patches.py) | Generación de 72 940 parches 256×256 |
| [`scripts/01_data_preparation/verificar_capas_vacias.py`](scripts/01_data_preparation/verificar_capas_vacias.py) | QA de capas vectoriales vacías |

**Salidas versionadas en este repo:** `data/processed/spatial_splits/{train,val,test,all}_zones.gpkg` + `spatial_split_report.txt`.

### 4. Modeling · *Modelado*

#### 4.1 Modelos de Deep Learning

Tres arquitecturas con `segmentation-models-pytorch`, mismo loop de entrenamiento, scheduler `CosineAnnealingWarmRestarts(T₀=10, T_mult=2)`, loss combinada CE + Dice ponderada por clase.

| Recurso | Modelo / propósito |
|---|---|
| [`scripts/02_training/train_unet.py`](scripts/02_training/train_unet.py) · [`notebooks/02_train_unet.ipynb`](notebooks/02_train_unet.ipynb) | U-Net (ResNet-34) |
| [`scripts/02_training/train_deeplabv3_gpu_optimized.py`](scripts/02_training/train_deeplabv3_gpu_optimized.py) · [`notebooks/03_train_deeplabv3.ipynb`](notebooks/03_train_deeplabv3.ipynb) | DeepLabv3+ (ResNet-50) |
| [`scripts/02_training/train_unetplusplus.py`](scripts/02_training/train_unetplusplus.py) · [`notebooks/04_train_unetplusplus.ipynb`](notebooks/04_train_unetplusplus.ipynb) | U-Net++ (ResNet-50) |
| [`scripts/02_training/debug_gpu_detailed.py`](scripts/02_training/debug_gpu_detailed.py) · [`test_cpu_training.py`](scripts/02_training/test_cpu_training.py) | Diagnósticos de GPU/CPU para entrenamiento |
| [`scripts/utils/trainers.py`](scripts/utils/trainers.py) | Clases `UNetTrainer`, `DeepLabV3Trainer`, `UnetPlusPlusTrainer` |
| [`scripts/utils/dataset.py`](scripts/utils/dataset.py) | `SegmentationDataset` + dataloaders |
| [`scripts/utils/augmentation.py`](scripts/utils/augmentation.py) | Albumentations: flips + rot90 |
| [`scripts/utils/metrics.py`](scripts/utils/metrics.py) | mIoU, IoU por clase, pixel accuracy |
| [`scripts/utils/config.py`](scripts/utils/config.py) | Paths, dispositivo, número de clases |

#### 4.2 Baselines clásicos (comparación rigurosa)

Se reimplementan tres enfoques tradicionales **sobre el mismo split de train/val/test** que los modelos de Deep Learning, para que la comparativa NO dependa de valores de literatura externa. Todos consumen las mismas etiquetas, los mismos parches y se evalúan con la misma clase `SegmentationMetrics`.

| Recurso | Algoritmo |
|---|---|
| [`scripts/02_training/train_rf_baseline.py`](scripts/02_training/train_rf_baseline.py) | **Random Forest** píxel-a-píxel · 15 features handcrafted (RGB + estadísticas locales 5×5 + Sobel + Laplaciano) · 200 árboles |
| [`scripts/02_training/train_svm_baseline.py`](scripts/02_training/train_svm_baseline.py) | **Linear SVM** (SGDClassifier `loss='hinge'`) · mismas features que RF · `class_weight='balanced'` |
| [`scripts/02_training/train_obia_baseline.py`](scripts/02_training/train_obia_baseline.py) | **OBIA** · superpíxeles SLIC (n=150, compactness=10) · 14 features por segmento (color, textura, geometría) · clasificador RandomForest |
| [`scripts/utils/handcrafted_features.py`](scripts/utils/handcrafted_features.py) | Módulo común de feature engineering (píxel + segmento) |
| [`scripts/02_training/_baseline_common.py`](scripts/02_training/_baseline_common.py) | Loader de parches y sampling estratificado |
| [`scripts/02_training/run_all_baselines.bat`](scripts/02_training/run_all_baselines.bat) | Ejecuta entrenamiento + evaluación de los tres en secuencia |
| [`scripts/03_evaluation/evaluate_baseline.py`](scripts/03_evaluation/evaluate_baseline.py) | Evaluador común — produce `test_metrics.json` y `confusion_matrix.npy` con la **misma estructura** que los modelos DL |
| [`notebooks/09_baselines_clasicos.ipynb`](notebooks/09_baselines_clasicos.ipynb) | **Notebook todo-en-uno** — entrena los 3 baselines, evalúa, guarda métricas y genera comparativa final (tabla + gráficos) DL vs baselines |

### 5. Evaluation · *Evaluación*

Inferencia sobre 18 991 parches de test, comparativa entre modelos, postprocesamiento morfológico y generación de figuras.

| Recurso | Propósito |
|---|---|
| [`notebooks/05_inference_test_set.ipynb`](notebooks/05_inference_test_set.ipynb) | Inferencia DeepLabv3+ sobre test |
| [`notebooks/06_inference_unet_test.ipynb`](notebooks/06_inference_unet_test.ipynb) | Inferencia U-Net sobre test |
| [`notebooks/07_comparativa_tres_modelos.ipynb`](notebooks/07_comparativa_tres_modelos.ipynb) | Comparativa cuantitativa entre arquitecturas |
| [`notebooks/08_postprocessing.ipynb`](notebooks/08_postprocessing.ipynb) | Postprocesamiento morfológico (apertura/cierre) |
| [`scripts/03_evaluation/generate_figura_3_3.py`](scripts/03_evaluation/generate_figura_3_3.py) | Partición espacial por grid 1 km × 1 km |
| [`scripts/03_evaluation/generate_figura_4_2.py`](scripts/03_evaluation/generate_figura_4_2.py) · [`_zoom.py`](scripts/03_evaluation/generate_figura_4_2_zoom.py) · [`_madrid_zoom.py`](scripts/03_evaluation/generate_figura_4_2_madrid_zoom.py) | Vectores generados (Tocancipá / Madrid + zooms) |
| [`scripts/03_evaluation/generate_figura_4_4.py`](scripts/03_evaluation/generate_figura_4_4.py) | Comparación ráster vs. vectorial · Mosquera |
| [`scripts/03_evaluation/generate_figura_4_6.py`](scripts/03_evaluation/generate_figura_4_6.py) | Curvas de pérdida y mIoU por época |
| [`scripts/03_evaluation/generate_figura_4_7.py`](scripts/03_evaluation/generate_figura_4_7.py) | Matriz de confusión normalizada (DeepLabv3+) |
| [`scripts/03_evaluation/generate_figura_4_8.py`](scripts/03_evaluation/generate_figura_4_8.py) | Histograma de áreas de polígonos |
| [`scripts/03_evaluation/generate_figura_4_9.py`](scripts/03_evaluation/generate_figura_4_9.py) | Simplificación RDP (antes/después) |
| [`scripts/03_evaluation/generate_figura_4_10.py`](scripts/03_evaluation/generate_figura_4_10.py) | Mosaico de parches con predicciones de los 3 modelos |

**Métricas versionadas en este repo:** `results/test_evaluation/<modelo>/test_metrics.json` + `confusion_matrix.npy` + `results/comparativa_tres_modelos/tabla_resumen_tres_modelos.csv`.

### 6. Deployment · *Despliegue*

Conversión raster → vector + empaquetado para ArcGIS Pro (Deep Learning Package).

| Recurso | Propósito |
|---|---|
| [`scripts/04_deployment/exportar_dlpk.py`](scripts/04_deployment/exportar_dlpk.py) · [`exportar_dlpk_arcgislearn.py`](scripts/04_deployment/exportar_dlpk_arcgislearn.py) | Exportación del modelo a `.dlpk` para ArcGIS Pro |
| [`scripts/04_deployment/postprocesar_arcpy.py`](scripts/04_deployment/postprocesar_arcpy.py) · [`notebooks/11_postprocesamiento_arcpy.ipynb`](notebooks/11_postprocesamiento_arcpy.ipynb) | Postprocesamiento morfológico con ArcPy |
| [`scripts/04_deployment/vectorizar_arcpy.py`](scripts/04_deployment/vectorizar_arcpy.py) · [`notebooks/12_vectorizacion_arcpy.ipynb`](notebooks/12_vectorizacion_arcpy.ipynb) | Vectorización + simplificación RDP a 0.5 m |
| [`scripts/04_deployment/reconstruir_mosaicos.py`](scripts/04_deployment/reconstruir_mosaicos.py) | Reconstrucción de mosaicos de inferencia |
| [`scripts/04_deployment/exportar_mosaicos_png.py`](scripts/04_deployment/exportar_mosaicos_png.py) | Exportación de mosaicos a PNG |
| [`scripts/04_deployment/exportar_spatial_splits_gpkg.py`](scripts/04_deployment/exportar_spatial_splits_gpkg.py) | Exporta splits espaciales a GPKG limpio (EPSG:9377) |

---

## 🌳 Estructura del repositorio

```
IGAC_1K_SEGMENTATION/
│
├── 📄 README.md                    ← este archivo
├── 📄 LICENSE                      ← MIT
├── 📄 INSTALL.md                   ← instalación detallada
├── 📄 INSTALL_QUICK.md             ← instalación rápida
├── 📄 requirements.txt             ← dependencias pip
├── 📄 environment.yml              ← entorno conda
├── 📄 install_pip.bat              ← script de instalación Windows
│
├── 📁 data/
│   ├── raw/                                ← ortofotos IGAC (Drive)
│   ├── interim/
│   │   ├── README.md
│   │   ├── estadisticas/                   ← .gitkeep
│   │   ├── normalization_stats.json        ← (Drive)
│   │   ├── class_weights.json              ← (Drive)
│   │   └── vectores_cleaned/               ← (Drive)
│   └── processed/
│       ├── README.md
│       ├── spatial_splits/                 ✓ versionado
│       │   ├── all_zones.gpkg
│       │   ├── train_zones.gpkg            (60 % · 86 celdas)
│       │   ├── val_zones.gpkg              (20 % · 27 celdas)
│       │   ├── test_zones.gpkg             (20 % · 33 celdas)
│       │   └── spatial_split_report.txt
│       ├── rasters_etiquetas/              ← (Drive)
│       └── patches/                        ← (Drive, ~70 GB)
│
├── 📁 notebooks/                           # Jupyter notebooks (10)
│   ├── 01_explore_dataset.ipynb            [Data Understanding]
│   ├── 02_train_unet.ipynb                 [Modeling]
│   ├── 03_train_deeplabv3.ipynb            [Modeling]
│   ├── 04_train_unetplusplus.ipynb         [Modeling]
│   ├── 05_inference_test_set.ipynb         [Evaluation]
│   ├── 06_inference_unet_test.ipynb        [Evaluation]
│   ├── 07_comparativa_tres_modelos.ipynb   [Evaluation]
│   ├── 08_postprocessing.ipynb             [Evaluation]
│   ├── 09_baselines_clasicos.ipynb         [Modeling + Evaluation — RF, SVM, OBIA]
│   ├── 11_postprocesamiento_arcpy.ipynb    [Deployment]
│   ├── 12_vectorizacion_arcpy.ipynb        [Deployment]
│   └── README.md
│
├── 📁 scripts/                             # Código ejecutable (CRISP-DM)
│   ├── README.md
│   ├── __init__.py
│   │
│   ├── 📁 01_data_preparation/             [Data Understanding + Preparation]
│   │   ├── 03_create_spatial_splits.py
│   │   ├── 04_calculate_normalization_stats.py
│   │   ├── 06_create_patches.py
│   │   ├── 07_analyze_balance.py
│   │   ├── consolidar_y_rasterizar.py
│   │   ├── explore_gdb_structure.py
│   │   ├── extract_vector_layers.py
│   │   ├── extract_cabecera_municipal.py
│   │   ├── process_vias_buffer_cut.py
│   │   ├── process_cercas_buffer.py
│   │   ├── close_gaps_two_pass.py
│   │   ├── close_gaps_ciclorutas.py
│   │   └── verificar_capas_vacias.py
│   │
│   ├── 📁 02_training/                     [Modeling]
│   │   ├── train_unet.py
│   │   ├── train_deeplabv3_gpu_optimized.py
│   │   ├── debug_gpu_detailed.py
│   │   └── test_cpu_training.py
│   │
│   ├── 📁 03_evaluation/                   [Evaluation]
│   │   ├── generate_figura_3_3.py          (partición espacial)
│   │   ├── generate_figura_4_2.py          (vectores Tocancipá)
│   │   ├── generate_figura_4_2_zoom.py     (Tocancipá + zooms)
│   │   ├── generate_figura_4_2_madrid_zoom.py (Madrid + zooms)
│   │   ├── generate_figura_4_4.py          (comparación Mosquera)
│   │   ├── generate_figura_4_6.py          (curvas entrenamiento)
│   │   ├── generate_figura_4_7.py          (matriz confusión)
│   │   ├── generate_figura_4_8.py          (histograma de áreas)
│   │   ├── generate_figura_4_9.py          (simplificación RDP)
│   │   └── generate_figura_4_10.py         (mosaico predicciones)
│   │
│   ├── 📁 04_deployment/                   [Deployment]
│   │   ├── exportar_dlpk.py
│   │   ├── exportar_dlpk_arcgislearn.py
│   │   ├── exportar_mosaicos_png.py
│   │   ├── exportar_spatial_splits_gpkg.py
│   │   ├── reconstruir_mosaicos.py
│   │   ├── postprocesar_arcpy.py
│   │   └── vectorizar_arcpy.py
│   │
│   └── 📁 utils/                           [Modeling — librería interna]
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── augmentation.py
│       ├── metrics.py
│       └── trainers.py
│
├── 📁 models/
│   ├── README.md
│   └── checkpoints/                        ← (Drive, ~700 MB en .pth)
│
├── 📁 logs/                                ← (Drive — TensorBoard runs)
│
└── 📁 results/
    ├── figures/                            ← (Drive, ~73 MB en PNG)
    ├── vectores/                           ← (Drive, ~50 MB en GPKG + JSON)
    ├── inferencia_arcgis/                  ← (Drive, ~105 GB en TIF)
    ├── mosaicos/                           ← (Drive, ~7 GB)
    ├── comparativa_tres_modelos/
    │   └── tabla_resumen_tres_modelos.csv  ✓ versionado
    └── test_evaluation/
        └── <modelo>/
            ├── test_metrics.json           ✓ versionado
            └── confusion_matrix.npy        ✓ versionado
            (PNG e imágenes derivadas → Drive)
```

---

## 📦 Datos pesados (Google Drive)

Por restricciones de tamaño de GitHub, los siguientes recursos se distribuyen aparte por Google Drive (enlace contactando al autor):

| Recurso | Tamaño | Ruta esperada al descargar |
|---|---|---|
| Ortofotos crudas IGAC | ~280 GB | `data/raw/ortofotos/` |
| Parches 256×256 (72 940) | ~70 GB | `data/processed/patches/` |
| Rasters de etiquetas | ~5 GB | `data/processed/rasters_etiquetas/` |
| Rasters de inferencia | ~105 GB | `results/inferencia_arcgis/` |
| Mosaicos | ~7 GB | `results/mosaicos/` |
| Postprocessed (subset) | ~4.8 GB | `results/test_evaluation/deeplabv3_postprocessed*/` |
| Checkpoints entrenados | ~700 MB | `models/checkpoints/` |
| Paquete DLPK | ~430 MB | `dlpk_output/` + `deeplabv3plus_igac.dlpk` |
| Figuras de la tesis | ~73 MB | `results/figures/` |
| Capas vectoriales GPKG | ~50 MB | `results/vectores/` |
| Logs de TensorBoard | ~500 MB | `logs/tensorboard/` |

> Las figuras y vectores se pueden **regenerar localmente** ejecutando los scripts en `scripts/03_evaluation/` y `scripts/04_deployment/` una vez descargados los datos crudos y los checkpoints.

---

## 🚀 Reproducir el pipeline

### Instalación

```bash
# Conda (recomendado)
conda env create -f environment.yml
conda activate igac1k

# o con pip
pip install -r requirements.txt
```

Requiere PyTorch con soporte CUDA (probado con CUDA 12.1) y `segmentation-models-pytorch`. Para ArcPy se necesita ArcGIS Pro 3.x.

### Flujo end-to-end

```bash
# 1. Data Preparation (requiere data/raw/ y data/processed/rasters_etiquetas/)
python scripts/01_data_preparation/03_create_spatial_splits.py
python scripts/01_data_preparation/04_calculate_normalization_stats.py
python scripts/01_data_preparation/06_create_patches.py
python scripts/01_data_preparation/07_analyze_balance.py

# 2. Modeling (uno por arquitectura)
python scripts/02_training/train_unet.py
python scripts/02_training/train_deeplabv3_gpu_optimized.py
# U-Net++ → ver notebooks/04_train_unetplusplus.ipynb

# 3. Evaluation
jupyter nbconvert --execute notebooks/05_inference_test_set.ipynb
jupyter nbconvert --execute notebooks/07_comparativa_tres_modelos.ipynb

# 4. Generar las figuras de la tesis
for f in scripts/03_evaluation/generate_figura_*.py; do python "$f"; done

# 5. Deployment (postprocesado y vectorización requieren ArcGIS Pro / ArcPy)
python scripts/04_deployment/postprocesar_arcpy.py
python scripts/04_deployment/vectorizar_arcpy.py
python scripts/04_deployment/exportar_dlpk.py
```

---

## ⚙️ Configuración del entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Tamaño de parche | 256 × 256 px (256 × 256 m a 1 m/px) |
| Número de clases | 9 (8 clases + background) |
| Batch size | 16 (U-Net), 8 (DeepLabv3+, U-Net++) |
| Optimizador | Adam · lr inicial 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀ = 10, T_mult = 2) |
| Loss | CE + Dice ponderada por clase |
| Augmentation | flips + rotaciones de 90° |
| Early stopping | paciencia 10 épocas sobre val mIoU |

Particionado espacial estratificado por municipio sobre cuadrículas de 1 × 1 km (146 celdas → 60 % train / 20 % val / 20 % test) con semilla 42.

---

## 📚 Citar este trabajo

```
Forero, S. (2026). Segmentación semántica para la generación automática
de cartografía base a escala 1:1 000 sobre ortofotografías IGAC.
Trabajo de grado, Maestría en Ciencias de la Información y las Comunicaciones,
Universidad Distrital Francisco José de Caldas, Bogotá.
```

---

## 👤 Autor y contacto

**Sebastián Forero** · MCIC, Universidad Distrital
sebastian.forero.77@gmail.com

Director: Prof. Rubén Medina.

---

## 📜 Licencia

Código liberado bajo licencia **MIT** (ver archivo [`LICENSE`](LICENSE)). Los datos crudos del IGAC están sujetos a los términos de uso del Instituto Geográfico Agustín Codazzi.
