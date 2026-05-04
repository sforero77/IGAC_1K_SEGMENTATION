"""
Script para crear patches 512x512 de ortofotos (X) y máscaras (Y) normalizados

OBJETIVO:
- Extraer patches de 512x512 píxeles
- Ortofoto y máscara deben venir del MISMO cuadro exacto
- Normalizar ortofotos on-the-fly: (X - mean) / std
- Guardar con el MISMO nombre para emparejar X-Y
- Organizar por split: train/val/test
- Filtrar patches con >95% NoData (solo TRAIN)

ESTRUCTURA DE SALIDA:
data/processed/patches/
├── train/
│   ├── images/     # Ortofotos normalizadas (float32, 3 bandas)
│   └── masks/      # Máscaras (uint8, 1 banda)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

NOMENCLATURA:
- {ciudad}_{x}_{y}.npy
- Ejemplo: madrid_4856700_2075500.npy
- El MISMO nombre para imagen y máscara

SALIDAS:
- Patches en data/processed/patches/{split}/images/ y masks/
- data/processed/patches/patches_index.csv
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios
ORTOFOTOS_DIR = r"D:\IGAC_1K_SEGMENTATION\data\raw\ortofotos"
MASCARAS_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\rasters_etiquetas"
SPLITS_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\spatial_splits"
NORMALIZATION_STATS_PATH = r"D:\IGAC_1K_SEGMENTATION\data\interim\normalization_stats.json"
OUTPUT_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\patches"

# Ciudades
CIUDADES = ['madrid', 'mosquera', 'tabio', 'tocancipa']

# Parámetros de patches
PATCH_SIZE = 512  # Tamaño del patch en píxeles
OVERLAP = 128     # Overlap entre patches (25%)

# Filtrado (solo TRAIN)
MAX_NODATA_PERCENT_TRAIN = 95  # Filtrar patches con >95% NoData en TRAIN
# En VAL/TEST conservar todos los patches


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def find_ortofoto_file(ciudad_dir):
    """Encuentra el archivo de ortofoto en el directorio de la ciudad"""
    ciudad_path = Path(ciudad_dir)
    tif_files = list(ciudad_path.glob("*.tif"))
    tif_files = [f for f in tif_files if not f.name.endswith('.ovr') and '.aux.' not in f.name]
    if tif_files:
        return str(tif_files[0])
    return None


def find_mascara_file(ciudad):
    """Encuentra el archivo de máscara para una ciudad"""
    mascara_path = os.path.join(MASCARAS_DIR, f"{ciudad}_labels.tif")
    if os.path.exists(mascara_path):
        return mascara_path
    return None


def load_normalization_stats(stats_path):
    """Carga estadísticas de normalización desde JSON"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    mean = np.array(stats['mean'], dtype=np.float32)
    std = np.array(stats['std'], dtype=np.float32)

    # Reshape para broadcasting: (1, 1, 3)
    mean = mean.reshape(1, 1, 3)
    std = std.reshape(1, 1, 3)

    return mean, std


def normalize_patch(patch, mean, std):
    """
    Normaliza un patch con z-score: (x - mean) / std

    Args:
        patch: numpy array (H, W, 3) con valores uint8 [0-255]
        mean: numpy array (1, 1, 3)
        std: numpy array (1, 1, 3)

    Returns:
        numpy array (H, W, 3) con valores float32 normalizados
    """
    # Convertir a float32
    patch_float = patch.astype(np.float32)

    # Normalizar
    normalized = (patch_float - mean) / std

    return normalized


def extract_patches_from_zone(ortofoto_src, mascara_src, zone_bounds, ciudad,
                              mean, std, patch_size=512, overlap=128):
    """
    Extrae patches de ortofoto y máscara desde un cuadro de la grilla

    Args:
        ortofoto_src: rasterio dataset abierto (ortofoto)
        mascara_src: rasterio dataset abierto (máscara)
        zone_bounds: (minx, miny, maxx, maxy) del cuadro
        ciudad: nombre de la ciudad
        mean, std: estadísticas de normalización
        patch_size: tamaño del patch
        overlap: overlap entre patches

    Returns:
        Lista de tuplas: (patch_id, patch_ortofoto_norm, patch_mascara, metadata)
    """
    patches = []

    # Calcular ventana en el raster para el cuadro
    try:
        ortofoto_window = ortofoto_src.window(*zone_bounds)
        mascara_window = mascara_src.window(*zone_bounds)
    except Exception as e:
        return patches

    # Leer datos de la zona completa
    try:
        # Ortofoto: primeras 3 bandas (RGB)
        ortofoto_data = ortofoto_src.read([1, 2, 3], window=ortofoto_window)  # (3, H, W)
        ortofoto_data = np.transpose(ortofoto_data, (1, 2, 0))  # (H, W, 3)

        # Máscara: 1 banda
        mascara_data = mascara_src.read(1, window=mascara_window)  # (H, W)

    except Exception as e:
        return patches

    # Ajustar dimensiones si no coinciden (tomar área común)
    if ortofoto_data.shape[:2] != mascara_data.shape[:2]:
        # Guardar dimensiones originales para logging
        orig_ortofoto_shape = ortofoto_data.shape[:2]
        orig_mascara_shape = mascara_data.shape[:2]

        # Tomar el mínimo de ambas dimensiones (intersección)
        min_height = min(ortofoto_data.shape[0], mascara_data.shape[0])
        min_width = min(ortofoto_data.shape[1], mascara_data.shape[1])

        # Crop ambos arrays al tamaño común
        ortofoto_data = ortofoto_data[:min_height, :min_width, :]
        mascara_data = mascara_data[:min_height, :min_width]

        # Calcular diferencias
        diff_height = max(orig_ortofoto_shape[0], orig_mascara_shape[0]) - min_height
        diff_width = max(orig_ortofoto_shape[1], orig_mascara_shape[1]) - min_width

        # Solo advertir si la diferencia es significativa (>1% del área)
        total_loss = diff_height * min_width + diff_width * min_height
        total_area = min_height * min_width
        loss_percent = (total_loss / total_area) * 100 if total_area > 0 else 0

        if loss_percent > 1.0:
            print(f"      ADVERTENCIA: Ajuste {orig_ortofoto_shape}->{orig_mascara_shape}. Perdida: {loss_percent:.1f}%")

    zone_height, zone_width = ortofoto_data.shape[:2]

    # Calcular step (cuántos píxeles avanzar entre patches)
    step = patch_size - overlap

    # Extraer patches con ventana deslizante
    for y in range(0, zone_height - patch_size + 1, step):
        for x in range(0, zone_width - patch_size + 1, step):
            # Extraer patch de ortofoto
            patch_ortofoto = ortofoto_data[y:y+patch_size, x:x+patch_size, :]

            # Extraer patch de máscara
            patch_mascara = mascara_data[y:y+patch_size, x:x+patch_size]

            # Verificar tamaño (descartar bordes incompletos)
            if patch_ortofoto.shape[0] != patch_size or patch_ortofoto.shape[1] != patch_size:
                continue

            # Calcular coordenadas geográficas del centro del patch
            # Convertir coordenadas de pixel a coordenadas geográficas
            window_transform = ortofoto_src.window_transform(ortofoto_window)
            from rasterio.transform import xy
            center_x_geo, center_y_geo = xy(
                window_transform,
                y + patch_size // 2,
                x + patch_size // 2
            )

            # Nombre del patch: ciudad_x_y (coordenadas redondeadas)
            patch_id = f"{ciudad}_{int(center_x_geo)}_{int(center_y_geo)}"

            # Calcular estadísticas del patch
            nodata_mask_ortofoto = np.all(patch_ortofoto == 0, axis=2)
            nodata_mask_mascara = (patch_mascara == 0)

            nodata_percent = (np.sum(nodata_mask_mascara) / (patch_size ** 2)) * 100

            # Clases presentes en la máscara (sin contar 0)
            unique_classes = np.unique(patch_mascara[~nodata_mask_mascara])
            unique_classes = unique_classes[unique_classes > 0].tolist()

            # Normalizar ortofoto
            patch_ortofoto_norm = normalize_patch(patch_ortofoto, mean, std)

            # Metadata
            metadata = {
                'patch_id': patch_id,
                'ciudad': ciudad,
                'center_x': int(center_x_geo),
                'center_y': int(center_y_geo),
                'nodata_percent': round(nodata_percent, 2),
                'num_classes': len(unique_classes),
                'classes_present': unique_classes
            }

            patches.append((patch_id, patch_ortofoto_norm, patch_mascara, metadata))

    return patches


def process_split(split_name, split_zones_gdf, mean, std):
    """
    Procesa todos los cuadros de un split (train/val/test)

    Args:
        split_name: 'train', 'val' o 'test'
        split_zones_gdf: GeoDataFrame con cuadros del split
        mean, std: estadísticas de normalización

    Returns:
        DataFrame con índice de patches
    """
    print(f"\n{'='*80}")
    print(f"Procesando split: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"  Cuadros en {split_name}: {len(split_zones_gdf)}")

    # Crear directorios de salida
    images_dir = os.path.join(OUTPUT_DIR, split_name, 'images')
    masks_dir = os.path.join(OUTPUT_DIR, split_name, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    all_metadata = []
    total_patches = 0
    filtered_patches = 0

    # Agrupar cuadros por ciudad
    # La columna 'ciudad' puede tener valores como "madrid", "mosquera", "madrid+mosquera", etc.
    # Necesitamos mapear cada cuadro a la ciudad principal

    for idx, zone in tqdm(split_zones_gdf.iterrows(), total=len(split_zones_gdf), desc=f"  {split_name}"):
        # Determinar ciudad(es) del cuadro
        ciudad_str = zone['ciudad']

        # Si es combinación (ej: "madrid+mosquera"), tomar la primera
        ciudad = ciudad_str.split('+')[0]

        # Buscar archivos de ortofoto y máscara
        ciudad_dir = os.path.join(ORTOFOTOS_DIR, ciudad)
        ortofoto_path = find_ortofoto_file(ciudad_dir)
        mascara_path = find_mascara_file(ciudad)

        if not ortofoto_path or not mascara_path:
            print(f"\n    ADVERTENCIA: No se encontraron archivos para {ciudad}. Saltando cuadro {zone['grid_id']}")
            continue

        # Abrir rasters
        try:
            with rasterio.open(ortofoto_path) as ortofoto_src, \
                 rasterio.open(mascara_path) as mascara_src:

                # Obtener bounds del cuadro
                zone_bounds = zone.geometry.bounds

                # Extraer patches
                patches = extract_patches_from_zone(
                    ortofoto_src, mascara_src, zone_bounds, ciudad,
                    mean, std, PATCH_SIZE, OVERLAP
                )

                # Guardar patches
                for patch_id, patch_ortofoto_norm, patch_mascara, metadata in patches:
                    # Filtrar patches con mucho NoData (solo en TRAIN)
                    if split_name == 'train' and metadata['nodata_percent'] > MAX_NODATA_PERCENT_TRAIN:
                        filtered_patches += 1
                        continue

                    # Guardar ortofoto normalizada
                    image_path = os.path.join(images_dir, f"{patch_id}.npy")
                    np.save(image_path, patch_ortofoto_norm.astype(np.float32))

                    # Guardar máscara
                    mask_path = os.path.join(masks_dir, f"{patch_id}.npy")
                    np.save(mask_path, patch_mascara.astype(np.uint8))

                    # Agregar metadata
                    metadata['split'] = split_name
                    metadata['grid_id'] = zone['grid_id']
                    metadata['image_path'] = image_path
                    metadata['mask_path'] = mask_path
                    all_metadata.append(metadata)

                    total_patches += 1

        except Exception as e:
            print(f"\n    ERROR procesando cuadro {zone['grid_id']}: {e}")
            continue

    print(f"\n  Resumen {split_name.upper()}:")
    print(f"    Patches generados: {total_patches}")
    if split_name == 'train':
        print(f"    Patches filtrados (>95% NoData): {filtered_patches}")

    return pd.DataFrame(all_metadata)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal
    """
    print("="*80)
    print("CREACIÓN DE PATCHES 512x512")
    print("="*80)

    # 1. Cargar estadísticas de normalización
    print("\n[1/5] Cargando estadísticas de normalización...")

    if not os.path.exists(NORMALIZATION_STATS_PATH):
        print(f"\n  ERROR: No se encontró {NORMALIZATION_STATS_PATH}")
        print("  Ejecuta primero: python scripts/04_calculate_normalization_stats.py")
        return

    mean, std = load_normalization_stats(NORMALIZATION_STATS_PATH)
    print(f"  OK Estadísticas cargadas")
    print(f"    Mean: {mean.flatten()}")
    print(f"    Std:  {std.flatten()}")

    # 2. Cargar splits espaciales
    print("\n[2/5] Cargando splits espaciales...")

    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(SPLITS_DIR, f'{split_name}_zones.gpkg')
        if os.path.exists(split_path):
            splits[split_name] = gpd.read_file(split_path)
            print(f"  OK {split_name}: {len(splits[split_name])} cuadros")
        else:
            print(f"  ERROR: No se encontró {split_path}")
            return

    # 3. Procesar cada split
    print("\n[3/5] Extrayendo patches por split...")

    all_indices = []

    for split_name in ['train', 'val', 'test']:
        split_index = process_split(split_name, splits[split_name], mean, std)
        all_indices.append(split_index)

    # 4. Consolidar índice
    print("\n[4/5] Consolidando índice...")

    full_index = pd.concat(all_indices, ignore_index=True)

    print(f"  Total patches generados: {len(full_index)}")
    print(f"\n  Distribución por split:")
    print(full_index['split'].value_counts())

    # 5. Guardar índice
    print("\n[5/5] Guardando índice...")

    index_path = os.path.join(OUTPUT_DIR, 'patches_index.csv')
    full_index.to_csv(index_path, index=False)

    print(f"  OK Guardado: {index_path}")

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)

    print(f"\n  Configuración:")
    print(f"    Tamaño de patch: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"    Overlap: {OVERLAP} píxeles")
    print(f"    Filtro NoData TRAIN: >{MAX_NODATA_PERCENT_TRAIN}%")

    print(f"\n  Patches generados:")
    for split_name in ['train', 'val', 'test']:
        count = len(full_index[full_index['split'] == split_name])
        print(f"    {split_name.upper():5s}: {count:6d} patches")

    print(f"\n  Distribución por ciudad:")
    ciudad_counts = full_index.groupby(['ciudad', 'split']).size().unstack(fill_value=0)
    print(ciudad_counts)

    print(f"\n  Archivos generados:")
    print(f"    - {OUTPUT_DIR}/train/images/")
    print(f"    - {OUTPUT_DIR}/train/masks/")
    print(f"    - {OUTPUT_DIR}/val/images/")
    print(f"    - {OUTPUT_DIR}/val/masks/")
    print(f"    - {OUTPUT_DIR}/test/images/")
    print(f"    - {OUTPUT_DIR}/test/masks/")
    print(f"    - {index_path}")

    print("\n✓ Proceso completado exitosamente!")
    print("="*80)


if __name__ == "__main__":
    main()
