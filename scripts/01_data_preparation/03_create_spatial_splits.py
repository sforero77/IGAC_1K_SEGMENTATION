"""
Script para crear partición espacial con cuadrículas de 1km × 1km

OBJETIVO:
- Dividir las 4 ciudades en cuadros de 1km × 1km
- Filtrar cuadros con >90% NoData o con solo 1 clase
- Asignar cuadros a TRAIN (60%), VAL (20%), TEST (20%)
- Estratificar por ciudad para garantizar diversidad

METODOLOGÍA:
1. Leer los 4 rasters de etiquetas
2. Crear grilla de 1km × 1km sobre extent global
3. Por cada cuadro: calcular estadísticas (clases, % NoData)
4. Filtrar cuadros inválidos
5. Asignar estratificadamente a splits
6. Guardar shapefiles y reporte

SALIDAS:
- data/processed/spatial_splits/train_zones.gpkg
- data/processed/spatial_splits/val_zones.gpkg
- data/processed/spatial_splits/test_zones.gpkg
- data/processed/spatial_splits/all_zones.gpkg (con atributo 'split')
- data/processed/spatial_splits/spatial_split_report.txt
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios
RASTERS_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\rasters_etiquetas"
OUTPUT_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\spatial_splits"

# Tamaño de cuadrícula en metros (1km × 1km)
GRID_SIZE = 1000

# Criterios de filtrado
MAX_NODATA_PERCENT = 90  # Filtrar cuadros con >90% NoData
MIN_CLASSES = 2  # Mínimo 2 clases diferentes (sin contar NoData)

# Proporciones de split
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Semilla para reproducibilidad
RANDOM_SEED = 42

# Nombres de ciudades
CIUDADES = ['madrid', 'mosquera', 'tabio', 'tocancipa']


# ============================================================================
# FUNCIONES
# ============================================================================

def load_raster_metadata(raster_path):
    """
    Carga metadatos de un raster sin cargar todos los datos

    Returns:
        dict con 'path', 'bounds', 'crs', 'transform', 'width', 'height', 'ciudad'
    """
    with rasterio.open(raster_path) as src:
        ciudad = Path(raster_path).stem.replace('_labels', '')
        return {
            'path': raster_path,
            'ciudad': ciudad,
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
        }


def create_grid(bounds, grid_size, crs):
    """
    Crea una grilla de cuadros de tamaño fijo

    Args:
        bounds: (minx, miny, maxx, maxy)
        grid_size: Tamaño del cuadro en metros
        crs: Sistema de coordenadas

    Returns:
        GeoDataFrame con polígonos de la grilla
    """
    minx, miny, maxx, maxy = bounds

    # Calcular número de filas y columnas
    cols = int(np.ceil((maxx - minx) / grid_size))
    rows = int(np.ceil((maxy - miny) / grid_size))

    print(f"\n  Creando grilla de {cols} columnas × {rows} filas = {cols * rows} cuadros")

    # Crear cuadros
    grid_polygons = []
    grid_ids = []

    for i in range(cols):
        for j in range(rows):
            # Coordenadas del cuadro
            x1 = minx + (i * grid_size)
            y1 = miny + (j * grid_size)
            x2 = min(x1 + grid_size, maxx)
            y2 = min(y1 + grid_size, maxy)

            # Crear polígono
            grid_polygons.append(box(x1, y1, x2, y2))
            grid_ids.append(f"grid_{i:04d}_{j:04d}")

    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'geometry': grid_polygons
    }, crs=crs)

    return gdf


def calculate_grid_statistics(grid_gdf, raster_metadata_list):
    """
    Calcula estadísticas de clases para cada cuadro de la grilla

    Args:
        grid_gdf: GeoDataFrame con cuadros de la grilla
        raster_metadata_list: Lista de metadatos de rasters

    Returns:
        GeoDataFrame con columnas adicionales de estadísticas
    """
    print(f"\n  Calculando estadísticas por cuadro...")

    # Inicializar columnas
    grid_gdf['ciudad'] = None
    grid_gdf['total_pixels'] = 0
    grid_gdf['nodata_pixels'] = 0
    grid_gdf['nodata_percent'] = 0.0
    grid_gdf['num_classes'] = 0
    grid_gdf['classes_present'] = None
    grid_gdf['pixel_counts'] = None
    grid_gdf['is_valid'] = False

    # Procesar cada raster
    for raster_meta in raster_metadata_list:
        print(f"\n    Procesando {raster_meta['ciudad']}...")

        with rasterio.open(raster_meta['path']) as src:
            # Para cada cuadro de la grilla
            for idx, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc=f"      {raster_meta['ciudad']}"):
                # Verificar si el cuadro intersecta con este raster
                grid_bounds = row.geometry.bounds
                raster_bounds = raster_meta['bounds']

                # Verificar intersección
                if not (grid_bounds[0] < raster_bounds[2] and
                       grid_bounds[2] > raster_bounds[0] and
                       grid_bounds[1] < raster_bounds[3] and
                       grid_bounds[3] > raster_bounds[1]):
                    continue

                # Calcular ventana de lectura
                try:
                    window = src.window(*grid_bounds)

                    # Leer datos de la ventana
                    data = src.read(1, window=window)

                    if data.size == 0:
                        continue

                    # Contar píxeles por clase
                    unique, counts = np.unique(data, return_counts=True)
                    pixel_counts = dict(zip(unique.tolist(), counts.tolist()))

                    # Calcular estadísticas
                    total_pixels = data.size
                    nodata_pixels = pixel_counts.get(0, 0)
                    nodata_percent = (nodata_pixels / total_pixels) * 100

                    # Clases presentes (sin contar NoData=0)
                    classes_present = [c for c in unique if c > 0]
                    num_classes = len(classes_present)

                    # Actualizar GeoDataFrame (acumular si ya había datos)
                    if grid_gdf.at[idx, 'total_pixels'] == 0:
                        # Primera vez que se procesa este cuadro
                        grid_gdf.at[idx, 'ciudad'] = raster_meta['ciudad']
                        grid_gdf.at[idx, 'total_pixels'] = total_pixels
                        grid_gdf.at[idx, 'nodata_pixels'] = nodata_pixels
                        grid_gdf.at[idx, 'nodata_percent'] = nodata_percent
                        grid_gdf.at[idx, 'num_classes'] = num_classes
                        grid_gdf.at[idx, 'classes_present'] = str(classes_present)
                        grid_gdf.at[idx, 'pixel_counts'] = str(pixel_counts)
                    else:
                        # Ya tenía datos de otra ciudad → acumular
                        grid_gdf.at[idx, 'total_pixels'] += total_pixels
                        grid_gdf.at[idx, 'nodata_pixels'] += nodata_pixels

                        # Recalcular porcentaje
                        new_nodata_percent = (grid_gdf.at[idx, 'nodata_pixels'] / grid_gdf.at[idx, 'total_pixels']) * 100
                        grid_gdf.at[idx, 'nodata_percent'] = new_nodata_percent

                        # Actualizar clases
                        prev_counts = eval(grid_gdf.at[idx, 'pixel_counts'])
                        for c, count in pixel_counts.items():
                            prev_counts[c] = prev_counts.get(c, 0) + count

                        classes_present_all = [c for c in prev_counts.keys() if c > 0]
                        grid_gdf.at[idx, 'num_classes'] = len(classes_present_all)
                        grid_gdf.at[idx, 'classes_present'] = str(classes_present_all)
                        grid_gdf.at[idx, 'pixel_counts'] = str(prev_counts)
                        grid_gdf.at[idx, 'ciudad'] = grid_gdf.at[idx, 'ciudad'] + f"+{raster_meta['ciudad']}"

                except Exception as e:
                    # Cuadro fuera del extent del raster
                    continue

    # Marcar cuadros válidos
    grid_gdf['is_valid'] = (
        (grid_gdf['nodata_percent'] <= MAX_NODATA_PERCENT) &
        (grid_gdf['num_classes'] >= MIN_CLASSES)
    )

    return grid_gdf


def assign_splits_stratified(grid_gdf, train_ratio, val_ratio, test_ratio, random_seed):
    """
    Asigna cuadros a TRAIN/VAL/TEST de forma estratificada por ciudad

    Args:
        grid_gdf: GeoDataFrame con cuadros válidos
        train_ratio: Proporción de TRAIN (0-1)
        val_ratio: Proporción de VAL (0-1)
        test_ratio: Proporción de TEST (0-1)
        random_seed: Semilla aleatoria

    Returns:
        GeoDataFrame con columna 'split' asignada
    """
    print(f"\n  Asignando splits estratificados por ciudad...")

    np.random.seed(random_seed)

    # Filtrar solo cuadros válidos
    valid_grid = grid_gdf[grid_gdf['is_valid']].copy()

    print(f"    Total cuadros válidos: {len(valid_grid)}")
    print(f"    Distribución objetivo: TRAIN={train_ratio:.0%}, VAL={val_ratio:.0%}, TEST={test_ratio:.0%}")

    # Inicializar columna split
    valid_grid['split'] = None

    # Agrupar por ciudad
    for ciudad in valid_grid['ciudad'].unique():
        ciudad_mask = valid_grid['ciudad'] == ciudad
        ciudad_indices = valid_grid[ciudad_mask].index.tolist()

        # Shuffle
        np.random.shuffle(ciudad_indices)

        # Calcular tamaños de splits
        n_total = len(ciudad_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # n_test es el resto para evitar errores de redondeo

        # Asignar splits
        train_indices = ciudad_indices[:n_train]
        val_indices = ciudad_indices[n_train:n_train+n_val]
        test_indices = ciudad_indices[n_train+n_val:]

        valid_grid.loc[train_indices, 'split'] = 'train'
        valid_grid.loc[val_indices, 'split'] = 'val'
        valid_grid.loc[test_indices, 'split'] = 'test'

        print(f"\n    {ciudad}:")
        print(f"      Total: {n_total} cuadros")
        print(f"      TRAIN: {len(train_indices)} ({len(train_indices)/n_total:.1%})")
        print(f"      VAL:   {len(val_indices)} ({len(val_indices)/n_total:.1%})")
        print(f"      TEST:  {len(test_indices)} ({len(test_indices)/n_total:.1%})")

    return valid_grid


def save_splits(grid_gdf, output_dir):
    """
    Guarda los splits en archivos GeoPackage separados

    Args:
        grid_gdf: GeoDataFrame con splits asignados
        output_dir: Directorio de salida
    """
    print(f"\n  Guardando shapefiles...")

    os.makedirs(output_dir, exist_ok=True)

    # Guardar todos los cuadros (incluyendo inválidos) para referencia
    all_path = os.path.join(output_dir, 'all_zones.gpkg')
    grid_gdf.to_file(all_path, driver='GPKG', layer='all_zones')
    print(f"    OK {all_path}")

    # Guardar solo válidos con split asignado
    valid_grid = grid_gdf[grid_gdf['is_valid']].copy()

    # Guardar por split
    for split in ['train', 'val', 'test']:
        split_gdf = valid_grid[valid_grid['split'] == split].copy()
        split_path = os.path.join(output_dir, f'{split}_zones.gpkg')
        split_gdf.to_file(split_path, driver='GPKG', layer=f'{split}_zones')
        print(f"    OK {split_path} ({len(split_gdf)} cuadros)")


def generate_report(grid_gdf, output_dir):
    """
    Genera reporte con estadísticas de la partición espacial

    Args:
        grid_gdf: GeoDataFrame con splits asignados
        output_dir: Directorio de salida
    """
    report_path = os.path.join(output_dir, 'spatial_split_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE PARTICIÓN ESPACIAL\n")
        f.write("="*80 + "\n\n")

        # Estadísticas generales
        f.write("CONFIGURACION:\n")
        f.write(f"  Tamano de cuadricula: {GRID_SIZE}m x {GRID_SIZE}m\n")
        f.write(f"  Filtro NoData: <={MAX_NODATA_PERCENT}%\n")
        f.write(f"  Filtro clases minimas: >={MIN_CLASSES}\n")
        f.write(f"  Semilla aleatoria: {RANDOM_SEED}\n\n")

        # Cuadros totales vs válidos
        total_cuadros = len(grid_gdf)
        valid_cuadros = grid_gdf['is_valid'].sum()
        invalid_cuadros = total_cuadros - valid_cuadros

        f.write("CUADROS:\n")
        f.write(f"  Total generados: {total_cuadros}\n")
        f.write(f"  Válidos: {valid_cuadros} ({valid_cuadros/total_cuadros:.1%})\n")
        f.write(f"  Inválidos (filtrados): {invalid_cuadros} ({invalid_cuadros/total_cuadros:.1%})\n\n")

        # Razones de filtrado
        f.write("RAZONES DE FILTRADO:\n")
        high_nodata = ((grid_gdf['nodata_percent'] > MAX_NODATA_PERCENT) & (~grid_gdf['is_valid'])).sum()
        low_classes = ((grid_gdf['num_classes'] < MIN_CLASSES) & (~grid_gdf['is_valid'])).sum()
        f.write(f"  >90% NoData: {high_nodata}\n")
        f.write(f"  <{MIN_CLASSES} clases: {low_classes}\n\n")

        # Distribución por split
        valid_grid = grid_gdf[grid_gdf['is_valid']]

        f.write("DISTRIBUCIÓN POR SPLIT:\n")
        for split in ['train', 'val', 'test']:
            split_count = (valid_grid['split'] == split).sum()
            split_percent = split_count / valid_cuadros
            f.write(f"  {split.upper():5s}: {split_count:4d} cuadros ({split_percent:.1%})\n")

        f.write("\n")

        # Distribución por ciudad y split
        f.write("DISTRIBUCIÓN POR CIUDAD Y SPLIT:\n")
        f.write(f"  {'Ciudad':<15} {'TRAIN':>8} {'VAL':>8} {'TEST':>8} {'Total':>8}\n")
        f.write("  " + "-"*55 + "\n")

        for ciudad in sorted(valid_grid['ciudad'].unique()):
            ciudad_data = valid_grid[valid_grid['ciudad'] == ciudad]
            train_count = (ciudad_data['split'] == 'train').sum()
            val_count = (ciudad_data['split'] == 'val').sum()
            test_count = (ciudad_data['split'] == 'test').sum()
            total_count = len(ciudad_data)

            f.write(f"  {ciudad:<15} {train_count:8d} {val_count:8d} {test_count:8d} {total_count:8d}\n")

        f.write("\n")

        # Estadísticas de clases
        f.write("ESTADÍSTICAS DE CLASES EN CUADROS VÁLIDOS:\n")

        class_names = {
            1: 'Construccion',
            2: 'Bosque',
            3: 'DepositoAgua',
            4: 'ZonaVerde',
            5: 'ZonaDura',
            6: 'Via',
            7: 'Cerca',
            8: 'CabeceraMunicipal'
        }

        # Contar presencia de cada clase
        class_presence = {i: 0 for i in range(1, 9)}

        for idx, row in valid_grid.iterrows():
            if row['classes_present']:
                classes = eval(row['classes_present'])
                for c in classes:
                    if c in class_presence:
                        class_presence[c] += 1

        f.write(f"  {'Clase':<20} {'Cuadros con clase':>20} {'% del total':>15}\n")
        f.write("  " + "-"*60 + "\n")

        for class_id in sorted(class_presence.keys()):
            count = class_presence[class_id]
            percent = (count / valid_cuadros) * 100
            class_name = class_names.get(class_id, f'Clase_{class_id}')
            f.write(f"  {class_name:<20} {count:20d} {percent:14.1f}%\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\n  OK Reporte generado: {report_path}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal
    """
    print("="*80)
    print("CREACIÓN DE PARTICIÓN ESPACIAL")
    print("="*80)

    # 1. Cargar metadatos de rasters
    print("\n[1/7] Cargando metadatos de rasters...")

    raster_files = [os.path.join(RASTERS_DIR, f"{ciudad}_labels.tif") for ciudad in CIUDADES]
    raster_metadata = []

    for raster_path in raster_files:
        if os.path.exists(raster_path):
            meta = load_raster_metadata(raster_path)
            raster_metadata.append(meta)
            print(f"  OK {meta['ciudad']}: {meta['width']} x {meta['height']} pixeles")
        else:
            print(f"  ERROR No encontrado: {raster_path}")

    if not raster_metadata:
        print("\n  ERROR: No se encontraron rasters de etiquetas")
        return

    # 2. Calcular extent global
    print("\n[2/7] Calculando extent global...")

    all_bounds = [meta['bounds'] for meta in raster_metadata]
    global_minx = min(b[0] for b in all_bounds)
    global_miny = min(b[1] for b in all_bounds)
    global_maxx = max(b[2] for b in all_bounds)
    global_maxy = max(b[3] for b in all_bounds)

    global_bounds = (global_minx, global_miny, global_maxx, global_maxy)
    crs = raster_metadata[0]['crs']

    print(f"  Extent global: ({global_minx:.2f}, {global_miny:.2f}, {global_maxx:.2f}, {global_maxy:.2f})")
    print(f"  CRS: {crs}")

    # 3. Crear grilla
    print("\n[3/7] Creando grilla de cuadros...")
    grid_gdf = create_grid(global_bounds, GRID_SIZE, crs)

    # 4. Calcular estadísticas por cuadro
    print("\n[4/7] Calculando estadísticas por cuadro...")
    grid_gdf = calculate_grid_statistics(grid_gdf, raster_metadata)

    # 5. Asignar splits
    print("\n[5/7] Asignando splits...")
    grid_gdf = assign_splits_stratified(grid_gdf, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED)

    # 6. Guardar archivos
    print("\n[6/7] Guardando archivos...")
    save_splits(grid_gdf, OUTPUT_DIR)

    # 7. Generar reporte
    print("\n[7/7] Generando reporte...")
    generate_report(grid_gdf, OUTPUT_DIR)

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)

    total = len(grid_gdf)
    valid = grid_gdf['is_valid'].sum()
    train = (grid_gdf['split'] == 'train').sum()
    val = (grid_gdf['split'] == 'val').sum()
    test = (grid_gdf['split'] == 'test').sum()

    print(f"  Total cuadros: {total}")
    print(f"  Cuadros válidos: {valid} ({valid/total:.1%})")
    print(f"  TRAIN: {train} ({train/valid:.1%})")
    print(f"  VAL:   {val} ({val/valid:.1%})")
    print(f"  TEST:  {test} ({test/valid:.1%})")

    print("\n  Archivos generados:")
    print(f"    - {OUTPUT_DIR}/train_zones.gpkg")
    print(f"    - {OUTPUT_DIR}/val_zones.gpkg")
    print(f"    - {OUTPUT_DIR}/test_zones.gpkg")
    print(f"    - {OUTPUT_DIR}/all_zones.gpkg")
    print(f"    - {OUTPUT_DIR}/spatial_split_report.txt")

    print("\n[OK] Particion espacial completada exitosamente!")
    print("="*80)


if __name__ == "__main__":
    main()
