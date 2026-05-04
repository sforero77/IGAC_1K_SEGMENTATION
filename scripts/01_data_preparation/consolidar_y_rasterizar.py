"""
Script para consolidar vectores por clase y rasterizar
1. Por cada municipio, lee todas las capas
2. Hace un dissolve (unión) por cada clase, creando un solo feature por clase
3. Guarda un shapefile consolidado con atributo 'class_id'
4. Rasteriza ese shapefile consolidado usando el atributo 'class_id'

Ventajas:
- Más eficiente: primero consolida vectores, luego rasteriza una sola vez
- Evita conflictos de prioridad píxel por píxel
- Genera un vector intermedio útil para análisis
"""

import os
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN DE CLASES
# ============================================================================

# Mapeo de clases a IDs (0 = NoData/Fondo)
# IMPORTANTE: Los nombres deben coincidir EXACTAMENTE como aparecen en las GDBs
CLASS_MAPPING = {
    # Nombre normalizado: (ID, lista de nombres exactos en las GDBs)
    'Construccion': (1, ['Construccion']),
    'Bosque': (2, ['Bosque']),
    'DepositoAgua': (3, ['DepositoAgua']),
    'ZonaVerde': (4, ['ZonaVerde']),
    'ZonaDura': (5, ['ZonaDura']),
    'Via': (6, ['Via']),
    'Lvia': (7, ['Lvia']),
    'Cicloruta': (8, ['Cicloruta']),
    'Cerca': (9, ['Cerca']),
    'CabeceraMunicipal': (10, ['CabeceraMunicipal']),
}

# Orden de procesamiento (de menor a mayor prioridad para overlay)
# Se dibuja de menor a mayor: las de mayor prioridad se dibujan al final
# y sobrescriben a las de menor prioridad
PRIORITY_ORDER = [
    ('CabeceraMunicipal', 1),  # Mas baja - se dibuja primero (fondo/mascara)
    ('Cerca', 2),
    ('Via', 3),
    ('Lvia', 4),
    ('Cicloruta', 5),
    ('ZonaDura', 6),
    ('ZonaVerde', 7),
    ('DepositoAgua', 8),
    ('Bosque', 9),
    ('Construccion', 10),      # Mas alta - se dibuja al final (encima de todo)
]

# Resolución espacial en metros (10cm)
RESOLUTION = 0.1

# CRS de salida
OUTPUT_CRS = None  # Si es None, se usa el CRS de la primera capa encontrada


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def find_layer_names(available_layers, class_name):
    """
    Encuentra todos los nombres reales de capas en la GDB para una clase
    """
    if class_name not in CLASS_MAPPING:
        return []

    _, variations = CLASS_MAPPING[class_name]
    found_layers = []
    for variation in variations:
        if variation in available_layers:
            found_layers.append(variation)
    return found_layers


def consolidate_class_geometries(gdb_path, class_name, class_id, target_crs):
    """
    Consolida todas las geometrías de una clase en un solo feature

    Args:
        gdb_path: Ruta a la GDB
        class_name: Nombre de la clase
        class_id: ID de la clase
        target_crs: CRS objetivo

    Returns:
        GeoDataFrame con un solo feature (o vacío si no hay datos)
    """
    available_layers = fiona.listlayers(gdb_path)
    layer_names = find_layer_names(available_layers, class_name)

    if not layer_names:
        return gpd.GeoDataFrame(columns=['geometry', 'class_id', 'class_name'], crs=target_crs)

    # Leer todas las capas de esta clase
    all_geometries = []
    for layer_name in layer_names:
        try:
            gdf = gpd.read_file(gdb_path, layer=layer_name)

            if gdf.empty:
                continue

            # Reproyectar si es necesario
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

            # Procesar geometrías: arreglar inválidas con buffer(0)
            for geom in gdf.geometry:
                if geom is None:
                    continue

                # Arreglar geometrías inválidas con buffer(0)
                if not geom.is_valid:
                    try:
                        geom = geom.buffer(0)
                    except:
                        continue

                if geom.is_valid and not geom.is_empty:
                    all_geometries.append(geom)

        except Exception as e:
            print(f"    Advertencia al leer {layer_name}: {e}")

    if not all_geometries:
        return gpd.GeoDataFrame(columns=['geometry', 'class_id', 'class_name'], crs=target_crs)

    print(f"    {class_name}: {len(all_geometries)} features recolectados")

    # NO hacer dissolve - mantener geometrías separadas para evitar pérdida de datos
    # Crear MultiPolygon/GeometryCollection con todas las geometrías
    from shapely.geometry import GeometryCollection
    unified_geometry = GeometryCollection(all_geometries)

    # Crear GeoDataFrame con el feature consolidado
    gdf_consolidated = gpd.GeoDataFrame(
        {
            'geometry': [unified_geometry],
            'class_id': [class_id],
            'class_name': [class_name]
        },
        crs=target_crs
    )

    return gdf_consolidated


def consolidate_municipality_vectors(gdb_path, output_shapefile):
    """
    Consolida todos los vectores de un municipio en un solo shapefile
    con un atributo 'class_id' por cada clase

    Args:
        gdb_path: Ruta a la GDB
        output_shapefile: Ruta del shapefile de salida (.gpkg)

    Returns:
        True si se consolidó exitosamente, False si falló
    """
    print(f"\n  Consolidando vectores por clase...")

    # Obtener CRS de la primera capa disponible
    available_layers = fiona.listlayers(gdb_path)
    if not available_layers:
        print(f"    ERROR: No hay capas en la GDB")
        return False

    with fiona.open(gdb_path, layer=available_layers[0]) as src:
        target_crs = OUTPUT_CRS if OUTPUT_CRS else src.crs

    # Consolidar cada clase
    all_classes = []
    for class_name, priority in PRIORITY_ORDER:
        class_id = CLASS_MAPPING[class_name][0]
        gdf_class = consolidate_class_geometries(gdb_path, class_name, class_id, target_crs)

        if not gdf_class.empty:
            all_classes.append(gdf_class)

    if not all_classes:
        print(f"    ERROR: No se pudo consolidar ninguna clase")
        return False

    # Concatenar todos los GeoDataFrames
    gdf_all = gpd.GeoDataFrame(pd.concat(all_classes, ignore_index=True), crs=target_crs)

    # Guardar como GeoPackage (mejor que shapefile para geometrías complejas)
    print(f"\n  Guardando vectores consolidados: {output_shapefile}")
    gdf_all.to_file(output_shapefile, driver='GPKG', layer='clases')

    # Estadísticas
    print(f"\n  Estadísticas del shapefile consolidado:")
    print(f"    Total de clases: {len(gdf_all)}")
    for idx, row in gdf_all.iterrows():
        area_km2 = row.geometry.area / 1_000_000  # Asumiendo CRS en metros
        print(f"      ID {row['class_id']:2d}: {row['class_name']:20s} - Área: {area_km2:.2f} km²")

    return True


def rasterize_consolidated_vectors(shapefile_path, output_raster, resolution=RESOLUTION):
    """
    Rasteriza el shapefile consolidado usando el atributo 'class_id'
    Rasteriza en orden de prioridad: menor a mayor (últimas sobrescriben)

    Args:
        shapefile_path: Ruta al shapefile consolidado
        output_raster: Ruta del raster de salida
        resolution: Resolución espacial en metros

    Returns:
        True si se rasterizó exitosamente, False si falló
    """
    print(f"\n  Rasterizando vectores consolidados...")

    # Leer shapefile
    gdf = gpd.read_file(shapefile_path, layer='clases')

    if gdf.empty:
        print(f"    ERROR: El shapefile está vacío")
        return False

    # Calcular extent
    bounds = gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds
    print(f"    Extent: ({min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f})")

    # Calcular dimensiones
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    print(f"    Dimensiones: {width} x {height} píxeles")
    print(f"    Tamaño aproximado: {width * height * 1 / (1024**2):.2f} MB")

    # Crear transform
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    # Inicializar raster vacío
    label_raster = np.zeros((height, width), dtype=np.uint8)

    # Crear diccionario de prioridades
    priority_dict = {class_name: priority for class_name, priority in PRIORITY_ORDER}

    # Ordenar GeoDataFrame por prioridad ASCENDENTE
    # Prioridad 1 se dibuja primero (fondo), prioridad 8 se dibuja al final (encima)
    gdf['priority'] = gdf['class_name'].map(priority_dict)
    gdf_sorted = gdf.sort_values('priority', ascending=True)

    # Rasterizar cada clase en orden de prioridad
    print(f"    Rasterizando por prioridad (1=primero/fondo -> 8=ultimo/encima):")
    for idx, row in gdf_sorted.iterrows():
        shapes = [(row.geometry, row.class_id)]

        temp_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        # Sobrescribir píxeles donde temp_raster > 0
        mask = temp_raster > 0
        label_raster[mask] = temp_raster[mask]

        pixel_count = np.sum(mask)
        print(f"      [{row.priority:2d}] {row.class_name}: {pixel_count:,} píxeles (ID {row.class_id})")

    # REMAP: Unir Via (6), Lvia (7) y Cicloruta (8) en una sola clase Via (6)
    print(f"\n  Remapeando clases de vias a una sola clase 'Via' (ID 6):")
    pixels_lvia = np.sum(label_raster == 7)
    pixels_cicloruta = np.sum(label_raster == 8)
    print(f"    Lvia (7) -> Via (6): {pixels_lvia:,} píxeles")
    print(f"    Cicloruta (8) -> Via (6): {pixels_cicloruta:,} píxeles")

    label_raster[label_raster == 7] = 6  # Lvia -> Via
    label_raster[label_raster == 8] = 6  # Cicloruta -> Via

    # Reindexar clases restantes: Cerca (9->7), CabeceraMunicipal (10->8)
    print(f"    Reindexando clases restantes:")
    print(f"    Cerca (9) -> (7)")
    print(f"    CabeceraMunicipal (10) -> (8)")
    label_raster[label_raster == 9] = 7   # Cerca: 9 -> 7
    label_raster[label_raster == 10] = 8  # CabeceraMunicipal: 10 -> 8

    # Guardar raster
    print(f"\n  Guardando raster: {output_raster}")

    # Tabla de colores (actualizada después del remap)
    color_table = {
        0: (0, 0, 0, 0),          # NoData (transparente)
        1: (255, 0, 0, 255),      # Construccion (rojo)
        2: (0, 128, 0, 255),      # Bosque (verde oscuro)
        3: (0, 0, 255, 255),      # DepositoAgua (azul)
        4: (0, 255, 0, 255),      # ZonaVerde (verde claro)
        5: (128, 128, 128, 255),  # ZonaDura (gris)
        6: (64, 64, 64, 255),     # Via (incluye Via+Lvia+Cicloruta) (gris oscuro)
        7: (139, 69, 19, 255),    # Cerca (cafe)
        8: (255, 255, 0, 255),    # CabeceraMunicipal (amarillo)
    }

    with rasterio.open(
        output_raster,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=gdf.crs,
        transform=transform,
        nodata=0,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(label_raster, 1)
        dst.write_colormap(1, color_table)

        # Metadatos
        dst.update_tags(
            descripcion='Raster de etiquetas consolidado',
            resolucion=f'{resolution}m',
            nodata='0'
        )

    # Estadísticas (después del remap)
    unique, counts = np.unique(label_raster, return_counts=True)
    print(f"\n  Estadisticas del raster final:")
    print(f"    Clase | Nombre           | Pixeles     | Porcentaje")
    print(f"    ------|------------------|-------------|------------")
    total_pixels = label_raster.size

    # Mapeo de IDs a nombres DESPUÉS del remap
    id_to_name_final = {
        0: 'NoData/Fondo',
        1: 'Construccion',
        2: 'Bosque',
        3: 'DepositoAgua',
        4: 'ZonaVerde',
        5: 'ZonaDura',
        6: 'Via',  # Incluye Via+Lvia+Cicloruta
        7: 'Cerca',
        8: 'CabeceraMunicipal'
    }

    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        class_name = id_to_name_final.get(class_id, 'Desconocido')
        print(f"    {class_id:5d} | {class_name:16s} | {count:11,d} | {percentage:6.2f}%")

    return True


def process_municipality(municipio_path, output_vectors_dir, output_rasters_dir):
    """
    Procesa un municipio completo:
    1. Consolida vectores por clase
    2. Rasteriza el shapefile consolidado

    Args:
        municipio_path: Ruta al directorio del municipio
        output_vectors_dir: Directorio de salida para vectores consolidados
        output_rasters_dir: Directorio de salida para rasters

    Returns:
        True si se procesó exitosamente, False si falló
    """
    municipio_name = os.path.basename(municipio_path)
    print(f"\n{'='*80}")
    print(f"Procesando: {municipio_name}")
    print(f"{'='*80}")

    # Buscar la GDB
    gdb_files = list(Path(municipio_path).glob("*.gdb"))
    if not gdb_files:
        print(f"  ERROR: No se encontró ninguna GDB en {municipio_path}")
        return False

    gdb_path = str(gdb_files[0])
    print(f"  GDB encontrada: {os.path.basename(gdb_path)}")

    # Paths de salida
    output_shapefile = os.path.join(output_vectors_dir, f"{municipio_name}_consolidated.gpkg")
    output_raster = os.path.join(output_rasters_dir, f"{municipio_name}_labels.tif")

    # PASO 1: Consolidar vectores por clase
    if not consolidate_municipality_vectors(gdb_path, output_shapefile):
        return False

    # PASO 2: Rasterizar vectores consolidados
    if not rasterize_consolidated_vectors(output_shapefile, output_raster, RESOLUTION):
        return False

    print(f"\n  OK Municipio {municipio_name} procesado exitosamente!")
    return True


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal: procesa todos los municipios
    """
    import pandas as pd

    # Directorios
    base_dir = r"D:\IGAC_1K_SEGMENTATION\data\processed\vectores_consolidados"
    output_vectors_dir = r"D:\IGAC_1K_SEGMENTATION\data\processed\vectores_consolidados_clase"
    output_rasters_dir = r"D:\IGAC_1K_SEGMENTATION\data\processed\rasters_etiquetas"

    # Crear directorios de salida
    os.makedirs(output_vectors_dir, exist_ok=True)
    os.makedirs(output_rasters_dir, exist_ok=True)

    print("="*80)
    print(" CONSOLIDACIÓN Y RASTERIZACIÓN DE VECTORES POR CLASE")
    print("="*80)
    print(f"  Directorio base: {base_dir}")
    print(f"  Directorio vectores consolidados: {output_vectors_dir}")
    print(f"  Directorio rasters: {output_rasters_dir}")
    print(f"  Resolución: {RESOLUTION}m ({int(RESOLUTION*100)}cm)")
    print(f"  Número de clases: {len(CLASS_MAPPING)}")
    print(f"\n  Proceso:")
    print(f"    1. Por cada municipio, leer todas las capas de cada clase")
    print(f"    2. Hacer dissolve (union) de cada clase -> 1 feature por clase")
    print(f"    3. Guardar shapefile consolidado con atributo 'class_id'")
    print(f"    4. Rasterizar el shapefile consolidado")

    print(f"\n  Clases definidas:")
    for class_name, priority in PRIORITY_ORDER:
        class_id, variations = CLASS_MAPPING[class_name]
        print(f"    ID {class_id}: {class_name:20s} (capas: {', '.join(variations)})")

    # Buscar municipios
    municipios = [d for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d))]

    print(f"\n  Municipios encontrados: {len(municipios)}")
    for i, mun in enumerate(municipios, 1):
        print(f"    {i}. {mun}")

    # Procesar cada municipio
    resultados = {}
    for municipio in municipios:
        municipio_path = os.path.join(base_dir, municipio)
        success = process_municipality(municipio_path, output_vectors_dir, output_rasters_dir)
        resultados[municipio] = success

    # Resumen final
    print(f"\n{'='*80}")
    print(" RESUMEN FINAL")
    print(f"{'='*80}")
    exitosos = sum(1 for v in resultados.values() if v)
    print(f"  Total municipios: {len(resultados)}")
    print(f"  Exitosos: {exitosos}")
    print(f"  Fallidos: {len(resultados) - exitosos}")

    if exitosos < len(resultados):
        print(f"\n  Municipios con errores:")
        for mun, success in resultados.items():
            if not success:
                print(f"    - {mun}")

    print(f"\n  Archivos de salida:")
    print(f"    Vectores: {output_vectors_dir}")
    print(f"    Rasters: {output_rasters_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
