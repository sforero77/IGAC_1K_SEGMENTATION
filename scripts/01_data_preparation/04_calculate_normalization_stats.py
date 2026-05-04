"""
Script para calcular estadísticas de normalización (μ, σ) SOLO en zonas TRAIN

OBJETIVO:
- Leer ortofotos SOLO en zonas TRAIN (usando train_zones.gpkg)
- Calcular media (μ) y desviación estándar (σ) por banda
- Guardar normalization_stats.json
- NO reescribir ortofotos (solo extraer stats)

METODOLOGÍA:
1. Leer cuadros TRAIN desde train_zones.gpkg
2. Para cada ortofoto, identificar cuadros TRAIN que la intersectan
3. Leer píxeles SOLO en esas ventanas
4. Calcular μ y σ por banda (R, G, B, NIR)
5. Guardar stats en JSON

SALIDAS:
- data/interim/normalization_stats.json
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios
ORTOFOTOS_DIR = r"D:\IGAC_1K_SEGMENTATION\data\raw\ortofotos"
TRAIN_ZONES_PATH = r"D:\IGAC_1K_SEGMENTATION\data\processed\spatial_splits\train_zones.gpkg"
OUTPUT_JSON = r"D:\IGAC_1K_SEGMENTATION\data\interim\normalization_stats.json"

# Ciudades
CIUDADES = ['madrid', 'mosquera', 'tabio', 'tocancipa']

# Número de bandas (siempre 3: RGB)
N_BANDS = 3  # Solo RGB (ignorar NIR si existe)


# ============================================================================
# FUNCIONES
# ============================================================================

def find_ortofoto_file(ciudad_dir):
    """
    Encuentra el archivo de ortofoto en el directorio de la ciudad

    Args:
        ciudad_dir: Directorio de la ciudad

    Returns:
        Path del archivo .tif o None
    """
    ciudad_path = Path(ciudad_dir)
    tif_files = list(ciudad_path.glob("*.tif"))

    # Filtrar archivos .ovr (overview) y .aux.xml
    tif_files = [f for f in tif_files if not f.name.endswith('.ovr') and '.aux.' not in f.name]

    if tif_files:
        return str(tif_files[0])
    return None


def get_train_zones_for_ortofoto(train_zones_gdf, ortofoto_bounds, ortofoto_crs):
    """
    Filtra cuadros TRAIN que intersectan con una ortofoto

    Args:
        train_zones_gdf: GeoDataFrame con cuadros TRAIN
        ortofoto_bounds: Bounds de la ortofoto (minx, miny, maxx, maxy)
        ortofoto_crs: CRS de la ortofoto

    Returns:
        GeoDataFrame con cuadros que intersectan
    """
    from shapely.geometry import box

    # Crear bbox de la ortofoto
    ortofoto_bbox = box(*ortofoto_bounds)

    # Reproyectar train_zones si es necesario
    if train_zones_gdf.crs != ortofoto_crs:
        train_zones_gdf = train_zones_gdf.to_crs(ortofoto_crs)

    # Filtrar cuadros que intersectan
    intersecting = train_zones_gdf[train_zones_gdf.intersects(ortofoto_bbox)]

    return intersecting


def extract_pixels_from_train_zones(ortofoto_path, train_zones_gdf, max_pixels_per_ortofoto=1000000):
    """
    Extrae píxeles de una ortofoto SOLO en zonas TRAIN

    Args:
        ortofoto_path: Path a la ortofoto
        train_zones_gdf: GeoDataFrame con cuadros TRAIN que intersectan
        max_pixels_per_ortofoto: Máximo de píxeles a extraer (para eficiencia)

    Returns:
        numpy array de shape (n_pixels, n_bands)
    """
    all_pixels = []
    total_pixels_extracted = 0

    with rasterio.open(ortofoto_path) as src:
        # Reproyectar train_zones si es necesario
        if train_zones_gdf.crs != src.crs:
            train_zones_gdf = train_zones_gdf.to_crs(src.crs)

        # Para cada cuadro TRAIN
        for idx, zone in train_zones_gdf.iterrows():
            if total_pixels_extracted >= max_pixels_per_ortofoto:
                break

            # Obtener bounds del cuadro
            zone_bounds = zone.geometry.bounds

            try:
                # Calcular ventana de lectura
                window = src.window(*zone_bounds)

                # Asegurar que la ventana esté dentro de los límites del raster
                window = window.intersection(Window(0, 0, src.width, src.height))

                if window.width <= 0 or window.height <= 0:
                    continue

                # Leer datos (solo primeras 3 bandas: RGB)
                data = src.read([1, 2, 3], window=window)  # Shape: (3, height, width)

                # Transponer a (height, width, bands)
                data = np.transpose(data, (1, 2, 0))

                # Reshape a (n_pixels, bands)
                pixels = data.reshape(-1, data.shape[2])

                # Filtrar píxeles con valor 0 (NoData/background)
                # Asumiendo que NoData es cuando todas las bandas son 0
                valid_mask = ~np.all(pixels == 0, axis=1)
                pixels = pixels[valid_mask]

                if len(pixels) > 0:
                    # Si hay demasiados píxeles, submuestrear
                    remaining_budget = max_pixels_per_ortofoto - total_pixels_extracted
                    if len(pixels) > remaining_budget:
                        indices = np.random.choice(len(pixels), remaining_budget, replace=False)
                        pixels = pixels[indices]

                    all_pixels.append(pixels)
                    total_pixels_extracted += len(pixels)

            except Exception as e:
                print(f"      Advertencia: Error leyendo ventana: {e}")
                continue

    if all_pixels:
        return np.vstack(all_pixels)
    else:
        # Si N_BANDS es None, devolver array vacío genérico
        if N_BANDS is not None:
            return np.array([]).reshape(0, N_BANDS)
        else:
            return np.array([])


def calculate_normalization_stats(train_zones_gdf, ciudades, ortofotos_dir):
    """
    Calcula estadísticas de normalización (μ, σ) por banda

    Args:
        train_zones_gdf: GeoDataFrame con cuadros TRAIN
        ciudades: Lista de nombres de ciudades
        ortofotos_dir: Directorio con ortofotos

    Returns:
        dict con 'mean' y 'std' por banda
    """
    print("\n  Extrayendo píxeles de zonas TRAIN...")

    all_pixels_per_band = [[] for _ in range(N_BANDS)]

    for ciudad in ciudades:
        print(f"\n    Procesando {ciudad}...")

        # Encontrar archivo de ortofoto
        ciudad_dir = os.path.join(ortofotos_dir, ciudad)
        ortofoto_path = find_ortofoto_file(ciudad_dir)

        if not ortofoto_path:
            print(f"      Advertencia: No se encontró ortofoto en {ciudad_dir}")
            continue

        print(f"      Ortofoto: {Path(ortofoto_path).name}")

        # Abrir ortofoto para obtener bounds y CRS
        with rasterio.open(ortofoto_path) as src:
            ortofoto_bounds = src.bounds
            ortofoto_crs = src.crs
            n_bands_current = src.count
            print(f"      Dimensiones: {src.width} x {src.height}, Bandas: {n_bands_current} (usando solo RGB)")

        # Filtrar cuadros TRAIN que intersectan con esta ortofoto
        intersecting_zones = get_train_zones_for_ortofoto(
            train_zones_gdf, ortofoto_bounds, ortofoto_crs
        )

        print(f"      Cuadros TRAIN intersectados: {len(intersecting_zones)}")

        if len(intersecting_zones) == 0:
            continue

        # Extraer píxeles
        pixels = extract_pixels_from_train_zones(ortofoto_path, intersecting_zones)

        print(f"      Píxeles extraídos: {len(pixels):,}")

        # Acumular píxeles por banda
        if len(pixels) > 0:
            for band_idx in range(N_BANDS):
                all_pixels_per_band[band_idx].append(pixels[:, band_idx])

    # Calcular estadísticas globales
    print("\n  Calculando estadísticas globales...")

    stats = {
        'mean': [],
        'std': [],
        'band_names': ['Red', 'Green', 'Blue'],
        'n_bands': N_BANDS,
        'n_pixels_sampled': 0
    }

    for band_idx in range(N_BANDS):
        if all_pixels_per_band[band_idx]:
            # Concatenar todos los píxeles de esta banda
            band_pixels = np.concatenate(all_pixels_per_band[band_idx])

            # Calcular μ y σ
            mean = float(np.mean(band_pixels))
            std = float(np.std(band_pixels))

            stats['mean'].append(mean)
            stats['std'].append(std)

            if band_idx == 0:
                stats['n_pixels_sampled'] = len(band_pixels)

            band_name = stats['band_names'][band_idx]
            print(f"    {band_name:6s}: mean = {mean:8.2f}, std = {std:8.2f}")
        else:
            stats['mean'].append(0.0)
            stats['std'].append(1.0)
            print(f"    Banda {band_idx}: Sin datos, usando mean=0, std=1")

    return stats


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal
    """
    print("="*80)
    print("CÁLCULO DE ESTADÍSTICAS DE NORMALIZACIÓN")
    print("="*80)

    # 1. Cargar cuadros TRAIN
    print("\n[1/3] Cargando cuadros TRAIN...")

    if not os.path.exists(TRAIN_ZONES_PATH):
        print(f"\n  ERROR: No se encontró {TRAIN_ZONES_PATH}")
        print("  Ejecuta primero: python scripts/03_create_spatial_splits.py")
        return

    train_zones_gdf = gpd.read_file(TRAIN_ZONES_PATH)
    print(f"  Cuadros TRAIN cargados: {len(train_zones_gdf)}")
    print(f"  CRS: {train_zones_gdf.crs}")

    # 2. Calcular estadísticas
    print("\n[2/3] Calculando estadísticas de normalización...")

    stats = calculate_normalization_stats(train_zones_gdf, CIUDADES, ORTOFOTOS_DIR)

    # 3. Guardar JSON
    print("\n[3/3] Guardando estadísticas...")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  OK Guardado: {OUTPUT_JSON}")

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print(f"  Pixeles muestreados: {stats['n_pixels_sampled']:,}")
    print(f"\n  Estadisticas por banda:")
    for i, band_name in enumerate(stats['band_names']):
        print(f"    {band_name:6s}: mean = {stats['mean'][i]:8.2f}, std = {stats['std'][i]:8.2f}")

    print(f"\n  Archivo generado: {OUTPUT_JSON}")
    print("\n  Próximo paso:")
    print("    python scripts/06_create_patches.py")
    print("="*80)


if __name__ == "__main__":
    # Fijar semilla para reproducibilidad en submuestreo
    np.random.seed(42)
    main()
