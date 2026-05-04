"""
Script para procesar Cercas con buffer:
1. Buffer 1m a TODAS las cercas
2. Union de todos los buffers
3. Explotar polígonos
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import numpy as np
from tqdm import tqdm
import json


def create_buffers(gdf, buffer_distance=0.5):
    """
    Crea buffers de todas las líneas de cerca

    Args:
        gdf: GeoDataFrame con líneas
        buffer_distance: Distancia del buffer (0.5m = 1m total)

    Returns:
        GeoDataFrame con buffers individuales
    """
    print(f"\n[PASO 1] Creando buffers de {buffer_distance*2}m (cap_style=square)...")

    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(
        distance=buffer_distance,
        cap_style=3  # square
    )

    print(f"  Buffers creados: {len(gdf_buffered)}")

    return gdf_buffered


def union_and_explode_buffers(buffers_gdf):
    """
    Une todos los buffers y explota en polígonos individuales

    Args:
        buffers_gdf: GeoDataFrame con buffers

    Returns:
        GeoDataFrame con polígonos unidos y explodidos
    """
    print(f"\n[PASO 2] Unión de todos los buffers (unary_union)...")

    # Unir todos los buffers
    union_geom = unary_union(buffers_gdf.geometry)

    print(f"  Geometría resultante: {union_geom.geom_type}")

    print(f"\n[PASO 3] Explotando polígonos...")

    # Convertir a GeoDataFrame y explotar
    if union_geom.geom_type == 'Polygon':
        union_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=buffers_gdf.crs)
    elif union_geom.geom_type == 'MultiPolygon':
        union_gdf = gpd.GeoDataFrame(geometry=list(union_geom.geoms), crs=buffers_gdf.crs)
    else:
        # GeometryCollection u otro tipo
        geoms = []
        if hasattr(union_geom, 'geoms'):
            for geom in union_geom.geoms:
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    if geom.geom_type == 'Polygon':
                        geoms.append(geom)
                    else:
                        geoms.extend(list(geom.geoms))
        union_gdf = gpd.GeoDataFrame(geometry=geoms, crs=buffers_gdf.crs)

    # Añadir ID único y área
    union_gdf['poly_id'] = range(len(union_gdf))
    union_gdf['area_m2'] = union_gdf.geometry.area

    print(f"  Polígonos resultantes: {len(union_gdf)}")

    return union_gdf


def process_cercas_municipio(input_path, output_dir, buffer_distance=0.5):
    """
    Procesa un municipio con el flujo completo
    """
    print("=" * 80)
    print("PROCESAMIENTO DE CERCAS - BUFFER")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Buffer: {buffer_distance*2}m ({buffer_distance}m cada lado)")

    # Leer capa original
    print("\n[INICIO] Leyendo capa de Cercas...")
    gdf_cercas = gpd.read_file(input_path)
    print(f"  Features: {len(gdf_cercas)}")
    print(f"  Tipo de geometría: {gdf_cercas.geom_type.unique()}")
    print(f"  CRS: {gdf_cercas.crs}")

    # Filtrar geometrías nulas
    null_geoms = gdf_cercas.geometry.isna().sum()
    if null_geoms > 0:
        print(f"  [ADVERTENCIA] Se encontraron {null_geoms} geometrías nulas. Se eliminarán.")
        gdf_cercas = gdf_cercas[~gdf_cercas.geometry.isna()].copy()
        print(f"  Features válidas: {len(gdf_cercas)}")

    # Validar y corregir geometrías inválidas
    print("\n  Validando geometrías...")
    invalid_count = (~gdf_cercas.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"  [ADVERTENCIA] {invalid_count} geometrías inválidas. Corrigiendo...")
        gdf_cercas['geometry'] = gdf_cercas.geometry.buffer(0)

    if len(gdf_cercas) == 0:
        print("\n[ERROR] No hay geometrías válidas para procesar!")
        return None

    # 1. Crear buffers de TODAS las cercas
    buffers_gdf = create_buffers(gdf_cercas, buffer_distance)

    # 2-3. Union y explotar
    final_gdf = union_and_explode_buffers(buffers_gdf)

    # Guardar resultados
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Polígonos finales
    if len(final_gdf) > 0:
        output_final = output_dir / "Cerca_polygons_final.gpkg"
        final_gdf.to_file(output_final, driver='GPKG')
        print(f"  [OK] Polígonos finales: {output_final}")

    # Reporte
    report = {
        'input_file': str(input_path),
        'buffer_distance': buffer_distance * 2,
        'original_lines': len(gdf_cercas),
        'polygons_final': len(final_gdf)
    }

    if len(final_gdf) > 0:
        report['final_stats'] = {
            'total_area_m2': float(final_gdf['area_m2'].sum()),
            'avg_area_m2': float(final_gdf['area_m2'].mean()),
            'min_area_m2': float(final_gdf['area_m2'].min()),
            'max_area_m2': float(final_gdf['area_m2'].max())
        }

    output_report = output_dir / "Cerca_processing_report.json"
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Reporte JSON: {output_report}")

    # Resumen
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"\nLíneas originales: {len(gdf_cercas)}")
    print(f"Polígonos finales: {len(final_gdf)}")

    if len(final_gdf) > 0:
        print(f"\nEstadísticas finales:")
        print(f"  - Área total: {final_gdf['area_m2'].sum():.2f} m²")
        print(f"  - Área promedio: {final_gdf['area_m2'].mean():.2f} m²")
        print(f"  - Área mínima: {final_gdf['area_m2'].min():.2f} m²")
        print(f"  - Área máxima: {final_gdf['area_m2'].max():.2f} m²")

    print("\n" + "="*80)
    print("PROCESAMIENTO COMPLETADO")
    print("="*80)

    return report


def main():
    """
    Función principal - Procesa todos los municipios
    """
    base_input_dir = Path("data/interim/vectores_cleaned")
    base_output_dir = Path("data/processed/cercas_buffer")
    buffer_distance = 0.5  # 1m total (0.5m cada lado)

    municipios = ["madrid", "mosquera", "tabio", "tocancipa"]

    print("="*80)
    print("PROCESAMIENTO BATCH - CERCAS BUFFER")
    print("="*80)
    print(f"\nMunicipios a procesar: {len(municipios)}")
    print(f"Buffer: {buffer_distance*2}m ({buffer_distance}m cada lado)\n")

    reports = {}

    for municipio in municipios:
        print("\n" + "="*80)
        print(f"PROCESANDO MUNICIPIO: {municipio.upper()}")
        print("="*80)

        input_path = base_input_dir / municipio / "Cerca.gpkg"
        output_dir = base_output_dir / municipio

        if not input_path.exists():
            print(f"[ADVERTENCIA] Archivo no encontrado: {input_path}")
            continue

        try:
            report = process_cercas_municipio(input_path, output_dir, buffer_distance)
            if report:
                reports[municipio] = report
        except Exception as e:
            print(f"\n[ERROR] Error procesando {municipio}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Resumen global
    print("\n\n" + "="*80)
    print("RESUMEN GLOBAL - TODOS LOS MUNICIPIOS")
    print("="*80)

    if len(reports) > 0:
        print(f"\nMunicipios procesados: {len(reports)}")

        for municipio, report in reports.items():
            print(f"\n{municipio.upper()}:")
            print(f"  - Líneas originales: {report['original_lines']}")
            print(f"  - Polígonos finales: {report['polygons_final']}")
            if 'final_stats' in report:
                print(f"  - Área total: {report['final_stats']['total_area_m2']:.2f} m²")
    else:
        print("\n[ADVERTENCIA] No se procesó ningún municipio.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
