"""
Script para procesar Vías con buffer, unión y erase:
1. Buffer 3m a TODAS las líneas
2. Union de todos los buffers
3. Explotar polígonos
4. Erase con Construccion_R
5. Erase con Zona_Verde
6. Erase con Zona_Dura
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString
from shapely.ops import unary_union
import numpy as np
from tqdm import tqdm
import json


def create_buffers(gdf, buffer_distance=1.5):
    """
    Crea buffers de todas las líneas

    Args:
        gdf: GeoDataFrame con líneas
        buffer_distance: Distancia del buffer (1.5m = 3m total)

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

    # Añadir ID único
    union_gdf['poly_id'] = range(len(union_gdf))

    print(f"  Polígonos resultantes: {len(union_gdf)}")

    return union_gdf


def erase_with_layers(polygons_gdf, construccion_path, zona_verde_path, zona_dura_path):
    """
    Hace erase a los polígonos con Construccion_R, Zona_Verde y Zona_Dura

    Args:
        polygons_gdf: GeoDataFrame con polígonos de vías
        construccion_path: Path al archivo Construccion_R.gpkg
        zona_verde_path: Path al archivo Zona_Verde.gpkg
        zona_dura_path: Path al archivo Zona_Dura.gpkg

    Returns:
        GeoDataFrame con polígonos después del erase
    """
    print(f"\n[PASO 4] Erase con capas de Construccion_R, Zona_Verde y Zona_Dura...")

    result_gdf = polygons_gdf.copy()

    # Lista de capas para hacer erase
    erase_layers = [
        ('Construccion_R', construccion_path),
        ('Zona_Verde', zona_verde_path),
        ('Zona_Dura', zona_dura_path)
    ]

    for layer_name, layer_path in erase_layers:
        if not layer_path.exists():
            print(f"  [ADVERTENCIA] Archivo no encontrado: {layer_path}")
            continue

        print(f"\n  Procesando erase con {layer_name}...")
        try:
            # Leer capa
            erase_gdf = gpd.read_file(layer_path)
            print(f"    Features en {layer_name}: {len(erase_gdf)}")

            if len(erase_gdf) == 0:
                print(f"    [INFO] {layer_name} está vacía. Saltando...")
                continue

            # Validar y corregir geometrías inválidas
            print(f"    Validando geometrías...")
            invalid_count = (~erase_gdf.geometry.is_valid).sum()
            if invalid_count > 0:
                print(f"    [ADVERTENCIA] {invalid_count} geometrías inválidas. Corrigiendo...")
                erase_gdf['geometry'] = erase_gdf.geometry.buffer(0)

            # Filtrar geometrías nulas o vacías
            erase_gdf = erase_gdf[~erase_gdf.geometry.isna()].copy()
            erase_gdf = erase_gdf[~erase_gdf.geometry.is_empty].copy()

            if len(erase_gdf) == 0:
                print(f"    [INFO] {layer_name} no tiene geometrías válidas. Saltando...")
                continue

            # Unir todas las geometrías de la capa de erase
            print(f"    Uniendo geometrías...")
            erase_union = unary_union(erase_gdf.geometry)

            # Validar geometrías de los polígonos antes del erase
            print(f"    Validando polígonos de vías...")
            invalid_poly_count = (~result_gdf.geometry.is_valid).sum()
            if invalid_poly_count > 0:
                print(f"    [ADVERTENCIA] {invalid_poly_count} polígonos inválidos. Corrigiendo...")
                result_gdf['geometry'] = result_gdf.geometry.buffer(0)

            # Hacer erase (difference) para cada polígono
            erased_polygons = []
            for idx, poly_row in tqdm(result_gdf.iterrows(), total=len(result_gdf), desc=f"    Erase con {layer_name}"):
                polygon = poly_row.geometry

                # Saltar geometrías nulas o vacías
                if polygon is None or polygon.is_empty:
                    continue

                # Verificar si intersecta con la capa de erase
                if not polygon.intersects(erase_union):
                    # No intersecta, mantener el polígono completo
                    erased_polygons.append(polygon)
                    continue

                # Hacer difference
                try:
                    erased = polygon.difference(erase_union)

                    # Verificar que el resultado no esté vacío
                    if erased.is_empty or erased.area < 0.01:
                        continue

                    # Manejar diferentes tipos de geometría resultante
                    if erased.geom_type == 'Polygon':
                        erased_polygons.append(erased)
                    elif erased.geom_type == 'MultiPolygon':
                        # Explotar MultiPolygon en polígonos individuales
                        for poly in erased.geoms:
                            if poly.area >= 0.01:
                                erased_polygons.append(poly)
                    elif erased.geom_type == 'GeometryCollection':
                        # Extraer solo polígonos de la colección
                        for geom in erased.geoms:
                            if geom.geom_type == 'Polygon' and geom.area >= 0.01:
                                erased_polygons.append(geom)
                            elif geom.geom_type == 'MultiPolygon':
                                for poly in geom.geoms:
                                    if poly.area >= 0.01:
                                        erased_polygons.append(poly)
                    else:
                        # Tipo inesperado, mantener original
                        erased_polygons.append(polygon)

                except Exception as e:
                    print(f"      Error en difference para polígono {idx}: {e}")
                    # En caso de error, mantener el polígono original
                    erased_polygons.append(polygon)

            # Actualizar result_gdf con los polígonos después del erase
            if len(erased_polygons) > 0:
                result_gdf = gpd.GeoDataFrame(geometry=erased_polygons, crs=polygons_gdf.crs)
                print(f"    Polígonos después de erase con {layer_name}: {len(result_gdf)}")
            else:
                print(f"    [ADVERTENCIA] No quedan polígonos después de erase con {layer_name}")
                result_gdf = gpd.GeoDataFrame(geometry=[], crs=polygons_gdf.crs)

        except Exception as e:
            print(f"  [ERROR] Error procesando {layer_name}: {e}")
            continue

    # Añadir área y ID
    if len(result_gdf) > 0:
        result_gdf['area_m2'] = result_gdf.geometry.area
        result_gdf['poly_id'] = range(len(result_gdf))

    print(f"\n  Polígonos finales después de todos los erases: {len(result_gdf)}")

    return result_gdf


def process_vias_municipio(input_path, municipio_dir, output_dir, buffer_distance=1.5):
    """
    Procesa un municipio con el flujo completo
    """
    print("=" * 80)
    print("PROCESAMIENTO DE VÍAS - BUFFER + ERASE")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Municipio dir: {municipio_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Buffer: {buffer_distance*2}m ({buffer_distance}m cada lado)")

    # Leer capa original
    print("\n[INICIO] Leyendo capa de Vías...")
    gdf_vias = gpd.read_file(input_path)
    print(f"  Features: {len(gdf_vias)}")
    print(f"  Tipo de geometría: {gdf_vias.geom_type.unique()}")
    print(f"  CRS: {gdf_vias.crs}")

    # Filtrar geometrías nulas
    null_geoms = gdf_vias.geometry.isna().sum()
    if null_geoms > 0:
        print(f"  [ADVERTENCIA] Se encontraron {null_geoms} geometrías nulas. Se eliminarán.")
        gdf_vias = gdf_vias[~gdf_vias.geometry.isna()].copy()
        print(f"  Features válidas: {len(gdf_vias)}")

    # 1. Crear buffers de TODAS las líneas (sin filtro por VTipo)
    buffers_gdf = create_buffers(gdf_vias, buffer_distance)

    # 2-3. Union y explotar
    union_gdf = union_and_explode_buffers(buffers_gdf)

    # 4. Erase con Construccion_R, Zona_Verde y Zona_Dura
    construccion_path = municipio_dir / "Construccion_R.gpkg"
    zona_verde_path = municipio_dir / "Zona_Verde.gpkg"
    zona_dura_path = municipio_dir / "Zona_Dura.gpkg"

    final_gdf = erase_with_layers(union_gdf, construccion_path, zona_verde_path, zona_dura_path)

    # Guardar resultados
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Polígonos finales
    if len(final_gdf) > 0:
        output_final = output_dir / "Via_polygons_final.gpkg"
        final_gdf.to_file(output_final, driver='GPKG')
        print(f"  [OK] Polígonos finales: {output_final}")

    # Polígonos antes del erase (para comparación)
    if len(union_gdf) > 0:
        output_union = output_dir / "Via_polygons_before_erase.gpkg"
        union_gdf.to_file(output_union, driver='GPKG')
        print(f"  [OK] Polígonos antes del erase: {output_union}")

    # Reporte
    report = {
        'input_file': str(input_path),
        'buffer_distance': buffer_distance * 2,
        'original_lines': len(gdf_vias),
        'polygons_after_union': len(union_gdf),
        'polygons_final': len(final_gdf)
    }

    if len(final_gdf) > 0:
        report['final_stats'] = {
            'total_area_m2': float(final_gdf['area_m2'].sum()),
            'avg_area_m2': float(final_gdf['area_m2'].mean()),
            'min_area_m2': float(final_gdf['area_m2'].min()),
            'max_area_m2': float(final_gdf['area_m2'].max())
        }

    output_report = output_dir / "Via_processing_report.json"
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Reporte JSON: {output_report}")

    # Resumen
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"\nLíneas originales: {len(gdf_vias)}")
    print(f"Polígonos después de unión: {len(union_gdf)}")
    print(f"Polígonos finales (después de erase): {len(final_gdf)}")

    if len(final_gdf) > 0:
        print(f"\nEstadísticas finales:")
        print(f"  - Área total: {final_gdf['area_m2'].sum():.2f} m²")
        print(f"  - Área promedio: {final_gdf['area_m2'].mean():.2f} m²")

    print("\n" + "="*80)
    print("PROCESAMIENTO COMPLETADO")
    print("="*80)

    return report


def main():
    """
    Función principal - Procesa todos los municipios
    """
    base_input_dir = Path("data/interim/vectores_cleaned")
    base_output_dir = Path("data/processed/vias_buffer_erase")
    buffer_distance = 1.5  # 3m total (1.5m cada lado)

    municipios = ["madrid", "mosquera", "tabio", "tocancipa"]

    print("="*80)
    print("PROCESAMIENTO BATCH - VÍAS BUFFER + ERASE")
    print("="*80)
    print(f"\nMunicipios a procesar: {len(municipios)}")
    print(f"Buffer: {buffer_distance*2}m ({buffer_distance}m cada lado)\n")

    reports = {}

    for municipio in municipios:
        print("\n" + "="*80)
        print(f"PROCESANDO MUNICIPIO: {municipio.upper()}")
        print("="*80)

        municipio_dir = base_input_dir / municipio
        input_path = municipio_dir / "Via.gpkg"
        output_dir = base_output_dir / municipio

        if not input_path.exists():
            print(f"[ADVERTENCIA] Archivo no encontrado: {input_path}")
            continue

        try:
            report = process_vias_municipio(input_path, municipio_dir, output_dir, buffer_distance)
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
