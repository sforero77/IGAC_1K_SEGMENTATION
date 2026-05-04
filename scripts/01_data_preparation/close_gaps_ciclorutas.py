"""
Script consolidado para cerrar huecos en CICLORUTAS con doble pasada
Flujo estilo ArcGIS con dos iteraciones automáticas
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Point, LineString, MultiLineString
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json


def extract_dangle_points(gdf):
    """
    Equivalente a Feature Vertices To Points con Point Type = DANGLE
    """
    # 1. Extraer todos los endpoints
    endpoint_data = []

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  Extrayendo endpoints"):
        geom = row.geometry

        # Skip null geometries
        if geom is None or pd.isna(geom):
            continue

        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            endpoint_data.append({
                'point': Point(coords[0]),
                'feature_idx': idx,
                'vertex_type': 'start'
            })
            endpoint_data.append({
                'point': Point(coords[-1]),
                'feature_idx': idx,
                'vertex_type': 'end'
            })
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                endpoint_data.append({
                    'point': Point(coords[0]),
                    'feature_idx': idx,
                    'vertex_type': 'start'
                })
                endpoint_data.append({
                    'point': Point(coords[-1]),
                    'feature_idx': idx,
                    'vertex_type': 'end'
                })

    # 2. Identificar dangles (puntos que solo se conectan con 1 línea)
    snap_tolerance = 0.1  # 10 cm
    point_connections = defaultdict(list)

    for i, ep_data in enumerate(tqdm(endpoint_data, desc="  Identificando dangles")):
        point = ep_data['point']
        key = (round(point.x / snap_tolerance) * snap_tolerance,
               round(point.y / snap_tolerance) * snap_tolerance)
        point_connections[key].append(i)

    # Filtrar solo dangles (1 conexión)
    dangle_indices = []
    for key, indices in point_connections.items():
        if len(indices) == 1:
            dangle_indices.append(indices[0])

    dangles = [endpoint_data[i] for i in dangle_indices]

    # Crear GeoDataFrame
    if len(dangles) == 0:
        # Crear GeoDataFrame vacío con las columnas correctas
        dangles_gdf = gpd.GeoDataFrame({
            'point': [],
            'feature_idx': [],
            'vertex_type': [],
            'OBJECTID': []
        }, geometry='point', crs=gdf.crs)
    else:
        dangles_gdf = gpd.GeoDataFrame(dangles, geometry='point', crs=gdf.crs)
        dangles_gdf['OBJECTID'] = range(len(dangles_gdf))

    return dangles_gdf, len(endpoint_data)


def near_analysis(dangles_gdf):
    """
    Equivalente a Near Tool (Analysis Tools > Proximity)
    """
    near_fid = []
    near_dist = []

    for idx, dangle in tqdm(dangles_gdf.iterrows(), total=len(dangles_gdf), desc="  Calculando distancias"):
        other_dangles = dangles_gdf[dangles_gdf.index != idx]

        if len(other_dangles) == 0:
            near_fid.append(-1)
            near_dist.append(np.inf)
            continue

        distances = other_dangles.geometry.distance(dangle['point'])
        min_idx = distances.idxmin()
        min_dist = distances.min()

        near_fid.append(other_dangles.loc[min_idx, 'OBJECTID'])
        near_dist.append(min_dist)

    dangles_gdf['NEAR_FID'] = near_fid
    dangles_gdf['NEAR_DIST'] = near_dist

    return dangles_gdf


def filter_valid_pairs(dangles_gdf, max_distance):
    """
    Filtra pares válidos basándose en la distancia
    """
    valid_dangles = dangles_gdf[dangles_gdf['NEAR_DIST'] < max_distance].copy()
    return valid_dangles


def add_geometry_attributes(dangles_gdf):
    """
    Añade coordenadas POINT_X y POINT_Y
    """
    dangles_gdf['POINT_X'] = dangles_gdf.geometry.x
    dangles_gdf['POINT_Y'] = dangles_gdf.geometry.y
    return dangles_gdf


def self_join_and_xy_to_line(dangles_gdf, original_gdf):
    """
    Self Join + XY to Line
    """
    dangles_gdf = dangles_gdf.reset_index(drop=True)

    dangles_for_join = dangles_gdf[['OBJECTID', 'POINT_X', 'POINT_Y']].copy()
    dangles_for_join.columns = ['NEAR_FID', 'POINT_X_NEAR', 'POINT_Y_NEAR']

    joined = dangles_gdf.merge(dangles_for_join, on='NEAR_FID', how='left')

    closure_lines = []
    processed_pairs = set()

    for idx, row in tqdm(joined.iterrows(), total=len(joined), desc="  Creando lineas de cierre"):
        if pd.isna(row['POINT_X_NEAR']):
            continue

        pair_key = tuple(sorted([row['OBJECTID'], row['NEAR_FID']]))
        if pair_key in processed_pairs:
            continue

        processed_pairs.add(pair_key)

        line = LineString([
            (row['POINT_X'], row['POINT_Y']),
            (row['POINT_X_NEAR'], row['POINT_Y_NEAR'])
        ])

        feature_attrs = {}
        if 'feature_idx' in row and row['feature_idx'] < len(original_gdf):
            original_feature = original_gdf.iloc[row['feature_idx']]
            for col in original_gdf.columns:
                if col != 'geometry':
                    feature_attrs[col] = original_feature[col]

        closure_lines.append({
            'geometry': line,
            'CLOSURE_ID': f"CLOSURE_{row['OBJECTID']}_{row['NEAR_FID']}",
            'CLOSURE_DIST': row['NEAR_DIST'],
            'IS_CLOSURE': 1,
            **feature_attrs
        })

    closures_gdf = gpd.GeoDataFrame(closure_lines, crs=original_gdf.crs)
    return closures_gdf


def merge_and_explode(original_gdf, closures_gdf):
    """
    Merge + Explode
    """
    original_gdf_copy = original_gdf.copy()
    original_gdf_copy['IS_CLOSURE'] = 0

    for col in original_gdf.columns:
        if col not in closures_gdf.columns and col != 'geometry':
            closures_gdf[col] = None

    for col in closures_gdf.columns:
        if col not in original_gdf_copy.columns and col != 'geometry':
            original_gdf_copy[col] = None

    merged_gdf = pd.concat([original_gdf_copy, closures_gdf], ignore_index=True)
    exploded_gdf = merged_gdf.explode(index_parts=False).reset_index(drop=True)
    exploded_gdf['FID'] = range(len(exploded_gdf))

    return exploded_gdf


def single_pass(gdf, max_distance, pass_name):
    """
    Ejecuta una sola pasada del algoritmo
    """
    print(f"\n{'='*80}")
    print(f"{pass_name}")
    print(f"{'='*80}")

    # 1. Extract dangle points
    print(f"\n[PASO 1] Extrayendo puntos DANGLE...")
    dangles_gdf, total_endpoints = extract_dangle_points(gdf)
    print(f"  Total de endpoints: {total_endpoints}")
    print(f"  Puntos DANGLE encontrados: {len(dangles_gdf)}")

    if len(dangles_gdf) == 0:
        print("\n[INFO] No se encontraron dangles. No se necesitan cierres.")
        return gdf, dangles_gdf, None, {
            'dangles_found': 0,
            'closures_created': 0,
            'total_length_closed': 0,
            'avg_closure_length': 0,
            'min_closure_length': 0,
            'max_closure_length': 0
        }

    # 2. Near analysis
    print(f"\n[PASO 2] Análisis NEAR...")
    dangles_gdf = near_analysis(dangles_gdf)
    valid_distances = [d for d in dangles_gdf['NEAR_DIST'] if d != np.inf]
    if valid_distances:
        print(f"  Distancia mínima: {min(valid_distances):.2f} m")
        print(f"  Distancia máxima: {max(valid_distances):.2f} m")
        print(f"  Distancia promedio: {np.mean(valid_distances):.2f} m")

    # 3. Filter valid pairs
    print(f"\n[PASO 3] Filtrando pares válidos (NEAR_DIST < {max_distance} m)...")
    valid_dangles = filter_valid_pairs(dangles_gdf, max_distance)
    print(f"  Pares válidos encontrados: {len(valid_dangles)}")

    if len(valid_dangles) == 0:
        print("\n[INFO] No se encontraron pares válidos dentro de la tolerancia.")
        return gdf, dangles_gdf, None, {
            'dangles_found': len(dangles_gdf),
            'closures_created': 0,
            'total_length_closed': 0,
            'avg_closure_length': 0,
            'min_closure_length': 0,
            'max_closure_length': 0
        }

    # 4. Add geometry attributes
    print(f"\n[PASO 4] Añadiendo atributos geométricos...")
    valid_dangles = add_geometry_attributes(valid_dangles)

    # 5. Self join + XY to Line
    print(f"\n[PASO 5] Auto-join y creación de líneas...")
    closures_gdf = self_join_and_xy_to_line(valid_dangles, gdf)
    print(f"  Líneas de cierre creadas: {len(closures_gdf)}")

    # 6-7. Merge + Explode
    print(f"\n[PASO 6-7] Merge y Explode...")
    final_gdf = merge_and_explode(gdf, closures_gdf)
    print(f"  Features después del merge: {len(final_gdf)}")

    # Estadísticas
    stats = {
        'dangles_found': len(dangles_gdf),
        'closures_created': len(closures_gdf),
        'total_length_closed': closures_gdf['CLOSURE_DIST'].sum() if len(closures_gdf) > 0 else 0,
        'avg_closure_length': closures_gdf['CLOSURE_DIST'].mean() if len(closures_gdf) > 0 else 0,
        'min_closure_length': closures_gdf['CLOSURE_DIST'].min() if len(closures_gdf) > 0 else 0,
        'max_closure_length': closures_gdf['CLOSURE_DIST'].max() if len(closures_gdf) > 0 else 0
    }

    return final_gdf, dangles_gdf, closures_gdf, stats


def two_pass_processing(input_path, output_dir, max_distance=40.0):
    """
    Ejecuta dos pasadas completas del algoritmo
    """
    print("=" * 80)
    print("PROCESAMIENTO DOBLE PASADA - CIERRE DE HUECOS EN CICLORUTAS")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Max distance: {max_distance} m")

    # Leer shapefile original
    print("\n[INICIO] Leyendo capa original...")
    gdf_original = gpd.read_file(input_path)
    print(f"  Features: {len(gdf_original)}")
    print(f"  Tipo de geometría: {gdf_original.geom_type.unique()}")
    print(f"  CRS: {gdf_original.crs}")

    # Filtrar geometrías nulas
    null_geoms = gdf_original.geometry.isna().sum()
    if null_geoms > 0:
        print(f"\n  [ADVERTENCIA] Se encontraron {null_geoms} geometrías nulas. Se eliminarán.")
        gdf_original = gdf_original[~gdf_original.geometry.isna()].copy()
        print(f"  Features válidas: {len(gdf_original)}")

    # PASADA 1
    gdf_pass1, dangles_pass1, closures_pass1, stats_pass1 = single_pass(
        gdf_original, max_distance, "PASADA 1"
    )

    # PASADA 2
    gdf_pass2, dangles_pass2, closures_pass2, stats_pass2 = single_pass(
        gdf_pass1, max_distance, "PASADA 2"
    )

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar resultados
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)

    # Resultado final
    output_final = output_dir / f"{input_path.stem}_closed_2pass.gpkg"
    gdf_pass2.to_file(output_final, driver='GPKG')
    print(f"  [OK] Archivo final: {output_final}")

    # Dangles de cada pasada (para diagnóstico)
    if len(dangles_pass1) > 0:
        output_dangles1 = output_dir / f"{input_path.stem}_dangles_pass1.gpkg"
        dangles_pass1.to_file(output_dangles1, driver='GPKG')
        print(f"  [OK] Dangles pasada 1: {output_dangles1}")

    if len(dangles_pass2) > 0:
        output_dangles2 = output_dir / f"{input_path.stem}_dangles_pass2.gpkg"
        dangles_pass2.to_file(output_dangles2, driver='GPKG')
        print(f"  [OK] Dangles pasada 2: {output_dangles2}")

    # Closures de cada pasada
    if closures_pass1 is not None and len(closures_pass1) > 0:
        output_closures1 = output_dir / f"{input_path.stem}_closures_pass1.gpkg"
        closures_pass1.to_file(output_closures1, driver='GPKG')
        print(f"  [OK] Cierres pasada 1: {output_closures1}")

    if closures_pass2 is not None and len(closures_pass2) > 0:
        output_closures2 = output_dir / f"{input_path.stem}_closures_pass2.gpkg"
        closures_pass2.to_file(output_closures2, driver='GPKG')
        print(f"  [OK] Cierres pasada 2: {output_closures2}")

    # Reporte estadístico
    report = {
        'input_file': str(input_path),
        'output_file': str(output_final),
        'max_distance': max_distance,
        'original_features': len(gdf_original),
        'final_features': len(gdf_pass2),
        'pass1': stats_pass1,
        'pass2': stats_pass2,
        'total': {
            'total_closures_created': stats_pass1['closures_created'] + stats_pass2['closures_created'],
            'total_length_closed': stats_pass1['total_length_closed'] + stats_pass2['total_length_closed'],
            'initial_dangles': stats_pass1['dangles_found'],
            'final_dangles': stats_pass2['dangles_found'],
            'dangle_reduction_percent': ((stats_pass1['dangles_found'] - stats_pass2['dangles_found']) / stats_pass1['dangles_found'] * 100) if stats_pass1['dangles_found'] > 0 else 0
        }
    }

    output_report = output_dir / f"{input_path.stem}_report.json"
    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Reporte JSON: {output_report}")

    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"\nFeatures originales: {len(gdf_original)}")
    print(f"Features finales: {len(gdf_pass2)}")
    print(f"\nPASADA 1:")
    print(f"  - Dangles detectados: {stats_pass1['dangles_found']}")
    print(f"  - Líneas de cierre creadas: {stats_pass1['closures_created']}")
    print(f"  - Longitud total cerrada: {stats_pass1['total_length_closed']:.2f} m")
    print(f"\nPASADA 2:")
    print(f"  - Dangles detectados: {stats_pass2['dangles_found']}")
    print(f"  - Líneas de cierre creadas: {stats_pass2['closures_created']}")
    print(f"  - Longitud total cerrada: {stats_pass2['total_length_closed']:.2f} m")
    print(f"\nTOTAL:")
    print(f"  - Total líneas de cierre: {report['total']['total_closures_created']}")
    print(f"  - Total longitud cerrada: {report['total']['total_length_closed']:.2f} m")
    print(f"  - Reducción de dangles: {report['total']['dangle_reduction_percent']:.1f}%")
    print(f"  - Dangles restantes: {stats_pass2['dangles_found']}")
    print("\n" + "="*80)
    print("PROCESAMIENTO COMPLETADO")
    print("="*80)

    return report


def main():
    """
    Función principal - Procesa todos los municipios
    """
    # Configuración
    base_input_dir = Path("data/interim/vectores_cleaned")
    base_output_dir = Path("data/processed/ciclorutas_closed_2pass")
    max_distance = 40.0  # metros

    # Municipios a procesar
    municipios = ["madrid", "mosquera", "tabio", "tocancipa"]

    print("="*80)
    print("PROCESAMIENTO BATCH - CICLORUTAS - DOBLE PASADA")
    print("="*80)
    print(f"\nMunicipios a procesar: {len(municipios)}")
    print(f"Tolerancia: {max_distance} m\n")

    reports = {}

    for municipio in municipios:
        print("\n" + "="*80)
        print(f"PROCESANDO MUNICIPIO: {municipio.upper()}")
        print("="*80)

        input_path = base_input_dir / municipio / "Ciclorruta.gpkg"
        output_dir = base_output_dir / municipio

        if not input_path.exists():
            print(f"[ADVERTENCIA] Archivo no encontrado: {input_path}")
            continue

        try:
            report = two_pass_processing(input_path, output_dir, max_distance)
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
        total_closures = sum(r['total']['total_closures_created'] for r in reports.values())
        total_length = sum(r['total']['total_length_closed'] for r in reports.values())

        print(f"\nMunicipios procesados: {len(reports)}")
        print(f"Total líneas de cierre creadas: {total_closures}")
        print(f"Total longitud cerrada: {total_length:.2f} m ({total_length/1000:.2f} km)")

        for municipio, report in reports.items():
            print(f"\n{municipio.upper()}:")
            print(f"  - Reducción dangles: {report['total']['dangle_reduction_percent']:.1f}%")
            print(f"  - Cierres creados: {report['total']['total_closures_created']}")
            print(f"  - Longitud: {report['total']['total_length_closed']:.2f} m")
    else:
        print("\n[ADVERTENCIA] No se procesó ningún municipio.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
