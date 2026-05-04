"""
Script para extraer la capa MDANMCateg (Cabecera Municipal) de las GDB
- Lee del dataset OrdenamientoTerritorial
- Filtra por MDANMCateg == 'Cabecera Municipal'
- Guarda como GPKG en vectores_cleaned
"""
import geopandas as gpd
from pathlib import Path
import fiona
import json


def list_gdb_contents(gdb_path):
    """
    Lista el contenido de una GDB
    """
    print(f"\n[EXPLORANDO] {gdb_path.name}")

    layers = fiona.listlayers(str(gdb_path))
    print(f"  Total de capas encontradas: {len(layers)}")

    # Buscar capas relacionadas con MDANMCateg o OrdenamientoTerritorial
    relevant_layers = []
    for layer in layers:
        if 'MDANM' in layer.upper() or 'ORDENAMIENTO' in layer.upper():
            print(f"    - {layer}")
            relevant_layers.append(layer)

    return layers, relevant_layers


def extract_cabecera_municipal(gdb_path, output_path, municipio_name):
    """
    Extrae la geometría de Cabecera Municipal de la GDB

    Args:
        gdb_path: Path a la GDB
        output_path: Path de salida para el GPKG
        municipio_name: Nombre del municipio
    """
    print("=" * 80)
    print(f"PROCESANDO MUNICIPIO: {municipio_name.upper()}")
    print("=" * 80)

    # Listar capas
    layers, relevant_layers = list_gdb_contents(gdb_path)

    # Buscar la capa MDANMCateg o MDANMu
    mdanm_layer = None

    # Intentar encontrar la capa exacta
    for layer in layers:
        if 'MDANMCateg' in layer or layer == 'MDANMu':
            mdanm_layer = layer
            break

    # Si no se encuentra exactamente, buscar variantes
    if not mdanm_layer:
        for layer in layers:
            if 'MDANM' in layer.upper():
                mdanm_layer = layer
                break

    if not mdanm_layer:
        print(f"\n[ERROR] No se encontró la capa MDANMCateg en {municipio_name}")
        print(f"Capas disponibles:")
        for layer in layers:
            print(f"  - {layer}")
        return None

    print(f"\n[ENCONTRADO] Capa: {mdanm_layer}")

    # Leer la capa
    try:
        print(f"\n[LEYENDO] Capa {mdanm_layer}...")
        gdf = gpd.read_file(gdb_path, layer=mdanm_layer)
        print(f"  Features totales: {len(gdf)}")
        print(f"  Columnas: {list(gdf.columns)}")

        # Buscar el campo MDANMCateg
        categ_field = None
        for col in gdf.columns:
            if 'MDANM' in col.upper() and 'CATEG' in col.upper():
                categ_field = col
                break

        if not categ_field:
            print(f"\n[ERROR] No se encontró el campo MDANMCateg")
            print(f"Campos disponibles: {list(gdf.columns)}")
            return None

        print(f"\n[CAMPO ENCONTRADO] {categ_field}")

        # Mostrar valores únicos
        print(f"\n[VALORES ÚNICOS] en {categ_field}:")
        unique_values = gdf[categ_field].unique()
        for val in unique_values:
            count = (gdf[categ_field] == val).sum()
            print(f"  - {val}: {count} features")

        # Filtrar por Cabecera Municipal
        # El campo MDANMCateg usa códigos de dominio (numéricos)
        # Código 4 = Cabecera Municipal
        # Código 3 = Centro Poblado

        # Intentar filtrar por código 4 (Cabecera Municipal)
        cabecera_gdf = gdf[gdf[categ_field] == 4]

        if len(cabecera_gdf) == 0:
            print(f"\n[ADVERTENCIA] No se encontraron features con código 4 (Cabecera Municipal)")
            print(f"Intentando con código 3 (Centro Poblado)...")
            cabecera_gdf = gdf[gdf[categ_field] == 3]

        if len(cabecera_gdf) == 0:
            print(f"\n[ERROR] No se encontraron features con código 3 o 4")
            return None

        print(f"\n[FILTRADO] Features de Cabecera Municipal: {len(cabecera_gdf)}")

        # Guardar
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cabecera_gdf.to_file(output_path, driver='GPKG')
        print(f"\n[GUARDADO] {output_path}")

        # Reporte
        report = {
            'municipio': municipio_name,
            'gdb_path': str(gdb_path),
            'layer_name': mdanm_layer,
            'field_name': categ_field,
            'total_features': len(gdf),
            'cabecera_features': len(cabecera_gdf),
            'output_path': str(output_path),
            'crs': str(cabecera_gdf.crs)
        }

        if len(cabecera_gdf) > 0:
            report['stats'] = {
                'total_area_m2': float(cabecera_gdf.geometry.area.sum()),
                'geometry_type': cabecera_gdf.geom_type.unique().tolist()
            }

        return report

    except Exception as e:
        print(f"\n[ERROR] Error leyendo la capa: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Función principal - Procesa todos los municipios
    """
    base_raw_dir = Path("data/raw/vectores")
    base_output_dir = Path("data/interim/vectores_cleaned")

    municipios = ["madrid", "mosquera", "tabio", "tocancipa"]

    print("="*80)
    print("EXTRACCIÓN DE CABECERA MUNICIPAL - MDANMCATEG")
    print("="*80)
    print(f"\nMunicipios a procesar: {len(municipios)}\n")

    reports = {}

    for municipio in municipios:
        municipio_raw_dir = base_raw_dir / municipio

        # Buscar la GDB en el directorio del municipio
        gdb_files = list(municipio_raw_dir.glob("*.gdb"))

        if len(gdb_files) == 0:
            print(f"\n[ERROR] No se encontró GDB para {municipio}")
            continue

        gdb_path = gdb_files[0]
        output_path = base_output_dir / municipio / "Cabecera_Municipal.gpkg"

        try:
            report = extract_cabecera_municipal(gdb_path, output_path, municipio)
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
        print(f"\nMunicipios procesados exitosamente: {len(reports)}")

        for municipio, report in reports.items():
            print(f"\n{municipio.upper()}:")
            print(f"  - Features de Cabecera Municipal: {report['cabecera_features']}")
            print(f"  - Output: {report['output_path']}")
            if 'stats' in report:
                print(f"  - Área total: {report['stats']['total_area_m2']:.2f} m²")

        # Guardar reporte JSON
        report_path = Path("data/interim/cabecera_municipal_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORTE] Guardado en: {report_path}")
    else:
        print("\n[ADVERTENCIA] No se procesó ningún municipio.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
