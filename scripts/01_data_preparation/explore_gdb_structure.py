"""
Script para explorar la estructura de los GDB y identificar las capas disponibles
"""
import os
from pathlib import Path
try:
    from osgeo import ogr
    ogr.UseExceptions()
    USE_OGR = True
except ImportError:
    USE_OGR = False
    print("GDAL/OGR no disponible, intentando con fiona...")
    try:
        import fiona
        USE_FIONA = True
    except ImportError:
        USE_FIONA = False
        print("Fiona tampoco disponible")

def explore_gdb_with_ogr(gdb_path):
    """Explorar GDB usando OGR (GDAL)"""
    print(f"\n{'='*80}")
    print(f"Explorando: {gdb_path}")
    print(f"{'='*80}")

    driver = ogr.GetDriverByName("OpenFileGDB")
    if driver is None:
        print("Driver OpenFileGDB no disponible, intentando FileGDB...")
        driver = ogr.GetDriverByName("FileGDB")

    if driver is None:
        print("No hay drivers disponibles para GDB")
        return

    ds = driver.Open(gdb_path, 0)
    if ds is None:
        print(f"No se pudo abrir el GDB: {gdb_path}")
        return

    print(f"\nNúmero de capas: {ds.GetLayerCount()}")
    print("\nCapas disponibles:")
    print("-" * 80)

    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayerByIndex(i)
        layer_name = layer.GetName()
        feature_count = layer.GetFeatureCount()
        geom_type = layer.GetGeomType()

        print(f"{i+1}. {layer_name}")
        print(f"   - Geometría: {ogr.GeometryTypeToName(geom_type)}")
        print(f"   - Features: {feature_count}")

    ds = None

def explore_gdb_with_fiona(gdb_path):
    """Explorar GDB usando Fiona"""
    print(f"\n{'='*80}")
    print(f"Explorando: {gdb_path}")
    print(f"{'='*80}")

    layers = fiona.listlayers(gdb_path)
    print(f"\nNúmero de capas: {len(layers)}")
    print("\nCapas disponibles:")
    print("-" * 80)

    for i, layer_name in enumerate(layers):
        try:
            with fiona.open(gdb_path, layer=layer_name) as src:
                print(f"{i+1}. {layer_name}")
                print(f"   - Geometría: {src.schema['geometry']}")
                print(f"   - Features: {len(src)}")
        except Exception as e:
            print(f"{i+1}. {layer_name} - Error: {e}")

def main():
    # Directorio base de vectores
    vectores_dir = Path("data/raw/vectores")

    # Capas objetivo
    target_layers = [
        "Zona_Dura",
        "Cerca",
        "Construccion_R",
        "Via",
        "Limite_Via",
        "Ciclorruta",
        "Deposito_Agua_R",
        "Zona_Verde",
        "Bosque"
    ]

    print("Capas objetivo a extraer:")
    for i, layer in enumerate(target_layers, 1):
        print(f"{i}. {layer}")

    # Explorar cada municipio
    municipios = [d for d in vectores_dir.iterdir() if d.is_dir()]

    for municipio_dir in municipios:
        municipio_name = municipio_dir.name

        # Buscar GDB en el directorio del municipio
        gdb_files = list(municipio_dir.glob("*.gdb"))

        if not gdb_files:
            print(f"\nNo se encontró GDB en {municipio_name}")
            continue

        for gdb_path in gdb_files:
            if USE_OGR:
                explore_gdb_with_ogr(str(gdb_path))
            elif USE_FIONA:
                explore_gdb_with_fiona(str(gdb_path))
            else:
                print(f"\nNo hay herramientas disponibles para leer el GDB: {gdb_path}")

    print("\n" + "="*80)
    print("Exploración completada")
    print("="*80)

if __name__ == "__main__":
    main()
