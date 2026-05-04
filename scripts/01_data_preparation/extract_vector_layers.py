"""
Script para extraer capas específicas de los GDB por municipio
Extrae las 9 capas objetivo:
1. Zona_Dura (ZDura)
2. Cerca
3. Construccion_R (Constr_R)
4. Via
5. Limite_Via (LVia)
6. Ciclorruta (Ciclor)
7. Deposito_Agua_R (DAgua_R)
8. Zona_Verde (ZVerde)
9. Bosque
"""
import os
from pathlib import Path
import fiona
import geopandas as gpd
from tqdm import tqdm
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapeo de nombres de capas objetivo a nombres en el GDB
LAYER_MAPPING = {
    'Zona_Dura': 'ZDura',
    'Cerca': 'Cerca',
    'Construccion_R': 'Constr_R',
    'Via': 'Via',
    'Limite_Via': 'LVia',
    'Ciclorruta': 'Ciclor',
    'Deposito_Agua_R': 'DAgua_R',
    'Zona_Verde': 'ZVerde',
    'Bosque': 'Bosque'
}

def extract_layer_from_gdb(gdb_path, layer_name_in_gdb, output_path, municipio):
    """
    Extrae una capa específica del GDB y la guarda como GeoPackage

    Args:
        gdb_path: Ruta al archivo GDB
        layer_name_in_gdb: Nombre de la capa en el GDB
        output_path: Ruta de salida para el archivo
        municipio: Nombre del municipio

    Returns:
        bool: True si la extracción fue exitosa, False en caso contrario
    """
    try:
        # Leer la capa del GDB
        logger.info(f"  Leyendo capa {layer_name_in_gdb} de {municipio}...")
        gdf = gpd.read_file(gdb_path, layer=layer_name_in_gdb)

        # Verificar si la capa está vacía
        if len(gdf) == 0:
            logger.warning(f"  ⚠ Capa {layer_name_in_gdb} está vacía en {municipio}")
            return False

        # Agregar columna de municipio
        gdf['municipio'] = municipio

        # Guardar como GeoPackage
        logger.info(f"  Guardando {len(gdf)} features en {output_path}...")
        gdf.to_file(output_path, driver='GPKG')

        logger.info(f"  ✓ Capa {layer_name_in_gdb} extraída exitosamente ({len(gdf)} features)")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error al extraer {layer_name_in_gdb} de {municipio}: {str(e)}")
        return False

def main():
    """
    Función principal para extraer todas las capas de todos los municipios
    """
    # Directorios
    vectores_dir = Path("data/raw/vectores")
    output_dir = Path("data/interim/vectores_cleaned")

    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener lista de municipios
    municipios = [d for d in vectores_dir.iterdir() if d.is_dir()]

    logger.info("="*80)
    logger.info("INICIO DE EXTRACCIÓN DE CAPAS VECTORIALES")
    logger.info("="*80)
    logger.info(f"Municipios a procesar: {[m.name for m in municipios]}")
    logger.info(f"Capas objetivo: {list(LAYER_MAPPING.keys())}")
    logger.info("="*80)

    # Estadísticas
    total_layers = len(LAYER_MAPPING) * len(municipios)
    successful_extractions = 0
    failed_extractions = 0
    empty_layers = 0

    # Procesar cada municipio
    for municipio_dir in municipios:
        municipio_name = municipio_dir.name
        logger.info(f"\n{'='*80}")
        logger.info(f"Procesando municipio: {municipio_name.upper()}")
        logger.info(f"{'='*80}")

        # Buscar GDB en el directorio del municipio
        gdb_files = list(municipio_dir.glob("*.gdb"))

        if not gdb_files:
            logger.warning(f"⚠ No se encontró GDB en {municipio_name}")
            failed_extractions += len(LAYER_MAPPING)
            continue

        gdb_path = gdb_files[0]
        logger.info(f"GDB encontrado: {gdb_path.name}")

        # Crear directorio de salida para el municipio
        municipio_output_dir = output_dir / municipio_name
        municipio_output_dir.mkdir(parents=True, exist_ok=True)

        # Extraer cada capa
        for target_layer_name, gdb_layer_name in LAYER_MAPPING.items():
            logger.info(f"\nExtrayendo: {target_layer_name} ({gdb_layer_name})")

            # Definir ruta de salida
            output_file = municipio_output_dir / f"{target_layer_name}.gpkg"

            # Verificar si la capa existe en el GDB
            try:
                available_layers = fiona.listlayers(str(gdb_path))
                if gdb_layer_name not in available_layers:
                    logger.warning(f"  ⚠ Capa {gdb_layer_name} no encontrada en el GDB")
                    failed_extractions += 1
                    continue
            except Exception as e:
                logger.error(f"  ✗ Error al listar capas del GDB: {str(e)}")
                failed_extractions += 1
                continue

            # Extraer la capa
            success = extract_layer_from_gdb(
                str(gdb_path),
                gdb_layer_name,
                str(output_file),
                municipio_name
            )

            if success:
                successful_extractions += 1
            else:
                failed_extractions += 1

    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE EXTRACCIÓN")
    logger.info("="*80)
    logger.info(f"Total de capas a procesar: {total_layers}")
    logger.info(f"Extracciones exitosas: {successful_extractions}")
    logger.info(f"Extracciones fallidas: {failed_extractions}")
    logger.info(f"Directorio de salida: {output_dir.absolute()}")
    logger.info("="*80)

    # Generar reporte detallado
    generate_extraction_report(output_dir, municipios)

def generate_extraction_report(output_dir, municipios):
    """
    Genera un reporte detallado de las capas extraídas
    """
    report_path = output_dir / "extraction_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE EXTRACCIÓN DE CAPAS VECTORIALES\n")
        f.write("="*80 + "\n\n")

        for municipio_dir in municipios:
            municipio_name = municipio_dir.name
            municipio_output_dir = output_dir / municipio_name

            f.write(f"\nMunicipio: {municipio_name.upper()}\n")
            f.write("-"*80 + "\n")

            if not municipio_output_dir.exists():
                f.write("  No se procesó este municipio\n")
                continue

            gpkg_files = list(municipio_output_dir.glob("*.gpkg"))

            if not gpkg_files:
                f.write("  No se extrajeron capas\n")
                continue

            for gpkg_file in sorted(gpkg_files):
                try:
                    gdf = gpd.read_file(gpkg_file)
                    layer_name = gpkg_file.stem
                    feature_count = len(gdf)
                    geom_type = gdf.geometry.type.unique()

                    f.write(f"  ✓ {layer_name}:\n")
                    f.write(f"     - Features: {feature_count}\n")
                    f.write(f"     - Geometría: {', '.join(geom_type)}\n")
                    f.write(f"     - Archivo: {gpkg_file.name}\n")

                except Exception as e:
                    f.write(f"  ✗ {gpkg_file.stem}: Error al leer - {str(e)}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")

    logger.info(f"\nReporte generado: {report_path.absolute()}")

if __name__ == "__main__":
    main()
