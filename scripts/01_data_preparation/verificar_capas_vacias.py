"""
Script para verificar qué capas tienen features y cuáles están vacías
"""

import geopandas as gpd
import fiona
import os
from pathlib import Path

base_dir = r"D:\IGAC_1K_SEGMENTATION\data\processed\vectores_consolidados"

municipios = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

print("="*80)
print(" REPORTE DE CAPAS VACIAS")
print("="*80)

for municipio in municipios:
    municipio_path = os.path.join(base_dir, municipio)
    gdb_files = list(Path(municipio_path).glob("*.gdb"))

    if not gdb_files:
        print(f"\n{municipio.upper()}: NO SE ENCONTRO GDB")
        continue

    gdb_path = str(gdb_files[0])
    layers = fiona.listlayers(gdb_path)

    print(f"\n{municipio.upper()}:")
    print(f"  GDB: {os.path.basename(gdb_path)}")
    print(f"  Capas: {len(layers)}")

    capas_con_datos = []
    capas_vacias = []

    for layer in layers:
        gdf = gpd.read_file(gdb_path, layer=layer)
        count = len(gdf)

        if count > 0:
            capas_con_datos.append((layer, count))
        else:
            capas_vacias.append(layer)

    if capas_con_datos:
        print(f"\n  CAPAS CON DATOS:")
        for capa, count in capas_con_datos:
            print(f"    - {capa:30s}: {count:6,} features")

    if capas_vacias:
        print(f"\n  CAPAS VACIAS (0 features):")
        for capa in capas_vacias:
            print(f"    - {capa}")

print(f"\n{'='*80}")
