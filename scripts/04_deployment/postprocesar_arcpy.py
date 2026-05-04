"""
Postprocesamiento de raster clasificado usando ArcPy + Spatial Analyst.
Aplica filtros morfologicos por clase sobre el resultado de inferencia del DLPK.

Operaciones por clase:
  - Majority Filter    : elimina pixeles aislados / ruido sal-pimienta
  - Boundary Clean     : suaviza bordes entre clases
  - Sieve              : elimina regiones pequenas aisladas (min_area en pixeles)
  - Expand + Shrink    : closing morfologico (rellena huecos internos)

Uso desde notebook (entorno igac1k):
    import subprocess
    subprocess.run([ARCGIS_PYTHON, 'scripts/postprocesar_arcpy.py',
                    '--input', 'ruta/clasificado.tif',
                    '--output', 'ruta/clasificado_pp.tif'])

Uso directo (entorno arcgispro-py3):
    python scripts/postprocesar_arcpy.py --input clasificado.tif --output clasificado_pp.tif
"""

import arcpy
from arcpy.sa import (
    MajorityFilter, BoundaryClean,
    Expand, Shrink, Con, IsNull,
    RegionGroup, Nibble
)
import argparse
import sys
import os
import time

# ── Clases del modelo ──────────────────────────────────────────────────────────
CLASES = {
    0: "Background",
    1: "Construccion",
    2: "Bosque",
    3: "DepositoAgua",
    4: "ZonaVerde",
    5: "ZonaDura",
    6: "Via",
    7: "Cerca",
    8: "CabeceraMunicipal",
}

# ── Parametros de PP por clase ─────────────────────────────────────────────────
# min_area: pixeles minimos para conservar una region (0 = no aplicar Sieve)
# closing:  aplicar Expand+Shrink (rellena huecos internos)
# majority: aplicar MajorityFilter
PP_CONFIG = {
    1: {"name": "Construccion",      "min_area": 200,   "closing": True,  "majority": True},
    2: {"name": "Bosque",            "min_area": 500,   "closing": True,  "majority": True},
    3: {"name": "DepositoAgua",      "min_area": 100,   "closing": True,  "majority": True},
    4: {"name": "ZonaVerde",         "min_area": 100,   "closing": False, "majority": True},
    5: {"name": "ZonaDura",          "min_area": 100,   "closing": False, "majority": True},
    6: {"name": "Via",               "min_area":  50,   "closing": False, "majority": True},
    7: {"name": "Cerca",             "min_area":   0,   "closing": False, "majority": False},
    8: {"name": "CabeceraMunicipal", "min_area": 5000,  "closing": True,  "majority": True},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def postprocesar(input_raster: str, output_raster: str, workspace: str = None):
    """
    Pipeline completo de postprocesamiento.

    Args:
        input_raster : ruta al raster clasificado (salida del DLPK)
        output_raster: ruta de salida del raster postprocesado
        workspace    : carpeta temporal para archivos intermedios
    """
    arcpy.CheckOutExtension("Spatial")

    # Workspace temporal
    if workspace is None:
        workspace = os.path.join(os.path.dirname(output_raster), "pp_temp")
    os.makedirs(workspace, exist_ok=True)
    arcpy.env.workspace      = workspace
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster     = input_raster
    arcpy.env.cellSize       = input_raster

    log(f"Input:     {input_raster}")
    log(f"Output:    {output_raster}")
    log(f"Workspace: {workspace}")

    # ── Paso 1: Majority Filter global (ruido sal-pimienta) ───────────────────
    log("Paso 1/4: Majority Filter global...")
    raster = MajorityFilter(input_raster, number_neighbors="EIGHT", majority_definition="MAJORITY")
    raster.save(os.path.join(workspace, "paso1_majority.tif"))
    raster = arcpy.Raster(os.path.join(workspace, "paso1_majority.tif"))

    # ── Paso 2: Boundary Clean global (suaviza bordes entre clases) ───────────
    log("Paso 2/4: Boundary Clean global...")
    raster = BoundaryClean(raster, sort_type="DESCEND", number_of_runs="TWO_WAY")
    raster.save(os.path.join(workspace, "paso2_boundary.tif"))
    raster = arcpy.Raster(os.path.join(workspace, "paso2_boundary.tif"))

    # ── Paso 3: PP por clase (Sieve + Expand/Shrink) ──────────────────────────
    log("Paso 3/4: PP por clase...")
    resultado = raster

    for cls_id, cfg in PP_CONFIG.items():
        nombre   = cfg["name"]
        min_area = cfg["min_area"]
        closing  = cfg["closing"]

        pasos = []
        if min_area > 0:       pasos.append(f"sieve({min_area}px)")
        if closing:            pasos.append("closing")
        if not pasos:
            continue

        log(f"  Clase {cls_id} {nombre}: {', '.join(pasos)}")

        # Mascara binaria de la clase
        mascara_cls = Con(resultado == cls_id, 1, 0)

        # Sieve: eliminar regiones menores a min_area px
        if min_area > 0:
            grupos = RegionGroup(mascara_cls, "FOUR", "WITHIN", "NO_LINK")
            mascara_cls = Con(
                (mascara_cls == 1) & (arcpy.sa.Lookup(grupos, "COUNT") >= min_area),
                1, 0
            )

        # Closing morfologico (Expand radio=2 + Shrink radio=2)
        if closing:
            mascara_cls = Expand(mascara_cls, 2, [1])
            mascara_cls = Shrink(mascara_cls, 2, [1])

        # Aplicar mascara al resultado: donde la clase fue modificada, actualizar
        resultado = Con(mascara_cls == 1, cls_id, resultado)

    paso3_path = os.path.join(workspace, "paso3_clases.tif")
    resultado.save(paso3_path)
    resultado = arcpy.Raster(paso3_path)

    # ── Paso 4: Nibble final (rellenar huecos con vecinos) ────────────────────
    log("Paso 4/4: Nibble final (rellenar NoData/huecos)...")
    # Solo aplicar si hay valores nulos
    null_count = arcpy.management.GetRasterProperties(resultado, "ANYNODATA").getOutput(0)
    if null_count == "1":
        mask_nodata = Con(IsNull(resultado), 1)
        resultado   = Nibble(resultado, mask_nodata, "ALL_VALUES")

    # ── Guardar resultado final ────────────────────────────────────────────────
    resultado.save(output_raster)
    log(f"Guardado: {output_raster}")

    # Limpiar workspace temporal
    try:
        import shutil
        shutil.rmtree(workspace)
    except Exception:
        pass

    arcpy.CheckInExtension("Spatial")
    log("Postprocesamiento completado.")
    return output_raster


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocesamiento ArcPy de raster clasificado")
    parser.add_argument("--input",     required=True,  help="Raster clasificado de entrada (.tif)")
    parser.add_argument("--output",    required=True,  help="Raster postprocesado de salida (.tif)")
    parser.add_argument("--workspace", default=None,   help="Carpeta temporal (default: junto al output)")
    args = parser.parse_args()

    postprocesar(args.input, args.output, args.workspace)
