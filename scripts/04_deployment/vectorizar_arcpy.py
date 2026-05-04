"""
vectorizar_arcpy.py
===================
Pipeline de vectorizacion de clases poligonales usando ArcPy.

Convierte el raster clasificado (salida del DLPK en ArcGIS Pro) en capas
vectoriales limpias para Construccion y Bosque, alineadas con los estandares
de produccion cartografica del IGAC (Resolucion 471/2020).

Flujo
-----
  1. PP liviano sobre mosaico completo (MajorityFilter + BoundaryClean)
  2. Extraccion por clase (Con)
  3. RasterToPolygon  (genera un poligono por region conexa — NO Dissolve)
  4. SimplifyPolygon POINT_REMOVE  (equivalente a Ramer-Douglas-Peucker)
  5. EliminatePolygonPart  (elimina huecos internos y esquirlas)
  6. CheckGeometry + RepairGeometry  (validacion OGC)
  7. Exportar a GeoPackage (.gpkg)
  8. Guardar estadisticas en JSON

Uso (desde entorno arcgispro-py3):
    python scripts/vectorizar_arcpy.py
        --input   D:\\...\\clasificado.tif
        --outdir  D:\\...\\results\\vectores
        [--municipio  tocancipa]
        [--tolerance  0.5]

Uso desde notebook igac1k (via subprocess):
    Ver notebooks/12_vectorizacion_arcpy.ipynb
"""

import arcpy
from arcpy.sa import MajorityFilter, BoundaryClean, Con, SetNull
import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracion de clases
# ---------------------------------------------------------------------------
CLASES_VECTORIZAR = {
    1: {
        "nombre":         "Construccion",
        "min_area_m2":    50,    # poligonos < 50 m2 se eliminan (ruido)
        "min_hole_m2":    10,    # huecos internos < 10 m2 se rellenan
        "simplify_tol":   0.5,   # tolerancia RDP en metros
    },
    2: {
        "nombre":         "Bosque",
        "min_area_m2":    100,
        "min_hole_m2":    25,
        "simplify_tol":   1.0,
    },
}

NOMBRES_CLASES = {
    0: "Background", 1: "Construccion", 2: "Bosque",
    3: "DepositoAgua", 4: "ZonaVerde", 5: "ZonaDura",
    6: "Via", 7: "Cerca", 8: "CabeceraMunicipal",
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Paso 1 — Postprocesamiento raster liviano (nivel mosaico)
# ---------------------------------------------------------------------------
def pp_raster(input_raster: str, workspace: str) -> str:
    """
    Aplica MajorityFilter + BoundaryClean sobre el mosaico completo.
    Esto NO se hace por parche — opera sobre toda la imagen de una vez,
    lo que permite conectar fragmentos que cruzan bordes de tile.
    """
    log("PP raster (MajorityFilter + BoundaryClean)...")

    r = MajorityFilter(input_raster, number_neighbors="EIGHT",
                       majority_definition="MAJORITY")
    r = BoundaryClean(r, sort_type="DESCEND", number_of_runs="TWO_WAY")

    out_path = os.path.join(workspace, "raster_pp.tif")
    r.save(out_path)
    log(f"  Raster PP guardado: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Paso 2-8 — Vectorizacion por clase
# ---------------------------------------------------------------------------
def vectorizar_clase(raster_pp: str, cls_id: int, cfg: dict,
                     workspace: str, outdir: str, municipio: str) -> dict:
    """
    Pipeline completo de vectorizacion para una clase.
    Retorna dict con estadisticas.
    """
    nombre    = cfg["nombre"]
    tol       = cfg["simplify_tol"]
    min_area  = cfg["min_area_m2"]
    min_hole  = cfg["min_hole_m2"]
    prefix    = os.path.join(workspace, f"cls{cls_id}")

    log(f"--- Clase {cls_id}: {nombre} ---")

    # ── 2. Extraer clase como raster binario ─────────────────────────────────
    log("  [2] Extraccion de clase...")
    cls_raster = Con(arcpy.Raster(raster_pp) == cls_id, cls_id)
    cls_raster_path = f"{prefix}_raster.tif"
    cls_raster.save(cls_raster_path)

    # ── 3. RasterToPolygon ───────────────────────────────────────────────────
    log("  [3] RasterToPolygon...")
    raw_poly = f"{prefix}_raw.shp"
    arcpy.conversion.RasterToPolygon(
        cls_raster_path, raw_poly,
        simplify="NO_SIMPLIFY",
        raster_field="Value"
    )
    n_raw = int(arcpy.management.GetCount(raw_poly).getOutput(0))
    log(f"       Poligonos raw: {n_raw:,}")

    # ── 4. SimplifyPolygon (RDP) ─────────────────────────────────────────────
    # NOTA: Se omite Dissolve intencionalmente. RasterToPolygon ya genera un
    # poligono por cada region conexa de la misma clase. Aplicar Dissolve con
    # campo "gridcode" fusionaria TODOS los poligonos de la misma clase en un
    # unico MultiPolygon, perdiendo las entidades individuales.
    log(f"  [4] SimplifyPolygon (POINT_REMOVE, tol={tol}m)...")
    simplified = f"{prefix}_simplified.shp"
    arcpy.cartography.SimplifyPolygon(
        raw_poly, simplified,
        algorithm="POINT_REMOVE",
        tolerance=f"{tol} Meters",
        minimum_area=f"{min_area} SquareMeters",
        collapsed_point_option="NO_KEEP"
    )
    n_simplified = int(arcpy.management.GetCount(simplified).getOutput(0))
    log(f"       Poligonos tras simplify: {n_simplified:,}")

    # ── 5. EliminatePolygonPart (huecos internos y esquirlas) ────────────────
    log(f"  [5] EliminatePolygonPart (huecos < {min_hole} m2)...")
    cleaned = f"{prefix}_cleaned.shp"
    arcpy.management.EliminatePolygonPart(
        simplified, cleaned,
        condition="AREA",
        part_area=f"{min_hole} SquareMeters",
        part_area_percent=0,
        part_option="CONTAINED_ONLY"
    )
    n_cleaned = int(arcpy.management.GetCount(cleaned).getOutput(0))
    log(f"       Poligonos tras eliminate: {n_cleaned:,}")

    # ── 6. CheckGeometry + RepairGeometry (validacion OGC) ───────────────────
    log("  [6] CheckGeometry + RepairGeometry...")
    errores_tabla = f"{prefix}_geom_errors.dbf"
    arcpy.management.CheckGeometry(cleaned, errores_tabla)
    n_errores = int(arcpy.management.GetCount(errores_tabla).getOutput(0))

    if n_errores > 0:
        log(f"       {n_errores} errores geometricos — aplicando RepairGeometry...")
        arcpy.management.RepairGeometry(cleaned)
    else:
        log("       Geometria valida (0 errores OGC)")

    # ── 7. Exportar a GeoPackage ─────────────────────────────────────────────
    gpkg_name = f"{nombre.lower()}_{municipio}.gpkg"
    gpkg_path = os.path.join(outdir, gpkg_name)
    log(f"  [7] Exportando a GeoPackage: {gpkg_path}")

    if arcpy.Exists(gpkg_path):
        arcpy.management.Delete(gpkg_path)

    arcpy.management.CreateSQLiteDatabase(gpkg_path, "GEOPACKAGE")
    arcpy.conversion.FeatureClassToGeodatabase(cleaned, gpkg_path)
    log(f"       GeoPackage creado.")

    # ── Estadisticas ─────────────────────────────────────────────────────────
    areas = []
    with arcpy.da.SearchCursor(cleaned, ["SHAPE@AREA"]) as cursor:
        for row in cursor:
            if row[0] is not None:
                areas.append(row[0])

    stats = {
        "clase":              nombre,
        "clase_id":           cls_id,
        "municipio":          municipio,
        "n_poligonos_raw":    n_raw,
        "n_poligonos_final":  n_cleaned,
        "errores_ogc":        n_errores,
        "area_total_m2":      round(sum(areas), 2),
        "area_media_m2":      round(sum(areas) / len(areas), 2) if areas else 0,
        "area_min_m2":        round(min(areas), 2) if areas else 0,
        "area_max_m2":        round(max(areas), 2) if areas else 0,
        "simplify_tol_m":     tol,
        "min_area_m2":        min_area,
        "min_hole_m2":        min_hole,
        "gpkg_path":          gpkg_path,
    }

    log(f"  Estadisticas: {n_cleaned} poligonos | area total={stats['area_total_m2']:,.1f} m2 | media={stats['area_media_m2']:.1f} m2")
    return stats


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def vectorizar(input_raster: str, outdir: str, municipio: str,
               tolerance: float = None, skip_pp: bool = False) -> dict:
    """
    Pipeline completo: PP raster → vectorizacion Construccion + Bosque.
    """
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster      = input_raster
    arcpy.env.cellSize        = input_raster

    # Workspace temporal
    workspace = os.path.join(outdir, "_tmp_vectorizacion")
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(outdir,    exist_ok=True)
    arcpy.env.workspace = workspace

    log(f"Input raster: {input_raster}")
    log(f"Output dir:   {outdir}")
    log(f"Municipio:    {municipio}")

    # Paso 1 — PP raster
    if skip_pp:
        log("PP raster omitido (--skip_pp)")
        raster_pp = input_raster
    else:
        raster_pp = pp_raster(input_raster, workspace)

    # Pasos 2-8 por clase
    resultados = {}
    for cls_id, cfg in CLASES_VECTORIZAR.items():
        if tolerance is not None:
            cfg = dict(cfg, simplify_tol=tolerance)
        stats = vectorizar_clase(raster_pp, cls_id, cfg, workspace, outdir, municipio)
        resultados[cfg["nombre"]] = stats

    # Guardar JSON de estadisticas
    json_path = os.path.join(outdir, f"estadisticas_vectores_{municipio}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    log(f"Estadisticas guardadas: {json_path}")

    # Limpiar workspace temporal
    try:
        import shutil
        shutil.rmtree(workspace)
        log("Workspace temporal eliminado.")
    except Exception as e:
        log(f"No se pudo limpiar workspace temporal: {e}")

    arcpy.CheckInExtension("Spatial")
    log("=== VECTORIZACION COMPLETADA ===")
    return resultados


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vectorizacion ArcPy: Construccion + Bosque desde raster clasificado"
    )
    parser.add_argument("--input",      required=True,
                        help="Raster clasificado de entrada (.tif)")
    parser.add_argument("--outdir",     required=True,
                        help="Carpeta de salida para GeoPackages y estadisticas")
    parser.add_argument("--municipio",  default="municipio",
                        help="Nombre del municipio (para nombrar archivos de salida)")
    parser.add_argument("--tolerance",  type=float, default=None,
                        help="Tolerancia RDP en metros (default: por clase en config)")
    parser.add_argument("--skip_pp",    action="store_true",
                        help="Omitir PP raster (usar si ya viene postprocesado)")
    args = parser.parse_args()

    vectorizar(
        input_raster=args.input,
        outdir=args.outdir,
        municipio=args.municipio,
        tolerance=args.tolerance,
        skip_pp=args.skip_pp,
    )
