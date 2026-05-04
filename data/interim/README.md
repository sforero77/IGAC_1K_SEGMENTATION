# Interim Data (Datos Intermedios)

Datos en proceso de transformación durante el preprocesamiento.

## 📂 Contenido:

### `ortofotos_normalizadas/`
Ortofotografías después de normalización radiométrica (Día 4):
- Valores convertidos a rango [0, 1]
- Normalización z-score aplicada: (pixel - μ) / σ
- Estadísticas calculadas SOLO sobre conjunto de entrenamiento
- Formato: GeoTIFF float32

### `vectores_por_clase/`
Vectores IGAC separados por clase cartográfica (Día 5):
- Separación de cada Feature Class individual
- Preparación para rasterización
- Formato: Shapefiles o GeoPackage

### `estadisticas/`
Archivos JSON/CSV con estadísticas y metadatos críticos:

**Archivos clave:**
- `normalization_stats.json` - μ y σ por banda para normalización
- `class_distribution.csv` - Distribución de píxeles por clase
- `class_weights.json` - Pesos calculados para loss function
- `spatial_splits_info.json` - Información de partición train/val/test
- `tile_metadata.json` - Metadatos por tile/patch

## ⚠️ Importante:

- **Reproducibilidad**: Los archivos en `estadisticas/` son CRÍTICOS para reproducir el pipeline
- **Versionar**: Estos JSON/CSV SÍ deben estar en Git (son pequeños)
- **Documentar**: Cualquier cambio en estadísticas debe documentarse

## 🔗 Scripts relacionados:

- `scripts/04_preprocess_orthos.py` → Genera `ortofotos_normalizadas/`
- `scripts/05_rasterize_vectors.py` → Usa `vectores_por_clase/`
- `scripts/07_analyze_balance.py` → Genera `class_weights.json`
