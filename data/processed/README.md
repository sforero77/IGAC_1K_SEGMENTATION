# Processed Data (Datos Procesados Finales)

Datos finales listos para entrenamiento de modelos.

## 📂 Contenido:

### `masks/`
Máscaras de segmentación rasterizadas por ciudad (Día 5):

**Estructura:**
```
masks/
├── madrid/
├── mosquera/
├── tabio/
└── tocancipa/
```

**Especificaciones:**
- Formato: GeoTIFF uint8
- Resolución: 10 cm/píxel (igual que ortofotos)
- Sistema: MAGNA-SIRGAS origen Bogotá
- Codificación de clases: 0-9 (0=Background, 1-9=clases)

**Esquema de clases:**
- 0: Background (sin etiqueta)
- 1: Zona_Dura
- 2: Construcción_R
- 3: Cerca
- 4: Vía
- 5: Límite_Vía
- 6: Cicloruta
- 7: Deposito_Agua_R
- 8: Zona_Verde
- 9: Bosque

### `patches/`
Tiles de 512×512 píxeles divididos en train/val/test (Día 6):

**Estructura:**
```
patches/
├── train/
│   ├── images/    # Patches de ortofotos (X)
│   └── masks/     # Patches de máscaras (Y)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Parámetros de tiling:**
- Tamaño de patch: 512×512 píxeles (51.2×51.2 m)
- Overlap: 25% (stride=384)
- Formato: GeoTIFF con georreferenciación
- Filtrado: Patches con >95% background descartados en train

**Metadatos por patch:**
- Ciudad origen
- Tile origen
- Coordenadas geográficas (bbox)
- Distribución de clases (% píxeles)
- Ver: `patches_index.csv`

### `spatial_splits/`
Polígonos que definen zonas de train/val/test (Día 3):

**Archivos:**
- `train_zones.shp` - Zonas de entrenamiento (60%)
- `val_zones.shp` - Zonas de validación (20%)
- `test_zones.shp` - Zonas de prueba (20%)

⚠️ **CRÍTICO**: Partición es ESPACIAL, NO aleatoria por píxeles
- Zonas geográficamente separadas
- Sin overlap entre splits
- Diseño pensado para evaluar generalización

## ⚠️ Importante:

1. **Integridad**: Verificar alineación perfecta ortofoto-máscara (offset=0)
2. **Balance**: Usar `class_weights.json` durante entrenamiento
3. **Validación**: Ver notebooks `02_preprocessing_validation.ipynb`

## 🔗 Scripts relacionados:

- `scripts/03_create_spatial_splits.py` → Genera `spatial_splits/`
- `scripts/05_rasterize_vectors.py` → Genera `masks/`
- `scripts/06_create_patches.py` → Genera `patches/`
- `scripts/07_analyze_balance.py` → Analiza distribución
