# Raw Data (Datos Originales)

⚠️ **NUNCA MODIFICAR ESTOS ARCHIVOS** ⚠️

Esta carpeta contiene los datos originales inmutables del proyecto.

## 📂 Contenido:

### `ortofotos/`
Ortofotografías originales organizadas por ciudad:
- **madrid/** - Ortofotografías de Madrid, Cundinamarca
- **mosquera/** - Ortofotografías de Mosquera
- **tabio/** - Ortofotografías de Tabio
- **tocancipa/** - Ortofotografías de Tocancipá

**Especificaciones:**
- GSD: 10 cm (Ground Sample Distance)
- Escala equivalente: 1:1000
- Bandas: RGB (verificar si incluye NIR)
- Sistema de coordenadas: MAGNA-SIRGAS origen Bogotá

### `vectores/`
Geodatabase vectorial del IGAC:
- **cartografia_igac.gdb/** - Feature Classes con 9 clases objetivo

**Clases incluidas:**
- LÍNEAS: Cerca, Vía, Límite_Vía, Cicloruta
- POLÍGONOS: Zona_Dura, Construcción_R, Deposito_Agua_R, Zona_Verde, Bosque

### `metadata/`
Documentación técnica y metadatos:
- Especificaciones técnicas IGAC
- Sistemas de referencia
- Documentación de captura
- Fechas y condiciones de vuelo

## ⚠️ Importante:

1. **NO MODIFICAR**: Estos datos son la fuente de verdad del proyecto
2. **NO VERSIONAR**: Archivos muy grandes, no incluir en Git (ver .gitignore)
3. **BACKUP**: Mantener copias de seguridad externas
4. **DOCUMENTAR**: Cualquier anomalía detectada debe documentarse en `metadata/`

## 📊 Checklist de Validación (Día 1):

- [ ] Verificar cobertura completa de 4 ciudades
- [ ] Confirmar GSD de 10 cm en todas las ortofotos
- [ ] Verificar sistema de coordenadas (MAGNA-SIRGAS)
- [ ] Confirmar bandas espectrales (RGB o RGB+NIR)
- [ ] Documentar fechas de captura
- [ ] Calcular área total cubierta (km²)
- [ ] Verificar integridad de GDB (abrir en QGIS/ArcGIS)
- [ ] Contar instancias por clase en vectores
- [ ] Documentar cualquier archivo corrupto o faltante

## 🔗 Referencias:

Ver `scripts/01_inventory_data.py` para proceso automatizado de inventario.
