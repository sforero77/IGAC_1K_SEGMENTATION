"""
Script para analizar balance de clases y calcular class weights

OBJETIVO:
- Contar píxeles por clase en el conjunto TRAIN
- Calcular métricas de desbalance
- Calcular class weights para la función de pérdida
- Guardar class_weights.json

MÉTODOS DE CÁLCULO DE WEIGHTS:
1. Inverse Frequency: w_i = 1 / freq_i
2. Inverse Frequency Normalizado: w_i = N / (n_classes * n_i)
3. Effective Number of Samples: w_i = (1 - beta) / (1 - beta^n_i)

SALIDAS:
- data/interim/class_weights.json
- data/interim/class_balance_report.txt
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Directorios
PATCHES_DIR = r"D:\IGAC_1K_SEGMENTATION\data\processed\patches"
PATCHES_INDEX_PATH = r"D:\IGAC_1K_SEGMENTATION\data\processed\patches\patches_index.csv"
OUTPUT_JSON = r"D:\IGAC_1K_SEGMENTATION\data\interim\class_weights.json"
OUTPUT_REPORT = r"D:\IGAC_1K_SEGMENTATION\data\interim\class_balance_report.txt"

# Mapeo de clases
CLASS_NAMES = {
    0: 'NoData/Background',
    1: 'Construccion',
    2: 'Bosque',
    3: 'DepositoAgua',
    4: 'ZonaVerde',
    5: 'ZonaDura',
    6: 'Via',
    7: 'Cerca',
    8: 'CabeceraMunicipal'
}

N_CLASSES = 9  # 0-8

# Beta para Effective Number of Samples (típicamente 0.99 o 0.999)
BETA = 0.999


# ============================================================================
# FUNCIONES
# ============================================================================

def count_pixels_per_class(masks_dir, patches_index_df):
    """
    Cuenta píxeles por clase en todas las máscaras de TRAIN

    Args:
        masks_dir: Directorio con máscaras
        patches_index_df: DataFrame con índice de patches (solo TRAIN)

    Returns:
        numpy array con conteo de píxeles por clase
    """
    print("\n  Contando píxeles por clase...")

    # Inicializar contadores
    pixel_counts = np.zeros(N_CLASSES, dtype=np.int64)

    # Procesar cada máscara
    for idx, row in tqdm(patches_index_df.iterrows(), total=len(patches_index_df), desc="  Procesando máscaras"):
        mask_path = row['mask_path']

        # Leer máscara
        try:
            mask = np.load(mask_path)

            # Contar píxeles por clase
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if 0 <= class_id < N_CLASSES:
                    pixel_counts[class_id] += count

        except Exception as e:
            print(f"\n    ERROR leyendo {mask_path}: {e}")
            continue

    return pixel_counts


def calculate_class_weights(pixel_counts, method='inverse_freq_normalized'):
    """
    Calcula class weights según el método especificado

    Args:
        pixel_counts: Array con conteo de píxeles por clase
        method: 'inverse_freq', 'inverse_freq_normalized', 'effective_samples'

    Returns:
        numpy array con weights por clase
    """
    # Total de píxeles (excluyendo background/NoData)
    total_pixels = np.sum(pixel_counts[1:])  # Suma clases 1-8
    n_classes = N_CLASSES - 1  # 8 clases (sin contar 0)

    weights = np.zeros(N_CLASSES, dtype=np.float32)

    if method == 'inverse_freq':
        # w_i = 1 / freq_i
        frequencies = pixel_counts / total_pixels
        # Evitar división por cero
        frequencies[frequencies == 0] = 1e-10
        weights = 1.0 / frequencies

    elif method == 'inverse_freq_normalized':
        # w_i = N / (n_classes * n_i)
        # Evitar división por cero
        pixel_counts_safe = pixel_counts.copy()
        pixel_counts_safe[pixel_counts_safe == 0] = 1
        weights = total_pixels / (n_classes * pixel_counts_safe)

    elif method == 'effective_samples':
        # w_i = (1 - beta) / (1 - beta^n_i)
        # Effective Number of Samples (ENS)
        for i in range(N_CLASSES):
            if pixel_counts[i] == 0:
                weights[i] = 0.0
            else:
                effective_num = (1.0 - BETA) / (1.0 - BETA ** pixel_counts[i])
                weights[i] = 1.0 / effective_num

        # Normalizar weights (suma = n_classes)
        weights_sum = np.sum(weights[1:])  # Suma clases 1-8
        if weights_sum > 0:
            weights[1:] = weights[1:] / weights_sum * n_classes

    # Siempre poner weight=0 para background (clase 0)
    weights[0] = 0.0

    return weights


def calculate_imbalance_metrics(pixel_counts):
    """
    Calcula métricas de desbalance

    Args:
        pixel_counts: Array con conteo de píxeles por clase

    Returns:
        dict con métricas
    """
    # Excluir background (clase 0)
    counts_no_bg = pixel_counts[1:]

    # Ratio máximo/mínimo
    max_count = np.max(counts_no_bg)
    min_count = np.min(counts_no_bg[counts_no_bg > 0])  # Ignorar clases con 0 píxeles

    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    # Clase mayoritaria y minoritaria
    majority_class = np.argmax(counts_no_bg) + 1  # +1 porque excluimos clase 0
    minority_class = np.argmin(counts_no_bg[counts_no_bg > 0]) + 1

    # Coeficiente de variación
    mean_count = np.mean(counts_no_bg)
    std_count = np.std(counts_no_bg)
    cv = (std_count / mean_count) if mean_count > 0 else 0

    return {
        'imbalance_ratio': float(imbalance_ratio),
        'majority_class': int(majority_class),
        'minority_class': int(minority_class),
        'coefficient_variation': float(cv),
        'max_count': int(max_count),
        'min_count': int(min_count)
    }


def generate_report(pixel_counts, weights_dict, metrics, output_path):
    """
    Genera reporte de balance de clases

    Args:
        pixel_counts: Array con conteo de píxeles por clase
        weights_dict: Dict con weights calculados por diferentes métodos
        metrics: Dict con métricas de desbalance
        output_path: Path del archivo de salida
    """
    total_pixels = np.sum(pixel_counts)
    total_pixels_no_bg = np.sum(pixel_counts[1:])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE BALANCE DE CLASES\n")
        f.write("="*80 + "\n\n")

        # Estadísticas generales
        f.write("ESTADISTICAS GENERALES:\n")
        f.write(f"  Total de pixeles: {total_pixels:,}\n")
        f.write(f"  Pixeles de clases (sin background): {total_pixels_no_bg:,}\n")
        f.write(f"  Pixeles de background (clase 0): {pixel_counts[0]:,} ({pixel_counts[0]/total_pixels*100:.2f}%)\n")
        f.write(f"\n  Numero de clases: {N_CLASSES - 1} (1-8)\n\n")

        # Distribución por clase
        f.write("DISTRIBUCION POR CLASE:\n")
        f.write(f"  {'Clase':<5} {'Nombre':<25} {'Pixeles':>15} {'% del total':>12} {'% sin BG':>12}\n")
        f.write("  " + "-"*75 + "\n")

        for class_id in range(N_CLASSES):
            count = pixel_counts[class_id]
            pct_total = (count / total_pixels) * 100 if total_pixels > 0 else 0

            if class_id == 0:
                pct_no_bg = 0.0
            else:
                pct_no_bg = (count / total_pixels_no_bg) * 100 if total_pixels_no_bg > 0 else 0

            class_name = CLASS_NAMES.get(class_id, f'Clase_{class_id}')
            f.write(f"  {class_id:<5} {class_name:<25} {count:>15,} {pct_total:>11.2f}% {pct_no_bg:>11.2f}%\n")

        # Métricas de desbalance
        f.write("\n\nMETRICAS DE DESBALANCE:\n")
        f.write(f"  Ratio de desbalance (max/min): {metrics['imbalance_ratio']:.2f}\n")
        f.write(f"  Clase mayoritaria: {metrics['majority_class']} ({CLASS_NAMES[metrics['majority_class']]})\n")
        f.write(f"  Clase minoritaria: {metrics['minority_class']} ({CLASS_NAMES[metrics['minority_class']]})\n")
        f.write(f"  Coeficiente de variacion: {metrics['coefficient_variation']:.4f}\n")
        f.write(f"  Max count: {metrics['max_count']:,}\n")
        f.write(f"  Min count: {metrics['min_count']:,}\n")

        # Interpretación
        f.write("\n\nINTERPRETACION:\n")
        if metrics['imbalance_ratio'] < 10:
            f.write("  - Dataset BALANCEADO (ratio < 10)\n")
        elif metrics['imbalance_ratio'] < 100:
            f.write("  - Dataset MODERADAMENTE DESBALANCEADO (10 <= ratio < 100)\n")
        else:
            f.write("  - Dataset ALTAMENTE DESBALANCEADO (ratio >= 100)\n")
            f.write("  - RECOMENDACION: Usar class weights en la funcion de perdida\n")

        # Class weights
        f.write("\n\nCLASS WEIGHTS CALCULADOS:\n\n")

        for method_name, weights in weights_dict.items():
            f.write(f"  {method_name.upper()}:\n")
            f.write(f"    {'Clase':<5} {'Nombre':<25} {'Weight':>12}\n")
            f.write("    " + "-"*45 + "\n")

            for class_id in range(N_CLASSES):
                if class_id == 0:
                    continue  # Skip background
                class_name = CLASS_NAMES.get(class_id, f'Clase_{class_id}')
                f.write(f"    {class_id:<5} {class_name:<25} {weights[class_id]:>12.6f}\n")

            f.write("\n")

        f.write("="*80 + "\n")

    print(f"\n  OK Reporte generado: {output_path}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal
    """
    print("="*80)
    print("ANALISIS DE BALANCE DE CLASES")
    print("="*80)

    # 1. Cargar índice de patches
    print("\n[1/5] Cargando indice de patches...")

    if not os.path.exists(PATCHES_INDEX_PATH):
        print(f"\n  ERROR: No se encontro {PATCHES_INDEX_PATH}")
        print("  Ejecuta primero: python scripts/06_create_patches.py")
        return

    patches_df = pd.read_csv(PATCHES_INDEX_PATH)

    # Filtrar solo TRAIN
    train_df = patches_df[patches_df['split'] == 'train']

    print(f"  Total patches en dataset: {len(patches_df)}")
    print(f"  Patches TRAIN: {len(train_df)}")

    if len(train_df) == 0:
        print("\n  ERROR: No hay patches en TRAIN")
        return

    # 2. Contar píxeles por clase
    print("\n[2/5] Contando pixeles por clase en TRAIN...")

    masks_dir = os.path.join(PATCHES_DIR, 'train', 'masks')
    pixel_counts = count_pixels_per_class(masks_dir, train_df)

    print(f"\n  Conteo completado:")
    total_pixels = np.sum(pixel_counts)
    for class_id in range(N_CLASSES):
        pct = (pixel_counts[class_id] / total_pixels) * 100
        print(f"    Clase {class_id} ({CLASS_NAMES[class_id]}): {pixel_counts[class_id]:,} ({pct:.2f}%)")

    # 3. Calcular métricas de desbalance
    print("\n[3/5] Calculando metricas de desbalance...")

    metrics = calculate_imbalance_metrics(pixel_counts)

    print(f"  Ratio de desbalance: {metrics['imbalance_ratio']:.2f}")
    print(f"  Clase mayoritaria: {metrics['majority_class']} ({CLASS_NAMES[metrics['majority_class']]})")
    print(f"  Clase minoritaria: {metrics['minority_class']} ({CLASS_NAMES[metrics['minority_class']]})")

    # 4. Calcular class weights (múltiples métodos)
    print("\n[4/5] Calculando class weights...")

    weights_dict = {
        'inverse_freq': calculate_class_weights(pixel_counts, 'inverse_freq'),
        'inverse_freq_normalized': calculate_class_weights(pixel_counts, 'inverse_freq_normalized'),
        'effective_samples': calculate_class_weights(pixel_counts, 'effective_samples')
    }

    print(f"\n  Metodos calculados:")
    for method_name in weights_dict.keys():
        print(f"    - {method_name}")

    # 5. Guardar resultados
    print("\n[5/5] Guardando resultados...")

    # Guardar JSON
    output_data = {
        'pixel_counts': pixel_counts.tolist(),
        'class_names': CLASS_NAMES,
        'metrics': metrics,
        'weights': {}
    }

    for method_name, weights in weights_dict.items():
        output_data['weights'][method_name] = weights.tolist()

    # Agregar recomendación de método
    if metrics['imbalance_ratio'] < 10:
        output_data['recommended_method'] = 'inverse_freq_normalized'
    elif metrics['imbalance_ratio'] < 100:
        output_data['recommended_method'] = 'effective_samples'
    else:
        output_data['recommended_method'] = 'effective_samples'

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  OK Guardado: {OUTPUT_JSON}")

    # Generar reporte
    generate_report(pixel_counts, weights_dict, metrics, OUTPUT_REPORT)

    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)

    print(f"\n  Total pixeles analizados: {total_pixels:,}")
    print(f"  Ratio de desbalance: {metrics['imbalance_ratio']:.2f}")

    print(f"\n  Metodo recomendado: {output_data['recommended_method']}")
    recommended_weights = weights_dict[output_data['recommended_method']]

    print(f"\n  Class weights recomendados:")
    for class_id in range(1, N_CLASSES):  # Skip background
        print(f"    Clase {class_id} ({CLASS_NAMES[class_id]}): {recommended_weights[class_id]:.6f}")

    print(f"\n  Archivos generados:")
    print(f"    - {OUTPUT_JSON}")
    print(f"    - {OUTPUT_REPORT}")

    print("\n  Proximo paso:")
    print("    - Usar class_weights en la funcion de perdida durante entrenamiento")
    print("    - Ejemplo: CrossEntropyLoss(weight=torch.tensor(weights))")

    print("\n[OK] Analisis completado exitosamente!")
    print("="*80)


if __name__ == "__main__":
    main()
