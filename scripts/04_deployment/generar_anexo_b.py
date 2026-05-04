"""Genera el Word del Anexo B (hiperparametros + bitacora) en la raiz del proyecto."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


HEADER_BG = "305496"
ALT_BG = "F2F2F2"


def set_cell_bg(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def style_header_row(row):
    for cell in row.cells:
        set_cell_bg(cell, HEADER_BG)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)


def style_body_cells(table, font_size=9):
    for i, row in enumerate(table.rows):
        if i == 0:
            continue
        if i % 2 == 0:
            for cell in row.cells:
                set_cell_bg(cell, ALT_BG)
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(font_size)


def add_table(doc, headers, rows, font_size=9, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = True

    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    style_header_row(table.rows[0])

    for r_idx, row_data in enumerate(rows, start=1):
        cells = table.rows[r_idx].cells
        for c_idx, val in enumerate(row_data):
            cells[c_idx].text = str(val)

    style_body_cells(table, font_size=font_size)

    if col_widths:
        for row in table.rows:
            for c_idx, w in enumerate(col_widths):
                row.cells[c_idx].width = w

    return table


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)
    return h


def add_para(doc, text, bold=False, italic=False, size=11, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run("Nota: ")
    run.bold = True
    run.italic = True
    run.font.size = Pt(10)
    run2 = p.add_run(text)
    run2.italic = True
    run2.font.size = Pt(10)


def build_document(output_path: Path):
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for section in doc.sections:
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)

    title = doc.add_heading("Anexo B. Hiperparametros Completos y Bitacora de Entrenamiento", level=0)
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)

    add_para(
        doc,
        "El presente anexo documenta los hiperparametros completos aplicados a cada una de las tres "
        "arquitecturas evaluadas (U-Net, DeepLabv3+, U-Net++) y la bitacora de entrenamiento que registra "
        "la trayectoria de las metricas por epoca. La documentacion incluye los valores de la tasa de "
        "aprendizaje en cada epoca bajo el scheduler CosineAnnealingWarmRestarts, los pesos de clase "
        "aplicados a la funcion de perdida, los parametros del aumento de datos y los registros del "
        "early stopping.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )

    # B.1
    add_heading(doc, "B.1 Configuracion computacional", level=1)
    add_table(
        doc,
        ["Parametro", "Valor"],
        [
            ["GPU", "NVIDIA GeForce RTX 4060 Laptop GPU"],
            ["VRAM total", "8.00 GB"],
            ["VRAM libre observada", "6.93 GB"],
            ["CUDA", "12.x"],
            ["Framework", "PyTorch + segmentation_models_pytorch"],
            ["Mixed Precision (AMP)", "Habilitado"],
            ["cudnn.benchmark", "True"],
            ["Sistema operativo", "Windows 11 Home"],
            ["num_workers (DataLoader)", "0 (requerido en Windows)"],
            ["Semilla aleatoria", "42"],
        ],
    )

    # B.2
    add_heading(doc, "B.2 Tabla maestra de hiperparametros", level=1)
    b2_rows = [
        ["Encoder backbone", "ResNet-34", "ResNet-50", "ResNet-50"],
        ["Pesos pre-entrenados", "ImageNet", "ImageNet", "ImageNet"],
        ["Parametros totales", "~24.4 M", "~39.6 M", "48,986,905"],
        ["Resolucion de entrada", "256 x 256 px", "256 x 256 px", "256 x 256 px"],
        ["Canales de entrada", "3 (RGB)", "3 (RGB)", "3 (RGB)"],
        ["Numero de clases", "9 (8 + background)", "9 (8 + background)", "9 (8 + background)"],
        ["Batch size", "8", "6", "4"],
        ["Gradient accumulation", "1", "1", "1"],
        ["Optimizador", "AdamW", "AdamW", "AdamW"],
        ["Learning rate inicial", "1e-4", "1e-4", "1e-4"],
        ["Weight decay", "1e-4", "1e-4", "1e-4"],
        ["Scheduler", "CosineAnnealingWarmRestarts", "CosineAnnealingWarmRestarts", "CosineAnnealingWarmRestarts"],
        ["T0 (primer ciclo)", "10", "10", "10"],
        ["T_mult", "2", "2", "2"],
        ["eta_min", "0", "0", "0"],
        ["Funcion de perdida", "CrossEntropyLoss ponderada (ignore_index=0)", "CrossEntropyLoss ponderada (ignore_index=0)", "CrossEntropyLoss ponderada (ignore_index=0)"],
        ["Mixed precision (AMP)", "Si", "Si", "Si"],
        ["Early stopping (patience)", "15 epocas", "15 epocas", "15 epocas"],
        ["Metrica de monitoreo", "mIoU validacion", "mIoU validacion", "mIoU validacion"],
        ["Epocas maximas configuradas", "100", "100", "100"],
        ["Epocas efectivas ejecutadas", "44", "42", "22"],
        ["Mejor epoca (best checkpoint)", "29", "27", "21"],
    ]
    add_table(doc, ["Hiperparametro", "U-Net", "DeepLabv3+", "U-Net++"], b2_rows, font_size=9)

    # B.3
    add_heading(doc, "B.3 Pesos de clase para la funcion de perdida", level=1)
    add_para(
        doc,
        "Los pesos se calcularon con la estrategia median frequency balancing sobre el conjunto de "
        "entrenamiento. Se cargan desde data/processed/class_weights.npy y se mantuvieron identicos "
        "para los tres modelos.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    add_table(
        doc,
        ["Indice", "Clase", "Peso"],
        [
            ["0", "Background (ignorado)", "0.000000"],
            ["1", "Construccion", "1.084058"],
            ["2", "Bosque", "0.548382"],
            ["3", "DepositoAgua", "5.415254"],
            ["4", "ZonaVerde", "59.666630"],
            ["5", "ZonaDura", "6.821473"],
            ["6", "Via", "1.558645"],
            ["7", "Cerca", "11.309224"],
            ["8", "CabeceraMunicipal", "0.239465"],
        ],
    )
    add_note(
        doc,
        "Background (0) se ignora con ignore_index=0 en CrossEntropyLoss. El peso 59.67 sobre ZonaVerde "
        "y 11.31 sobre Cerca refleja el desbalance extremo del dataset (clases muy minoritarias).",
    )

    # B.4
    add_heading(doc, "B.4 Pipeline de aumento de datos (Albumentations)", level=1)
    add_para(
        doc,
        "Aplicado unicamente al split de entrenamiento. Validacion y test solo aplican ToTensorV2().",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    add_para(doc, "Transformaciones geometricas (aplicadas a imagen Y mascara):", bold=True)
    add_table(
        doc,
        ["Transformacion", "Parametros", "Probabilidad"],
        [
            ["HorizontalFlip", "-", "0.5"],
            ["VerticalFlip", "-", "0.5"],
            ["RandomRotate90", "-", "0.5"],
            ["ShiftScaleRotate", "shift_limit=0.1, scale_limit=0.1, rotate_limit=15 deg, border_mode=0", "0.5"],
        ],
    )
    add_para(doc, "Transformaciones fotometricas (solo a la imagen):", bold=True)
    add_table(
        doc,
        ["Transformacion", "Parametros", "Probabilidad"],
        [
            ["RandomBrightnessContrast", "brightness_limit=+/-0.2, contrast_limit=+/-0.2", "0.5"],
            ["HueSaturationValue", "hue=+/-10, sat=+/-20, val=+/-10", "0.3"],
            ["GaussNoise", "var_limit=(10.0, 50.0)", "0.2"],
            ["GaussianBlur", "blur_limit=(3, 7)", "0.2"],
        ],
    )

    # B.5
    add_heading(doc, "B.5 Distribucion del dataset", level=1)
    add_table(
        doc,
        ["Split", "Patches"],
        [
            ["Train", "38,324"],
            ["Validacion", "15,625"],
            ["Test", "18,991"],
            ["Total", "72,940"],
        ],
    )
    add_para(doc, "Batches por epoca segun batch_size:", bold=True)
    add_table(
        doc,
        ["Modelo", "Train batches", "Val batches", "Test batches"],
        [
            ["U-Net (BS=8)", "4,791", "1,954", "2,374"],
            ["DeepLabv3+ (BS=6)", "6,388", "2,605", "3,166"],
            ["U-Net++ (BS=4)", "9,581", "3,907", "4,748"],
        ],
    )

    # B.6
    add_heading(doc, "B.6 Configuracion del scheduler CosineAnnealingWarmRestarts", level=1)
    add_para(
        doc,
        "Implementado con torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=0).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    bullets = [
        "T0 = 10: el primer ciclo coseno dura 10 epocas (epocas 1-10).",
        "T_mult = 2: cada reinicio duplica la longitud del ciclo siguiente.",
        "    Ciclo 1: epocas 1 -> 10",
        "    Ciclo 2: epocas 11 -> 30 (longitud 20)",
        "    Ciclo 3: epocas 31 -> 70 (longitud 40)",
        "eta_min = 0: el LR minimo en cada ciclo es ~0; al inicio de cada ciclo, el LR se reinicia "
        "bruscamente a 1e-4.",
        "En las trayectorias de las tres arquitecturas se observa el reinicio caracteristico en la "
        "epoca 10 (LR vuelve a 1.00e-04) y un segundo reinicio comenzaria en la epoca 30, cortado "
        "por el early stopping en U-Net y DeepLabv3+.",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    # B.7
    add_heading(doc, "B.7 Historico de tasas de aprendizaje efectivas (LR por epoca)", level=1)
    lr_table = [
        ("1", "9.758e-05", "16", "7.960e-05", "31", "9.985e-05"),
        ("2", "9.055e-05", "17", "7.297e-05", "32", "9.939e-05"),
        ("3", "7.960e-05", "18", "6.580e-05", "33", "9.863e-05"),
        ("4", "6.580e-05", "19", "5.824e-05", "34", "9.758e-05"),
        ("5", "5.050e-05", "20", "5.050e-05", "35", "9.623e-05"),
        ("6", "3.520e-05", "21", "4.276e-05", "36", "9.460e-05"),
        ("7", "2.140e-05", "22", "3.520e-05", "37", "9.271e-05"),
        ("8", "1.045e-05", "23", "2.803e-05", "38", "9.055e-05"),
        ("9", "3.423e-06", "24", "2.140e-05", "39", "8.814e-05"),
        ("10 (reinicio)", "1.000e-04", "25", "1.550e-05", "40", "8.550e-05"),
        ("11", "9.939e-05", "26", "1.045e-05", "41", "8.265e-05"),
        ("12", "9.758e-05", "27", "6.395e-06", "42", "7.960e-05"),
        ("13", "9.460e-05", "28", "3.423e-06", "43", "7.636e-05"),
        ("14", "9.055e-05", "29", "1.609e-06", "44", "7.297e-05"),
        ("15", "8.550e-05", "30 (reinicio)", "1.000e-04", "", ""),
    ]
    add_table(
        doc,
        ["Epoca", "LR", "Epoca", "LR", "Epoca", "LR"],
        lr_table,
        font_size=9,
    )
    add_note(doc, "Los valores son identicos para las tres arquitecturas (mismos T0, T_mult, LR inicial).")

    # B.8
    add_heading(doc, "B.8 Bitacora de entrenamiento - U-Net (ResNet-34)", level=1)
    unet_data = [
        (1, 1.6937, 0.1618, 0.3229, "9.76e-05"),
        (2, 1.5152, 0.1983, 0.3463, "9.05e-05"),
        (3, 1.4365, 0.2207, 0.3592, "7.96e-05"),
        (4, 1.3630, 0.2397, 0.3714, "6.58e-05"),
        (5, 1.3091, 0.2524, 0.3736, "5.05e-05"),
        (6, 1.2587, 0.2652, 0.3705, "3.52e-05"),
        (7, 1.2163, 0.2759, 0.4024, "2.14e-05"),
        (8, 1.1857, 0.2872, 0.4248, "1.05e-05"),
        (9, 1.1536, 0.2976, 0.4154, "3.42e-06"),
        (10, 1.1453, 0.3005, 0.4209, "1.00e-04"),
        (11, 1.2661, 0.2658, 0.3921, "9.94e-05"),
        (12, 1.2726, 0.2644, 0.3742, "9.76e-05"),
        (13, 1.2491, 0.2714, 0.3988, "9.46e-05"),
        (14, 1.2313, 0.2758, 0.4178, "9.05e-05"),
        (15, 1.2091, 0.2835, 0.4271, "8.55e-05"),
        (16, 1.1881, 0.2848, 0.4016, "7.96e-05"),
        (17, 1.1707, 0.2910, 0.4201, "7.30e-05"),
        (18, 1.1540, 0.3003, 0.4087, "6.58e-05"),
        (19, 1.1484, 0.3015, 0.4475, "5.82e-05"),
        (20, 1.1131, 0.3077, 0.4159, "5.05e-05"),
        (21, 1.1007, 0.3123, 0.4467, "4.28e-05"),
        (22, 1.0938, 0.3129, 0.4282, "3.52e-05"),
        (23, 1.0709, 0.3195, 0.4417, "2.80e-05"),
        (24, 1.0552, 0.3214, 0.4543, "2.14e-05"),
        (25, 1.0392, 0.3248, 0.4456, "1.55e-05"),
        (26, 1.0445, 0.3258, 0.4638, "1.05e-05"),
        (27, 1.0245, 0.3314, 0.4591, "6.40e-06"),
        (28, 1.0183, 0.3329, 0.4607, "3.42e-06"),
        (29, 1.0023, 0.3395, 0.4670, "1.61e-06"),
        (30, 1.0144, 0.3348, 0.4598, "1.00e-04"),
        (31, 1.1349, 0.3035, 0.4307, "9.98e-05"),
        (32, 1.1439, 0.2975, 0.4336, "9.94e-05"),
        (33, 1.1318, 0.3021, 0.4506, "9.86e-05"),
        (34, 1.1321, 0.2995, 0.4344, "9.76e-05"),
        (35, 1.1182, 0.3055, 0.4528, "9.62e-05"),
        (36, 1.1108, 0.3068, 0.4169, "9.46e-05"),
        (37, 1.1022, 0.3145, 0.4540, "9.27e-05"),
        (38, 1.1054, 0.3061, 0.4476, "9.05e-05"),
        (39, 1.0944, 0.3106, 0.4215, "8.81e-05"),
        (40, 1.0763, 0.3155, 0.4537, "8.55e-05"),
        (41, 1.0847, 0.3117, 0.4403, "8.26e-05"),
        (42, 1.0705, 0.3166, 0.4250, "7.96e-05"),
        (43, 1.0707, 0.3194, 0.4444, "7.64e-05"),
        (44, 1.0489, 0.3232, 0.4486, "7.30e-05"),
    ]
    rows = []
    for ep, tl, tm, vm, lr in unet_data:
        marker = " *" if ep == 29 else ""
        rows.append([str(ep) + marker, f"{tl:.4f}", f"{tm:.4f}", f"{vm:.4f}", lr])
    add_table(doc, ["Epoca", "Train Loss", "Train mIoU", "Val mIoU", "LR"], rows, font_size=9)
    add_note(
        doc,
        "Mejor mIoU de validacion: 0.4670 en la epoca 29 (marcada con *). Early stopping disparado "
        "tras 15 epocas sin mejora (epoca 44).",
    )

    # B.9
    add_heading(doc, "B.9 Bitacora de entrenamiento - DeepLabv3+ (ResNet-50)", level=1)
    deeplab_data = [
        (1, 1.5393, 0.1826, 0.3611, "9.76e-05"),
        (2, 1.3654, 0.2283, 0.3953, "9.05e-05"),
        (3, 1.2925, 0.2465, 0.4062, "7.96e-05"),
        (4, 1.2359, 0.2594, 0.4022, "6.58e-05"),
        (5, 1.1888, 0.2737, 0.3917, "5.05e-05"),
        (6, 1.1547, 0.2813, 0.4367, "3.52e-05"),
        (7, 1.1109, 0.2952, 0.4493, "2.14e-05"),
        (8, 1.0856, 0.3039, 0.4602, "1.05e-05"),
        (9, 1.0608, 0.3056, 0.4398, "3.42e-06"),
        (10, 1.0481, 0.3163, 0.4509, "1.00e-04"),
        (11, 1.1753, 0.2787, 0.3918, "9.94e-05"),
        (12, 1.1729, 0.2799, 0.4134, "9.76e-05"),
        (13, 1.1568, 0.2815, 0.3986, "9.46e-05"),
        (14, 1.1334, 0.2890, 0.4470, "9.05e-05"),
        (15, 1.1128, 0.2933, 0.4317, "8.55e-05"),
        (16, 1.1113, 0.2950, 0.4319, "7.96e-05"),
        (17, 1.0881, 0.3045, 0.4246, "7.30e-05"),
        (18, 1.0676, 0.3056, 0.4399, "6.58e-05"),
        (19, 1.0549, 0.3103, 0.4319, "5.82e-05"),
        (20, 1.0443, 0.3050, 0.4531, "5.05e-05"),
        (21, 1.0258, 0.3202, 0.4380, "4.28e-05"),
        (22, 1.0207, 0.3228, 0.4725, "3.52e-05"),
        (23, 1.0091, 0.3249, 0.4606, "2.80e-05"),
        (24, 0.9904, 0.3287, 0.4571, "2.14e-05"),
        (25, 0.9822, 0.3341, 0.4770, "1.55e-05"),
        (26, 0.9593, 0.3419, 0.4780, "1.05e-05"),
        (27, 0.9538, 0.3436, 0.4813, "6.40e-06"),
        (28, 0.9475, 0.3453, 0.4748, "3.42e-06"),
        (29, 0.9429, 0.3484, 0.4785, "1.61e-06"),
        (30, 0.9308, 0.3518, 0.4757, "1.00e-04"),
        (31, 1.0673, 0.3034, 0.4193, "9.98e-05"),
        (32, 1.0647, 0.3079, 0.4077, "9.94e-05"),
        (33, 1.0605, 0.3092, 0.4635, "9.86e-05"),
        (34, 1.0637, 0.3059, 0.4592, "9.76e-05"),
        (35, 1.0344, 0.3172, 0.4465, "9.62e-05"),
        (36, 1.0459, 0.3140, 0.4666, "9.46e-05"),
        (37, 1.0411, 0.3128, 0.4359, "9.27e-05"),
        (38, 1.0185, 0.3199, 0.4512, "9.05e-05"),
        (39, 1.0259, 0.3199, 0.4608, "8.81e-05"),
        (40, 1.0191, 0.3196, 0.4726, "8.55e-05"),
        (41, 1.0111, 0.3187, 0.4805, "8.26e-05"),
        (42, 0.9943, 0.3298, 0.4607, "7.96e-05"),
    ]
    rows = []
    for ep, tl, tm, vm, lr in deeplab_data:
        marker = " *" if ep == 27 else ""
        rows.append([str(ep) + marker, f"{tl:.4f}", f"{tm:.4f}", f"{vm:.4f}", lr])
    add_table(doc, ["Epoca", "Train Loss", "Train mIoU", "Val mIoU", "LR"], rows, font_size=9)
    add_note(
        doc,
        "Mejor mIoU de validacion: 0.4813 en la epoca 27 (marcada con *). Early stopping disparado "
        "tras 15 epocas sin mejora (epoca 42).",
    )

    # B.10
    add_heading(doc, "B.10 Bitacora de entrenamiento - U-Net++ (ResNet-50)", level=1)
    unetpp_data = [
        (1, 1.6704, 0.1644, 0.3459, "9.76e-05", "1:43:22", "17:01"),
        (2, 1.4933, 0.2100, 0.3753, "9.05e-05", "1:38:07", "16:56"),
        (3, 1.4167, 0.2304, 0.3960, "7.96e-05", "1:37:52", "16:58"),
        (4, 1.3623, 0.2446, 0.3911, "6.58e-05", "1:37:17", "16:43"),
        (5, 1.3114, 0.2581, 0.3743, "5.05e-05", "1:37:10", "16:43"),
        (6, 1.2646, 0.2701, 0.4292, "3.52e-05", "1:39:12", "20:01"),
        (7, 1.2287, 0.2818, 0.4412, "2.14e-05", "1:50:46", "21:37"),
        (8, 1.1960, 0.2915, 0.4092, "1.05e-05", "1:51:24", "21:53"),
        (9, 1.1791, 0.2966, 0.4199, "3.42e-06", "1:52:08", "20:51"),
        (10, 1.1533, 0.3032, 0.4414, "1.00e-04", "1:40:09", "19:04"),
        (11, 1.2814, 0.2660, 0.3771, "9.94e-05", "1:40:33", "17:26"),
        (12, 1.2792, 0.2689, 0.3480, "9.76e-05", "1:39:31", "17:29"),
        (13, 1.2582, 0.2718, 0.3865, "9.46e-05", "1:39:08", "17:21"),
        (14, 1.2441, 0.2760, 0.4133, "9.05e-05", "1:39:09", "17:21"),
        (15, 1.2150, 0.2819, 0.4160, "8.55e-05", "1:38:51", "17:16"),
        (16, 1.2107, 0.2880, 0.4388, "7.96e-05", "1:38:43", "17:06"),
        (17, 1.1849, 0.2920, 0.4261, "7.30e-05", "1:38:22", "17:05"),
        (18, 1.1703, 0.2980, 0.4054, "6.58e-05", "1:45:07", "21:41"),
        (19, 1.1489, 0.3056, 0.4160, "5.82e-05", "1:53:49", "22:22"),
        (20, 1.1407, 0.3071, 0.4415, "5.05e-05", "1:53:35", "20:38"),
        (21, 1.1267, 0.3099, 0.4540, "4.28e-05", "1:55:26", "21:05"),
    ]
    rows = []
    for ep, tl, tm, vm, lr, tt, tv in unetpp_data:
        marker = " *" if ep == 21 else ""
        rows.append([str(ep) + marker, f"{tl:.4f}", f"{tm:.4f}", f"{vm:.4f}", lr, tt, tv])
    rows.append(["22 (parcial)", "1.1161", "-", "-", "3.52e-05", "interrumpido al 35%", "-"])
    add_table(
        doc,
        ["Epoca", "Train Loss", "Train mIoU", "Val mIoU", "LR", "Tiempo Train", "Tiempo Val"],
        rows,
        font_size=9,
    )
    add_note(
        doc,
        "Mejor mIoU de validacion: 0.4540 en la epoca 21 (marcada con *). El entrenamiento se "
        "interrumpio manualmente durante la epoca 22 por restricciones de tiempo de computo "
        "(~2 h por epoca sobre RTX 4060 Laptop), antes de alcanzar el criterio formal de early stopping.",
    )

    # B.11
    add_heading(doc, "B.11 Configuracion del Early Stopping", level=1)
    add_table(
        doc,
        ["Parametro", "Valor"],
        [
            ["Metrica monitoreada", "mIoU sobre validacion (modo: maximizar)"],
            ["Patience", "15 epocas"],
            ["Min delta", "0 (cualquier mejora se considera valida)"],
            ["Accion", "Carga del mejor checkpoint y detencion del entrenamiento"],
        ],
    )
    add_para(doc, "Resultados del early stopping:", bold=True)
    add_table(
        doc,
        ["Modelo", "Mejor epoca", "Mejor val mIoU", "Epocas sin mejora", "Detencion"],
        [
            ["U-Net", "29", "0.4670", "15 (epocas 30-44)", "Epoca 44 (early stopping)"],
            ["DeepLabv3+", "27", "0.4813", "15 (epocas 28-42)", "Epoca 42 (early stopping)"],
            ["U-Net++", "21", "0.4540", "1 (parcial)", "Epoca 22 (interrupcion manual)"],
        ],
    )

    # B.12
    add_heading(doc, "B.12 Tiempos totales de entrenamiento", level=1)
    add_table(
        doc,
        ["Modelo", "Epocas ejecutadas", "Tiempo medio/epoca (train+val)", "Tiempo total aprox."],
        [
            ["U-Net (ResNet-34)", "44", "~50 min", "~37 horas"],
            ["DeepLabv3+ (ResNet-50)", "42", "~75 min", "~52 horas"],
            ["U-Net++ (ResNet-50)", "22", "~125 min", "~46 horas"],
        ],
    )
    add_note(
        doc,
        "Los tiempos para U-Net y DeepLabv3+ son estimaciones a partir de los logs de TensorBoard; "
        "los de U-Net++ son medidos directamente desde la salida estandar del notebook (columnas "
        "Tiempo Train / Tiempo Val en B.10).",
    )

    # B.13
    add_heading(doc, "B.13 Consumo de VRAM observado", level=1)
    add_table(
        doc,
        ["Modelo", "Batch size", "VRAM pico observada", "Estrategia"],
        [
            ["U-Net (ResNet-34)", "8", "~5.8 GB", "AMP + cuDNN benchmark"],
            ["DeepLabv3+ (ResNet-50)", "6", "~7.1 GB", "AMP + cuDNN benchmark"],
            ["U-Net++ (ResNet-50)", "4", "~6.5 GB", "AMP + cuDNN benchmark"],
        ],
    )

    # B.14
    add_heading(doc, "B.14 Curvas de perdida y mIoU por epoca", level=1)
    add_para(
        doc,
        "Las curvas se generan a partir de los archivos:",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    for line in [
        "models/checkpoints/unet_resnet34_baseline/training_history.json",
        "models/checkpoints/deeplabv3_resnet50_baseline/training_history.json",
        "Logs del notebook notebooks/04_train_unetplusplus.ipynb (U-Net++ no exporto training_history.json debido a la interrupcion manual).",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    add_para(doc, "Figura B.1.", bold=True)
    add_para(doc, "Curvas de Train Loss por epoca para los tres modelos.")
    add_para(doc, "Figura B.2.", bold=True)
    add_para(doc, "Curvas de Train/Val mIoU por epoca, con marcado del best checkpoint en cada arquitectura.")
    add_para(doc, "Figura B.3.", bold=True)
    add_para(
        doc,
        "Trayectoria del learning rate efectivo (escala logaritmica) mostrando los reinicios del "
        "scheduler CosineAnnealingWarmRestarts en las epocas 10 y 30.",
    )

    # B.15
    add_heading(doc, "B.15 Observaciones sobre la trayectoria de entrenamiento", level=1)
    obs = [
        ("Reinicio del scheduler en la epoca 10:",
         " las tres arquitecturas muestran un retroceso transitorio del val mIoU al reiniciar el LR a "
         "1e-4 (efecto esperado de WarmRestarts). La recuperacion se completa hacia la epoca 20."),
        ("Convergencia diferenciada:",
         " DeepLabv3+ alcanza 0.46 de val mIoU en la epoca 8, antes que las otras arquitecturas, "
         "gracias al modulo ASPP. U-Net++ presenta convergencia mas lenta debido al mayor numero "
         "de parametros (49 M) y al batch size reducido (4) impuesto por la limitacion de VRAM."),
        ("val_loss = NaN:",
         " todas las bitacoras registran NaN en val_loss. Esto se debe a que la funcion de perdida "
         "con ignore_index=0 puede generar valores no definidos cuando un batch contiene "
         "exclusivamente pixeles de fondo. La metrica de seleccion del mejor modelo (val_mIoU) no "
         "se ve afectada por este artefacto y es la utilizada para todas las decisiones de checkpoint."),
        ("Decision final:",
         " DeepLabv3+ (ResNet-50) fue seleccionado como modelo de produccion al obtener el mejor "
         "val mIoU (0.4813) y el mejor mIoU6 en test (0.5746)."),
    ]
    for i, (head, body) in enumerate(obs, start=1):
        p = doc.add_paragraph(style="List Number")
        run = p.add_run(head)
        run.bold = True
        p.add_run(body)

    doc.save(str(output_path))
    print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    output = Path(__file__).resolve().parent.parent / "Anexo_B.docx"
    build_document(output)
