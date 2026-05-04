"""Genera el Word del Anexo C (matrices de confusion) con heatmaps como imagenes."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENTATION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


HEADER_BG = "305496"
ALT_BG = "F2F2F2"

CLASS_FULL = [
    "Background", "Construccion", "Bosque", "DepositoAgua",
    "ZonaVerde", "ZonaDura", "Via", "Cerca", "CabeceraMunicipal",
]
CLASS_ABBR = ["BG", "Con", "Bos", "DAg", "ZV", "ZD", "Via", "Cer", "CM"]


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
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9)


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


def fmt_abs(v):
    if v == 0:
        return "0"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{int(v)}"


def plot_confusion_matrix(cm, mode, title, out_path):
    """mode: 'abs' | 'row' | 'col'."""
    n = cm.shape[0]
    if mode == "abs":
        data = cm.astype(np.float64).copy()
        data_display = cm
        cbar_label = "Pixeles (escala log)"
    elif mode == "row":
        rs = cm.sum(axis=1, keepdims=True)
        data = np.where(rs > 0, cm / rs, 0.0)
        data_display = data
        cbar_label = "Recall (proporcion)"
    elif mode == "col":
        cs = cm.sum(axis=0, keepdims=True)
        data = np.where(cs > 0, cm / cs, 0.0)
        data_display = data
        cbar_label = "Precision (proporcion)"
    else:
        raise ValueError(mode)

    fig, ax = plt.subplots(figsize=(10, 8))

    if mode == "abs":
        plot_data = np.where(data > 0, data, np.nan)
        vmin = max(1, np.nanmin(plot_data)) if np.any(data > 0) else 1
        vmax = np.nanmax(plot_data)
        im = ax.imshow(
            plot_data,
            cmap="Blues",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            aspect="auto",
        )
    else:
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_ABBR, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CLASS_ABBR, fontsize=10)
    ax.set_xlabel("Prediccion", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Texto en cada celda
    threshold_color = (np.nanmax(plot_data) / 4) if mode == "abs" else 0.5
    for i in range(n):
        for j in range(n):
            v = data_display[i, j]
            if mode == "abs":
                txt = fmt_abs(v) if v > 0 else "."
            else:
                txt = f"{v:.2f}" if v > 0.005 else "."
            # color de texto segun intensidad
            if mode == "abs":
                show_v = v if v > 0 else 0
                color = "white" if show_v > threshold_color else "black"
            else:
                color = "white" if v > threshold_color else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8)

    # Resaltar diagonal con borde
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#2E7D32", linewidth=2))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def per_class_metrics(cm):
    n = cm.shape[0]
    rows = []
    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        rows.append((CLASS_FULL[k], recall, precision, iou))
    return rows


def add_metrics_table(doc, cm):
    rows = per_class_metrics(cm)
    table = doc.add_table(rows=len(rows) + 1, cols=4)
    table.style = "Light Grid Accent 1"
    table.autofit = True
    hdr = table.rows[0].cells
    for i, h in enumerate(["Clase", "Recall", "Precision", "IoU"]):
        hdr[i].text = h
    style_header_row(table.rows[0])
    for r_idx, (name, rec, prec, iou) in enumerate(rows, start=1):
        cells = table.rows[r_idx].cells
        cells[0].text = name
        cells[1].text = ("-" if np.isnan(rec) else f"{rec:.4f}")
        cells[2].text = ("-" if np.isnan(prec) else f"{prec:.4f}")
        cells[3].text = ("-" if np.isnan(iou) else f"{iou:.4f}")
        if r_idx % 2 == 0:
            for c in cells:
                set_cell_bg(c, ALT_BG)
        for c in cells:
            for para in c.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)


def add_three_cm_images(doc, cm, model_label, fig_dir, file_prefix):
    """Genera y embebe las 3 matrices (abs, row, col) como imagenes."""
    paths = []
    for mode, suffix, title_suffix in [
        ("abs", "abs", "valores absolutos"),
        ("row", "row", "normalizada por fila (recall)"),
        ("col", "col", "normalizada por columna (precision)"),
    ]:
        out = fig_dir / f"{file_prefix}_{suffix}.png"
        plot_confusion_matrix(cm, mode,
                              f"{model_label} - {title_suffix}", out)
        paths.append((mode, out, title_suffix))

    for mode, p, title_suffix in paths:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run()
        run.add_picture(str(p), width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cap.add_run(f"Figura. {model_label} - matriz de confusion {title_suffix}.")
        cr.italic = True
        cr.font.size = Pt(9)


def cm_from_mosaics(gt, pred, n=9):
    mask = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[mask].astype(int) + pred[mask].astype(int), minlength=n * n).reshape(n, n)


def build_document(output_path: Path, project_root: Path):
    fig_dir = project_root / "results" / "figures" / "anexo_c"
    fig_dir.mkdir(parents=True, exist_ok=True)

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    section = doc.sections[0]
    section.orientation = WD_ORIENTATION.PORTRAIT
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)

    title = doc.add_heading(
        "Anexo C. Matrices de Confusion Completas por Modelo y Municipio",
        level=0,
    )
    for run in title.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x68)

    add_para(
        doc,
        "El presente anexo presenta las matrices de confusion completas (9 x 9) calculadas sobre el "
        "conjunto de prueba para cada uno de los tres modelos evaluados, asi como su descomposicion "
        "por municipio para identificar comportamientos territoriales especificos. Las matrices se "
        "reportan en valores absolutos (numero de pixeles) y normalizadas por fila (recall por clase) "
        "y por columna (precision por clase).",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )

    add_heading(doc, "C.1 Convenciones de lectura", level=1)
    add_para(
        doc,
        "Las filas representan el ground truth (clase real); las columnas representan la prediccion "
        "del modelo. La diagonal principal (resaltada con borde verde) corresponde a las "
        "clasificaciones correctas (verdaderos positivos por clase). Para los valores absolutos se "
        "utiliza una escala logaritmica de color y la convencion 'k' = miles de pixeles, "
        "'M' = millones de pixeles. Las matrices normalizadas usan escala lineal de 0 a 1.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    add_para(doc, "Abreviaturas de clases utilizadas:", bold=True)
    abr_table = doc.add_table(rows=1 + len(CLASS_FULL), cols=3)
    abr_table.style = "Light Grid Accent 1"
    hdr = abr_table.rows[0].cells
    hdr[0].text = "Indice"
    hdr[1].text = "Clase"
    hdr[2].text = "Abreviatura"
    style_header_row(abr_table.rows[0])
    for i, name in enumerate(CLASS_FULL):
        cells = abr_table.rows[i + 1].cells
        cells[0].text = str(i)
        cells[1].text = name
        cells[2].text = CLASS_ABBR[i]
        if (i + 1) % 2 == 0:
            for c in cells:
                set_cell_bg(c, ALT_BG)

    cm_unet = np.load(project_root / "results/test_evaluation/unet_resnet34_baseline/confusion_matrix.npy")
    cm_dlv3 = np.load(project_root / "results/test_evaluation/deeplabv3_resnet50_baseline/confusion_matrix.npy")
    cm_unetpp = np.load(project_root / "results/test_evaluation/unetplusplus_resnet50_baseline/confusion_matrix.npy")

    models = [
        ("U-Net (ResNet-34)", cm_unet, "C.2", "unet"),
        ("DeepLabv3+ (ResNet-50)", cm_dlv3, "C.3", "deeplabv3"),
        ("U-Net++ (ResNet-50)", cm_unetpp, "C.4", "unetpp"),
    ]
    for name, cm, sec, prefix in models:
        add_heading(doc, f"{sec} Matriz de confusion 9 x 9 - {name} (test set completo)", level=1)
        add_three_cm_images(doc, cm, name, fig_dir, prefix)
        add_para(doc, f"{sec}.4 Metricas derivadas por clase.", bold=True)
        add_metrics_table(doc, cm)
        doc.add_paragraph("")

    add_heading(doc, "C.5 Matrices descompuestas por municipio (DeepLabv3+, modelo de produccion)", level=1)
    add_para(
        doc,
        "Las matrices por municipio se calcularon a partir de los mosaicos reconstruidos de "
        "prediccion y ground truth (results/mosaicos/{municipio}/) sobre el modelo de produccion "
        "DeepLabv3+. Tocancipa se utilizo como municipio de entrenamiento y por tanto no se incluye "
        "en esta descomposicion territorial; los mosaicos disponibles son Madrid, Mosquera y Tabio.",
        align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    )
    add_note(
        doc,
        "Para U-Net y U-Net++ no se reconstruyeron mosaicos completos de prediccion; por ello la "
        "descomposicion por municipio se reporta unicamente para el modelo seleccionado en "
        "produccion. El conjunto de prueba global (secciones C.2 a C.4) si agrega las contribuciones "
        "de todos los patches de prueba para los tres modelos.",
    )

    munis = []
    for muni in ["madrid", "mosquera", "tabio"]:
        gt = np.load(project_root / f"results/mosaicos/{muni}/gt_mosaic.npy")
        pr = np.load(project_root / f"results/mosaicos/{muni}/pred_mosaic.npy")
        cm = cm_from_mosaics(gt, pr)
        munis.append((muni.capitalize(), cm))

    for i, (name, cm) in enumerate(munis, start=1):
        sec = f"C.5.{i}"
        add_heading(doc, f"{sec} {name} - DeepLabv3+", level=2)
        total_px = cm.sum()
        valid_px = cm[1:, :].sum()
        add_para(
            doc,
            f"Pixeles totales del mosaico: {total_px/1_000_000:.2f} M. "
            f"Pixeles validos (excluyendo GT=Background): {valid_px/1_000_000:.2f} M.",
            italic=True,
            size=10,
        )
        add_three_cm_images(doc, cm, f"DeepLabv3+ - {name}", fig_dir, f"deeplabv3_{name.lower()}")
        add_para(doc, "Metricas por clase en este municipio.", bold=True)
        add_metrics_table(doc, cm)
        doc.add_paragraph("")

    add_heading(doc, "C.6 Sintesis comparativa entre municipios (DeepLabv3+)", level=1)
    add_para(doc, "Recall (sensibilidad) por clase y municipio:", bold=True)
    rec_rows = []
    for k in range(9):
        row = [CLASS_FULL[k]]
        for _, cm in munis:
            tp = cm[k, k]
            denom = cm[k, :].sum()
            row.append("-" if denom == 0 else f"{tp/denom:.4f}")
        rec_rows.append(row)
    rec_table = doc.add_table(rows=1 + len(rec_rows), cols=4)
    rec_table.style = "Light Grid Accent 1"
    hdr = rec_table.rows[0].cells
    hdr[0].text = "Clase"
    hdr[1].text = "Madrid"
    hdr[2].text = "Mosquera"
    hdr[3].text = "Tabio"
    style_header_row(rec_table.rows[0])
    for r_idx, row in enumerate(rec_rows, start=1):
        cells = rec_table.rows[r_idx].cells
        for c_idx, v in enumerate(row):
            cells[c_idx].text = v
            cells[c_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if r_idx % 2 == 0:
            for c in cells:
                set_cell_bg(c, ALT_BG)
        for c in cells:
            for para in c.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)

    add_para(doc, "")
    add_para(doc, "IoU por clase y municipio:", bold=True)
    iou_rows = []
    for k in range(9):
        row = [CLASS_FULL[k]]
        for _, cm in munis:
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            denom = tp + fp + fn
            row.append("-" if denom == 0 else f"{tp/denom:.4f}")
        iou_rows.append(row)
    iou_table = doc.add_table(rows=1 + len(iou_rows), cols=4)
    iou_table.style = "Light Grid Accent 1"
    hdr = iou_table.rows[0].cells
    hdr[0].text = "Clase"
    hdr[1].text = "Madrid"
    hdr[2].text = "Mosquera"
    hdr[3].text = "Tabio"
    style_header_row(iou_table.rows[0])
    for r_idx, row in enumerate(iou_rows, start=1):
        cells = iou_table.rows[r_idx].cells
        for c_idx, v in enumerate(row):
            cells[c_idx].text = v
            cells[c_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if r_idx % 2 == 0:
            for c in cells:
                set_cell_bg(c, ALT_BG)
        for c in cells:
            for para in c.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)

    add_heading(doc, "C.7 Observaciones sobre las confusiones dominantes", level=1)
    obs = [
        ("Background <-> CabeceraMunicipal:",
         " la clase CabeceraMunicipal absorbe gran parte del area no etiquetada del mosaico, lo que "
         "se manifiesta como una banda intensa en la columna CM. Esto eleva el recall global de CM "
         "pero baja su precision."),
        ("Cerca (lineal):",
         " presenta el peor IoU en los tres modelos (~0.10 - 0.12). Su confusion principal es con "
         "CabeceraMunicipal y con el fondo. La estrechez de la entidad (1-2 pixeles) explica este "
         "comportamiento, consistente con el descarte de Cerca en el calculo de mIoU6."),
        ("ZonaVerde:",
         " segunda peor clase (IoU < 0.27). Confunde sistematicamente con CabeceraMunicipal y con "
         "Background, debido al solape espectral con coberturas urbanas no construidas."),
        ("DepositoAgua:",
         " la confusion principal es con CabeceraMunicipal (sombras y suelos saturados pueden "
         "predecirse como agua superficial)."),
        ("Construccion y Bosque:",
         " las dos clases dominantes presentan diagonales fuertes (recall > 0.89 en DeepLabv3+) y "
         "justifican su seleccion para vectorizacion final."),
        ("Comparacion entre modelos:",
         " DeepLabv3+ supera a U-Net y U-Net++ en la diagonal de Construccion, Bosque y "
         "CabeceraMunicipal; U-Net++ muestra mayor concentracion fuera de la diagonal en clases "
         "minoritarias, consistente con su menor mIoU global (0.4236)."),
        ("Variacion territorial:",
         " los recalls de Construccion y Bosque se mantienen estables entre Madrid, Mosquera y "
         "Tabio; ZonaVerde tiene cero presencia en Mosquera y Tabio (ausente en el GT), por lo que "
         "sus metricas no se reportan para esos municipios."),
    ]
    for head, body in obs:
        p = doc.add_paragraph(style="List Number")
        run = p.add_run(head)
        run.bold = True
        p.add_run(body)

    doc.save(str(output_path))
    print(f"Archivo generado: {output_path}")
    print(f"Imagenes en: {fig_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output = project_root / "Anexo_C.docx"
    build_document(output, project_root)
