"""Convierte el informe Markdown a PDF. Requiere requirements-informe.txt.

Uso:
    python -m scripts.informe_pdf    # genera report/informe.pdf desde report/informe.md

Entiende un Markdown simple: títulos (#, ##, ###), párrafos, tablas, imágenes,
**negrita**, `código`, epígrafes de tabla en cursiva (*Tabla 1. ...*) y la marca
<!-- pagebreak --> para saltar de página.
"""
from pathlib import Path
import argparse
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from src.experimento import RAIZ

ANCHO = 491   # ancho útil de la página A4 con márgenes de 52 puntos


def formato_en_linea(texto):
    """Pasa **negrita** y `código` de Markdown al formato que entiende reportlab."""
    texto = escape(texto)
    texto = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', texto)
    texto = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', texto)
    return texto


def crear_estilos():
    estilos = getSampleStyleSheet()
    estilos['Title'].fontName = 'Helvetica-Bold'
    estilos['Title'].fontSize = 22
    estilos['Title'].leading = 27
    estilos['Title'].textColor = colors.black
    estilos['Title'].alignment = 0
    estilos['Title'].spaceAfter = 20
    for nombre in ['Heading1', 'Heading2', 'Heading3']:
        estilos[nombre].textColor = colors.black
        estilos[nombre].spaceBefore = 14
        estilos[nombre].spaceAfter = 8
        estilos[nombre].keepWithNext = 1   # un título nunca queda solo al pie de la página
    estilos['Heading1'].fontSize = 17
    estilos['Heading1'].leading = 21
    estilos['Heading2'].fontSize = 12
    estilos['Heading2'].leading = 15
    estilos.add(ParagraphStyle(name='Cuerpo', fontName='Helvetica', fontSize=10,
                               leading=14, spaceAfter=9, alignment=TA_JUSTIFY,
                               splitLongWords=True))
    estilos.add(ParagraphStyle(name='Celda', fontName='Helvetica', fontSize=8,
                               leading=10, spaceAfter=0))
    estilos.add(ParagraphStyle(name='Epigrafe', fontName='Helvetica-Oblique', fontSize=8,
                               leading=11, spaceAfter=12, textColor=colors.HexColor('#53616b')))
    estilos.add(ParagraphStyle(name='EpigrafeTabla', parent=estilos['Epigrafe'], spaceBefore=4,
                               spaceAfter=5, keepWithNext=1))
    # Las referencias van alineadas a la izquierda: justificadas, las URL abren huecos.
    estilos.add(ParagraphStyle(name='Referencia', parent=estilos['Cuerpo'], alignment=0))
    return estilos


def crear_tabla(filas, estilos):
    """Tabla con la primera columna más ancha (23 % si hay 5 columnas, 32 % si no)."""
    n = len(filas[0])
    primera = (.23 if n == 5 else .32) * ANCHO
    anchos = [primera] + [(ANCHO - primera) / (n - 1)] * (n - 1)
    celdas = [[Paragraph(formato_en_linea(c), estilos['Celda']) for c in fila] for fila in filas]
    tabla = Table(celdas, colWidths=anchos, repeatRows=1, hAlign='LEFT')
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfebf1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f4f6f7')]),
        ('GRID', (0, 0), (-1, -1), .5, colors.HexColor('#d9d9d9')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return tabla


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fuente', type=Path, default=RAIZ / 'report' / 'informe.md')
    fuente = parser.parse_args().fuente.resolve()
    estilos = crear_estilos()
    contenido = []
    lineas = fuente.read_text(encoding='utf-8').splitlines()

    i = 0
    en_referencias = False
    while i < len(lineas):
        linea = lineas[i].strip()
        if not linea:
            i += 1
            continue
        if linea == '<!-- pagebreak -->':
            contenido.append(PageBreak())
        elif linea.startswith('# '):
            contenido.append(Paragraph(formato_en_linea(linea[2:]), estilos['Title']))
        elif linea.startswith('## '):
            contenido.append(Paragraph(formato_en_linea(linea[3:]), estilos['Heading1']))
            en_referencias = linea[3:].strip() == 'Referencias'
        elif linea.startswith('### '):
            contenido.append(Paragraph(formato_en_linea(linea[4:]), estilos['Heading2']))
        elif re.fullmatch(r'\*[^*].*[^*]\*', linea):
            # Línea entera en cursiva (*Tabla 1. ...*): epígrafe pegado a la tabla siguiente.
            contenido.append(Paragraph(formato_en_linea(linea[1:-1]), estilos['EpigrafeTabla']))
        elif linea.startswith('|'):
            # Tabla: se leen todas las líneas que empiezan con | y se saltea la de guiones.
            filas = []
            while i < len(lineas) and lineas[i].strip().startswith('|'):
                celdas = [c.strip() for c in lineas[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+', c) for c in celdas):
                    filas.append(celdas)
                i += 1
            contenido.extend([crear_tabla(filas, estilos), Spacer(1, 12)])
            continue
        elif linea.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', linea)
            imagen = Image(str(fuente.parent / m.group(2)))
            proporcion = imagen.imageHeight / imagen.imageWidth
            imagen.drawWidth = ANCHO
            imagen.drawHeight = ANCHO * proporcion
            contenido.append(imagen)
            contenido.append(Paragraph(formato_en_linea(m.group(1)) + '. Fuente: experimentos del proyecto.',
                                       estilos['Epigrafe']))
        else:
            # Párrafo: se juntan las líneas seguidas hasta una línea vacía o un elemento nuevo.
            parrafo = [linea]
            while (i + 1 < len(lineas) and lineas[i + 1].strip()
                   and not lineas[i + 1].startswith(('#', '|', '![', '<!--'))):
                i += 1
                parrafo.append(lineas[i].strip())
            estilo = estilos['Referencia'] if en_referencias else estilos['Cuerpo']
            contenido.append(Paragraph(formato_en_linea(' '.join(parrafo)), estilo))
        i += 1

    def pie_de_pagina(lienzo, documento):
        lienzo.setFont('Helvetica', 8)
        lienzo.setFillColor(colors.HexColor('#5b6770'))
        lienzo.drawString(52, 27, 'Semáforo RL · Machine Learning · 2026')
        lienzo.drawRightString(A4[0] - 52, 27, str(documento.page))

    salida = fuente.with_suffix('.pdf')
    documento = SimpleDocTemplate(str(salida), pagesize=A4, rightMargin=52, leftMargin=52,
                                  topMargin=42, bottomMargin=46,
                                  title=lineas[0].lstrip('# '),
                                  author='Marcos Devincenzi; Facundo Díaz; Santiago Alba; Domingo Leiva')
    documento.build(contenido, onFirstPage=pie_de_pagina, onLaterPages=pie_de_pagina)
    print(salida)


if __name__ == '__main__':
    main()
