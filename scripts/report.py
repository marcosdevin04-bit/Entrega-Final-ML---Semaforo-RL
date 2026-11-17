"""Convierte el informe Markdown a PDF. Requiere requirements-report.txt."""
from pathlib import Path
import argparse
import re
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from src.experiment import ROOT

def inline(text):
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    return text

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'report'/'informe.md')
    source=parser.parse_args().source.resolve()
    styles=getSampleStyleSheet()
    styles['Title'].fontName='Helvetica-Bold'
    styles['Title'].fontSize=22
    styles['Title'].leading=27
    styles['Title'].textColor=colors.black
    styles['Title'].alignment=0
    styles['Title'].spaceAfter=20
    for name in ['Heading1','Heading2','Heading3']:
        styles[name].textColor=colors.black
        styles[name].spaceBefore=14
        styles[name].spaceAfter=8
    styles['Heading1'].fontSize=17
    styles['Heading1'].leading=21
    styles['Heading2'].fontSize=12
    styles['Heading2'].leading=15
    styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=10,
                             leading=14,spaceAfter=9,alignment=TA_JUSTIFY,
                             splitLongWords=True))
    styles.add(ParagraphStyle(name='CellReport',fontName='Helvetica',fontSize=8,
                             leading=10,spaceAfter=0))
    styles.add(ParagraphStyle(name='CaptionReport',fontName='Helvetica-Oblique',fontSize=8,
                             leading=11,spaceAfter=12,textColor=colors.HexColor('#53616b')))
    story=[]
    lines=source.read_text(encoding='utf-8').splitlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1
            continue
        if line=='<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['Title']))
        elif line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),styles['Heading1']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['Heading2']))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',c) for c in cells):
                    rows.append(cells)
                i+=1
            n=len(rows[0])
            widths=[(.23 if n==5 else .32)*491]+[(491-(.23 if n==5 else .32)*491)/(n-1)]*(n-1)
            content=[[Paragraph(inline(c),styles['CellReport']) for c in row] for row in rows]
            table=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfebf1')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f6f7')]),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story.extend([table,Spacer(1,12)])
            continue
        elif line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            image=Image(str(source.parent/m.group(2)))
            ratio=image.imageHeight/image.imageWidth
            image.drawWidth=491
            image.drawHeight=491*ratio
            story.append(image)
            story.append(Paragraph(inline(m.group(1))+'. Fuente: experimentos del proyecto.',styles['CaptionReport']))
        else:
            paragraph=[line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','![','<!--')):
                i+=1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(' '.join(paragraph)),styles['BodyReport']))
        i+=1
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#5b6770'))
        canvas.drawString(52,27,'Semáforo RL · Machine Learning · 2026')
        canvas.drawRightString(A4[0]-52,27,str(doc.page))
    out=source.with_suffix('.pdf')
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=52,leftMargin=52,
                         topMargin=42,bottomMargin=46,
                         title=lines[0].lstrip('# '),
                         author='Marcos Devincenzi; Facundo Díaz; Santiago Alba; Domingo Leiva')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__=='__main__':
    main()
