"""Render the selected submission technical Markdown as a compact PDF."""
import argparse
from html import escape
from pathlib import Path
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MONO = 'Courier'


def plain_ascii(text):
    for old,new in {'–':'-','—':'-','‑':'-','’':"'",'‘':"'",'“':'"','”':'"','\u00a0':' ','→':'->','≤':'<='}.items():
        text = text.replace(old,new)
    return text


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#175E86">\1</link>',text)
    text = re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text = re.sub(r'`([^`]+)`',lambda m:f'<font name="{MONO}" size="9">{m.group(1)}</font>',text)
    return text


def command_lines(lines,language):
    result = []
    for line in lines:
        if language == 'powershell' and len(line)>99:
            wrapped = textwrap.wrap(line,width=97,break_long_words=False,break_on_hyphens=False,
                subsequent_indent='  ',replace_whitespace=False)
            result += [x+' '+chr(96) if i<len(wrapped)-1 else x for i,x in enumerate(wrapped)]
        else:
            result += textwrap.wrap(line,width=99,break_long_words=True,break_on_hyphens=False) or ['']
    return '\n'.join(result)


def main():
    global MONO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--markdown-out',type=Path,required=True)
    args = parser.parse_args()
    font = Path('C:/Windows/Fonts/consola.ttf')
    if font.exists():
        pdfmetrics.registerFont(TTFont('ReportMono',str(font)))
        MONO = 'ReportMono'
    text = args.source.read_text(encoding='utf-8')
    start,end = 'An additional external transition-head grid','## Reproduction and limitations'
    assert text.count(start)==text.count(end)==1
    a,b = text.index(start),text.index(end)
    text = text[:a]+'''Additional external experiments were not deployed: a transition-head mixture
did not improve the criterion, and partial-HAD training on public Metrica games
1/2 failed the guards. The latter used 94,773 supported event-derived frames
(H 39,689; A 33,245; D 21,839), with unsupported spans unknown. Source: Metrica
Sports' public 2021-04-15 sample release, published for community analytics with
source acknowledgement requested. No Game 3 labels train the model. Its frozen,
previously opened partial-label diagnostic supports transfer on degraded tracking
but does not establish dense spell F1 or an independent result. Detailed rejected
experiments and source receipts are retained in the repository.

'''+text[b:]
    addition = '''## Additional completed checks

A 30-epoch expanded-geometry experiment reproduced mean external spell F1
0.681791, but its paired difference interval [-0.003650, 0.010709] spans zero.
It does not replace the selected model. A separate metadata-derived projection
repair changed 128 weights and 128 biases without gradient training; its three
fixed variants leave spell F1 unchanged, so zero repair is retained. Both studies
use external data only. Neither result is evidence of BAR-ATH full-match F1.

'''
    marker = '## Exact regeneration commands'
    assert text.count(marker)==1
    text = plain_ascii(text.replace(marker,addition+marker))
    args.markdown_out.parent.mkdir(parents=True,exist_ok=True)
    args.markdown_out.write_text(text,encoding='utf-8')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=19,leading=23,
        textColor=colors.HexColor('#15324A'),spaceAfter=12,keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportHeading',fontName='Helvetica-Bold',fontSize=11.5,leading=15,
        textColor=colors.HexColor('#15324A'),spaceBefore=10,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10,leading=13.8,
        spaceAfter=7,textColor=colors.HexColor('#202D37'),alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='ReportCode',fontName=MONO,fontSize=8.2,leading=10.5,
        borderWidth=.5,borderColor=colors.HexColor('#DAE2E8'),backColor=colors.HexColor('#F4F7F9'),
        borderPadding=7,spaceBefore=4,spaceAfter=9))
    styles.add(ParagraphStyle(name='ReportKicker',fontName='Helvetica-Bold',fontSize=8.5,leading=12,
        textColor=colors.HexColor('#526B7D'),spaceAfter=6))
    story = [Paragraph('MORE THAN A HACK 2027 / INITIAL CHALLENGE',styles['ReportKicker'])]
    paragraph,code,language = [],None,''

    def flush():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)),styles['ReportBody']))
            paragraph.clear()

    for line in text.splitlines():
        if line.startswith('```'):
            flush()
            if code is None:
                code,language = [],line[3:].strip()
            else:
                story.append(Preformatted(command_lines(code,language),styles['ReportCode']))
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if not line.strip():
            flush()
        elif line.startswith('# '):
            flush();story.append(Paragraph(inline(line[2:]),styles['ReportTitle']))
        elif line.startswith('## '):
            flush();story.append(Paragraph(inline(line[3:]),styles['ReportHeading']))
        elif line.startswith('- '):
            flush();story.append(Paragraph('- '+inline(line[2:]),styles['ReportBody']))
        else:
            paragraph.append(line)
    flush()
    if code is not None:
        raise ValueError('Unclosed code block')
    document = SimpleDocTemplate(str(args.out),pagesize=A4,rightMargin=45,leftMargin=45,
        topMargin=39,bottomMargin=40,title='CoreOrigin - Possession reconstruction',author='CoreOrigin')

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#DAE2E8'));canvas.line(45,31,A4[0]-45,31)
        canvas.setFillColor(colors.HexColor('#607587'));canvas.setFont('Helvetica',8)
        canvas.drawString(45,20,'CoreOrigin | Frozen continuity-v4 | 14 November 2026')
        canvas.drawRightString(A4[0]-45,20,str(doc.page))
        canvas.restoreState()

    document.build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(args.out.resolve()))


if __name__ == '__main__':
    main()
