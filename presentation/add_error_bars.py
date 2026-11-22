"""Add 95% confidence-interval bars to the deck's charts after pptxgenjs writes it.

Usage: python presentation/add_error_bars.py <deck.pptx>
Reads <deck.pptx>.errorbars.json (chart number -> [minus values, plus values]) written by
build_presentation.js. Bar charts get vertical bars; scatter (forest) charts get horizontal
bars on the x values, and their vertical axis keeps its line (the zero line) without numbers.
"""
import io
import json
import re
import sys
import zipfile

src = sys.argv[1]
bars = json.load(open(src + ".errorbars.json"))


def lit(vals):
    pts = "".join(f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(vals))
    return f'<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="{len(vals)}"/>{pts}</c:numLit>'


def hide_vertical_axis_numbers(axis):
    ax = axis.group(0)
    if '<c:axPos val="l"/>' in ax:
        ax = re.sub(r'<c:tickLblPos val="\w+"/>', '<c:tickLblPos val="none"/>', ax)
    return ax


out = io.BytesIO()
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        d = zi.read(it.filename)
        m = re.fullmatch(r"ppt/charts/chart(\d+)\.xml", it.filename)
        if m and m.group(1) in bars:
            minus, plus = bars[m.group(1)]
            x = d.decode("utf-8")
            assert x.count("<c:ser>") == 1 and "<c:errBars>" not in x
            scatter = "<c:scatterChart>" in x
            eb = ('<c:errBars>' + ('<c:errDir val="x"/>' if scatter else '') +
                  '<c:errBarType val="both"/><c:errValType val="cust"/><c:noEndCap val="0"/>'
                  f'<c:plus>{lit(plus)}</c:plus><c:minus>{lit(minus)}</c:minus>'
                  '<c:spPr><a:ln w="22225"><a:solidFill><a:srgbClr val="182B40"/></a:solidFill></a:ln></c:spPr></c:errBars>')
            i = x.index("<c:xVal>" if scatter else "<c:cat>")
            x = x[:i] + eb + x[i:]
            if scatter:
                x = re.sub(r"<c:valAx>.*?</c:valAx>", hide_vertical_axis_numbers, x, flags=re.S)
                assert x.count('<c:tickLblPos val="none"/>') == 1
            print(it.filename, "error bars added")
            d = x.encode("utf-8")
        zo.writestr(it, d)
open(src, "wb").write(out.getvalue())
