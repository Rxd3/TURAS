import json, sys, zipfile, io, re
src, bars = sys.argv[1], json.load(open(sys.argv[1] + ".errorbars.json"))
def lit(vals):
    pts = "".join(f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(vals))
    return f'<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="{len(vals)}"/>{pts}</c:numLit>'
out = io.BytesIO()
with zipfile.ZipFile(src) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        d = zi.read(it.filename)
        m = re.fullmatch(r"ppt/charts/chart(\d+)\.xml", it.filename)
        if m and m.group(1) in bars:
            minus, plus = bars[m.group(1)]
            x = d.decode("utf-8")
            assert x.count("<c:ser>") == 1 and "<c:errBars>" not in x
            eb = ('<c:errBars><c:errBarType val="both"/><c:errValType val="cust"/><c:noEndCap val="0"/>'
                  f'<c:plus>{lit(plus)}</c:plus><c:minus>{lit(minus)}</c:minus>'
                  '<c:spPr><a:ln w="15875"><a:solidFill><a:srgbClr val="182B40"/></a:solidFill></a:ln></c:spPr></c:errBars>')
            i = x.index("<c:cat>")
            x = x[:i] + eb + x[i:]
            print(it.filename, "error bars added;", re.search(r"<c:v>([^<]+)</c:v>", x).group(1))
            d = x.encode("utf-8")
        zo.writestr(it, d)
open(src, "wb").write(out.getvalue())
