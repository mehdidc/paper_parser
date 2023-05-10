from joblib import Parallel, delayed
import random
import time
import os
from PIL import Image
from subprocess import call
import tempfile
import uuid
import gc
import io

tpl = r"""
\documentclass[crop]{standalone}
\begin{document}
${MATH}$
\end{document}
"""

from matplotlib import mathtext, font_manager
import matplotlib as mpl
mpl.rcParams['savefig.transparent'] = True
#texFont = font_manager.FontProperties(size=30, fname="./OpenSans-Medium.ttf")
texFont = font_manager.FontProperties(size=30, family='serif', math_fontfamily='cm')

def latex2imagev2(math):
    #gc.collect()
    #return Image.new(size=(1, 1), mode='RGB')
    fd = io.BytesIO()
    try:
        mathtext.math_to_image('$' + math + '$', fd, prop=texFont, dpi=100, format='png')
    except Exception:
        fd.close()
        return None
        #return latex2image(math)
        #return Image.new(size=(1, 1), mode='RGB')
    data = fd.getvalue()
    fd.close()
    #gc.collect()
    del fd
    return data

def latex2image(math):
    name = "tmp" + str(uuid.uuid4())
    tex_path = name + ".tex"
    png_path = name + ".png"
    ps_path = (name) + ".ps"
    dvi_path = (name) + ".dvi"
    pdf_path = (name) + ".pdf"
    aux_path = (name) + ".aux"
    log_path = (name) + ".log"
    data = tpl.replace("{MATH}", math)
    with open(tex_path, "w") as fd:
        fd.write(data)
    call(f"pdflatex -interaction=nonstopmode --job-name={name} {tex_path} 2>/dev/null 1>/dev/null;convert -density 256  {pdf_path} {png_path} 2>/dev/null 1>/dev/null", shell=True)
    #call(f"pdflatex -interaction=nonstopmode --job-name={name} {tex_path} 2>/dev/null 1>/dev/null;pdf2ps {pdf_path} {ps_path} 2>/dev/null 1>/dev/null", shell=True)
    #call(f"latex -interaction=nonstopmode --job-name={name} {tex_path} 2>/dev/null 1>/dev/null;dvips -o {ps_path} {dvi_path}", shell=True)
    #call(f"latex -interaction=nonstopmode --job-name={name} {tex_path} 2>/dev/null 1>/dev/null", shell=True)
    img = None
    if os.path.exists(ps_path):
        img = Image.open(ps_path)
        os.remove(ps_path)
    if os.path.exists(png_path):
        img = Image.open(png_path)
        os.remove(png_path)
    if os.path.exists(tex_path):
        os.remove(tex_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(aux_path):
        os.remove(aux_path)
    if os.path.exists(log_path):
        os.remove(log_path)
    if os.path.exists(dvi_path):
        os.remove(dvi_path)
    return img

if __name__ == "__main__":
    N = 1000
    t0 = time.time()
    Parallel(n_jobs=16, backend="threading")(( delayed(latex2imagev2)("C=1 + 10 + 10 + 20 + {i}") for i in range(N)))
    dt = time.time() - t0
    print(dt, N/dt)


