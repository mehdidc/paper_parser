import os
from PIL import Image
from subprocess import call
import tempfile

tpl = r"""
\documentclass[crop]{standalone}
\begin{document}
${MATH}$
\end{document}
"""

def latex2png(math):
    tf = tempfile.NamedTemporaryFile(mode="w")
    tex_path = tf.name + ".tex"
    png_path = tf.name + ".png"
    pdf_path = os.path.basename(tf.name) + ".pdf"
    aux_path = os.path.basename(tf.name) + ".aux"
    log_path = os.path.basename(tf.name) + ".log"
    data = tpl.replace("{MATH}", math)
    with open(tex_path, "w") as fd:
        fd.write(data)
    print(tex_path)
    call(f"rm fig.png;pdflatex --job-name={tf.name} {tex_path};convert -density 512  {pdf_path} {png_path}", shell=True)
    tf.close()
    img = None
    if os.path.exists(png_path):
        img = Image.open(png_path)
    if os.path.exists(tex_path):
        os.remove(tex_path)
    if os.path.exists(png_path):
        os.remove(png_path)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if os.path.exists(aux_path):
        os.remove(aux_path)
    if os.path.exists(log_path):
        os.remove(log_path)
    return img

if __name__ == "__main__":
    img = latex2png(r"\vec{C}")
    img.save("x.png")
