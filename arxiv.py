from pathlib import Path
from collections import defaultdict
import re
import time
import torch
import webdataset as wds
import io
from PIL import Image
from clize import run
import sys
from TexSoup import TexSoup, read
from TexSoup.data import TexMathModeEnv, TexEnv, TexCmd, TexText, TexGroup, BracketGroup
import os
import tarfile
import fsspec
from pdf2image import convert_from_bytes
from latex_render import latex2image, latex2imagev2
import objgraph

from pympler.tracker import SummaryTracker
from pympler import summary, muppy  

from pympler.classtracker import ClassTracker
from pympler import web

from mem_top import mem_top

class ArxivFigureCaptions(torch.utils.data.IterableDataset):

    def __init__(self, filelist, start=None, end=None):
        super().__init__()
        if start is None and end is None:
            start = 0
            end = len(filelist)
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.filelist = filelist

    def __iter__(self):
        for fs in self.filelist[self.start:self.end]:
            try:
                yield from parse_arxiv_shard_tar(fs, extract=["figure_captions"])
            except Exception as ex:
                print(ex)

    def __len__(self):
        return len(self.filelist[self.start:self.end])

class ArxivEquations(torch.utils.data.IterableDataset):

    def __init__(self, filelist, start=None, end=None):
        super().__init__()
        if start is None and end is None:
            start = 0
            end = len(filelist)
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
        self.filelist = filelist

    def __iter__(self):
        for fs in self.filelist[self.start:self.end]:
            try:
                yield from parse_arxiv_shard_tar(fs, extract=["math"])
            except Exception as ex:
                print(ex)
    def __len__(self):
        return len(self.filelist[self.start:self.end])

#tracker = SummaryTracker()

def parse_arxiv_shard_tar(path, extract=["figure_captions"]):
    """
    arxiv dump is divided into multiple shards (tar format),
    and in each  shard there are multiple papers
    """

    of = None
    try:
        of = fsspec.open(path)        
        with of as fd:
            data = fd.read()
        of.close()
    except Exception as ex:
        print(ex)
        of.close()
        return
    fd = io.BytesIO(data)
    tar = tarfile.open(fileobj=fd)
    #import gc
    members = tar.getmembers()
    print(f"Nb of papers from {path}: {len(members)}")
    for i, member in enumerate(members):

        if member.name.endswith(".gz"):
            #print(i, member.name)
            f = tar.extractfile(member)
            data = f.read()
            f.close()
            fd_gz = io.BytesIO(data)
            yield from parse_arxiv_paper_tar_gz(fd_gz, member.name, extract=extract)
            fd_gz.close()
            del fd_gz

    tar.close()
    fd.close()
    del fd
    print(f"End of {path}")

def parse_arxiv_shard_tar_to_list(path, extract):
    return list(parse_arxiv_shard_tar(path, extract=extract))


def clean(data):
    lines = data.split("\n")
    lines = [l for l in lines if not l.strip().startswith("%")]
    return "\n".join(lines)


def extract_figure_caption_pairs(data, filelist):
    """
    extract all figure-caption pairs by using a) latex file content and b) filelist of an existing paper (to extract image contents)
    
    """
    subfolders = ['']
    graphicspath = re.findall(r"\\graphicspath\{(\{.*?\})\}", data, flags=re.DOTALL)
    for g in graphicspath:
        g = g.replace('{', '')
        g = g.replace('}', '')
        subfolders.append(Path(g))
    figs = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure(.*?)\}", data, flags=re.DOTALL)
    filelist_no_ext = [Path(os.path.splitext(f)[0]) for f in filelist]
    filelist_orig = filelist
    filelist = [Path(fs) for fs in filelist]
    # find start figure tag
    figs_enter = re.finditer(r"\\begin\{figure\*?\}", data)
    nb = 0
    for fig_enter in figs_enter:
        # find end figure tag
        fig_close = data[fig_enter.start():fig_enter.end()].replace("\\begin", r"\\end").replace("*", r"\*")
        text_from_fig_enter = data[fig_enter.end():]
        match = re.search(fig_close, text_from_fig_enter, flags=re.DOTALL)
        if not match:
            continue
        # inside figure content
        fig = text_from_fig_enter[:match.start()]
        #if len(fig) > 2000:
        #    print("SKIP")
        #    continue
        # do TexSoup on it to parse
        try:
            #print(len(fig))
            #soup = TexSoup("\\begin{figure}" + fig + "\\end{figure}")
            soup = TexSoup(fig)
        except Exception as ex:
            print(ex)
            continue
            
        def get_filename(f):
            for sf in subfolders:
                s = sf / Path(f)
                if s in (filelist):
                    i = filelist.index(s)
                    fname = filelist_orig[i]
                    return fname
                elif s in filelist_no_ext:
                    fname = filelist_orig[filelist_no_ext.index(s)]
                    return fname
        fnames = []
        caption = None
        subcaptions = []

        # global caption
        for n in soup.children:
            if n.name == "caption":
                caption = node_to_string(n)

        # now, try to find subfigures or main figure
        #for n in soup.children:
        for n in nodes(soup):
            if n.name not in ("subfigure", "subfloat", "includegraphics"):
                continue
            local_caption = None
            # search caption
            parent = n.parent
            
            parents = [parent.name]
            p = parent
            while p:
                parents.append(p.name)
                p = p.parent
            if n.name == "subfloat":
                local_caption = None
                for arg in n.args:
                    if type(arg) == BracketGroup:
                        local_caption = arg.string
            elif n.name == "subfigure":
                for nn in n.children:
                    if nn.name == "caption":
                        local_caption = node_to_string(nn)
            if n.name in ("subfloat", "subfigure"):
                for c in nodes(n):
                    if c.name == "includegraphics":
                        for arg in c.args:
                            fname = get_filename(arg.string)
                            if fname is not None:
                                fnames.append(fname)
                                subcaptions.append(local_caption)
            elif n.name == "includegraphics" and all(par not in ("subfigure", "subfloat") for par in parents):
                # main figure
                for arg in n.args:
                    fname = get_filename(arg.string)
                    if fname is not None:
                        fnames.append(fname)
                        subcaptions.append(None)
        #if caption is None:
        #    if any(sb is None for sb in subcaptions):
        #        print(subcaptions)
        #        sys.exit(0)
        #        continue
        #if any(sb is None for sb in subcaptions):
        full_caption = ""
        if any(sc for sc in subcaptions):
            full_caption += "".join(f"<sf{i+1}>" + (sc if sc is not None else "") + f"</sf{i+1}>" for i, sc in enumerate(subcaptions))
        if caption is not None:
            full_caption += f"<f>{caption}</f>"
            
        #yield fnames, full_caption

        #option2: all subfigures as individual examples
        if caption is not None:
            global_caption = f"<f>{caption}</f>"
        else:
            global_caption = ""
        for i, (name, sc) in enumerate(zip(fnames, subcaptions)):
            full_caption = f"<sf{i+1}>" + (sc if sc is not None else "") + f"</sf{i+1}>"
            if global_caption:
                full_caption += global_caption
            yield [name], full_caption

def parse_arxiv_paper_tar_gz(fd, url, extract=("figure_captions",)):
    # process a single paper (usually a .tar.gz file) from a file description
    t0 = time.time()
    try:
        tar = tarfile.open(fileobj=fd, mode='r:gz')
    except Exception as ex: 
        #print(ex)
        return
    filelist = []
    latex_files = []
    members = {}
    for member in tar.getmembers():
        filelist.append(member.name)
        if member.name.endswith(".tex"):
            try:
                data = (tar.extractfile(member).read()).decode()
            except Exception as ex:
                #print(ex)
                continue
            latex_files.append(data)
        members[member.name] = member
    pairs = []
    nb = 0
    if "math" in extract:
        nb_actual_imgs  = 0
        for latex in latex_files:
            for eq in extract_math(latex):
                img = latex2imagev2(eq)
                if img is not None:
                    yield {"caption": eq, "img_content": img, "url": url, "img_path": "img.png"}
                nb += 1
    if "figure_captions" in extract:
        nb_actual_imgs  = 0
        latexs = ""
        for latex in latex_files:
            latexs += latex
            t0 = time.time()
            latex = clean(latex)
            nb_actual_imgs += len(re.findall(r"\\begin\{figure", latex))
            pairs.extend(list(extract_figure_caption_pairs(latex, filelist)))
        #for img_path, caption in pairs:
        for img_paths, caption in pairs:
            imgs = []
            for img_path in img_paths:
                if img_path not in members:
                    continue
                member = members[img_path]
                name, ext = os.path.splitext(member.name)
                try:
                    data = (tar.extractfile(member).read())
                except Exception as ex:
                    print(ex)
                    continue
                
                if ext == ".pdf":
                    t0 = time.time()
                    try:
                        images_pil = convert_from_bytes(data)
                    except Exception:
                        continue
                    if len(images_pil) != 1:
                        continue
                    image_pil = images_pil[0]
                    fd_image = io.BytesIO()
                    image_pil.save(fd_image, format='PNG')
                    data = fd_image.getvalue()
                    full_name = name + ".png"
                else:
                    full_name = member.name
                    try:
                        image_pil = Image.open(io.BytesIO(data))
                    except Exception:
                        continue
                if os.path.splitext(full_name)[1] == '':
                    continue
                imgs.append((image_pil, data, full_name))
            
            if len(imgs) == 1:
                img, data, full_name = imgs[0]
            elif len(imgs) > 1:
                
                c = caption.lower()
                #left_or_right = "(left)" in c or "(right)" in c or r"\textbf{left}" in c or r"\textbf(right)" in c
                #top_or_bottom = "(top)" in c or "(bottom)" in c or  r"\textbf{top}" in c or r"\textbf(bottom)" in c
                left_or_right = "left" in c or "right" in c
                top_or_bottom = "top" in c or "bottom" in c

                if left_or_right and top_or_bottom:
                    horiz = True
                elif left_or_right and not top_or_bottom:
                    horiz = True
                elif not left_or_right and top_or_bottom:
                    horiz = False
                else:
                    horiz = True
                    
                    total_width = sum(img.width for img, data, fn in imgs)
                    max_height = max(img.height for img, data, fn in imgs)
                    Wa = total_width
                    Ha = max_height

                    total_height = sum(img.height for img, data, fn in imgs)
                    max_width = max(img.width for img, data, fn in imgs)
                    Wb = max_width
                    Hb = total_height

                    if (abs(Wa/Ha-1))  <  (abs(Wb/Hb-1)):
                        horiz = True
                    else:
                        horiz = False
                if horiz:
                    total_width = sum(img.width for img, data, fn in imgs)
                    max_height = max(img.height for img, data, fn in imgs)
                    new_im = Image.new('RGB', (total_width, max_height),  (255,255,255,255))
                    x_offset = 0
                    for img, data, fn in imgs:
                        img = img.resize((img.width, max_height))
                        new_im.paste(img, (x_offset,0))
                        x_offset += img.size[0]
                else:
                    total_height = sum(img.height for img, data, fn in imgs)
                    max_width = max(img.width for img, data, fn in imgs)
                    new_im = Image.new('RGB', (max_width, total_height),  (255,255,255,255))
                    y_offset = 0
                    for img, data, fn in imgs:
                        img = img.resize((max_width, img.height))
                        new_im.paste(img, (0, y_offset))
                        y_offset += img.size[1]
                full_name = ""
                for img, data, fn in imgs:
                    full_name += fn+"_"
                full_name += ".png"
                fd_image = io.BytesIO()
                new_im.save(fd_image, format='PNG')
                data = fd_image.getvalue()
            else:
                data = None

            if data is not None:
                yield {"img_content": data, "caption": caption, "img_path": full_name, "url": url}
                nb += 1
    tar.close()
    print(f"Finished {url} in {time.time() - t0} in {os.getpid()} with {nb} pairs, there are {nb_actual_imgs} figures")
    
    if nb_actual_imgs != nb:
        with open("debug.tex", "w") as fd:
            fd.write(latexs)
        #sys.exit(0)

    
def node_to_string(tex_tree):
    """
    Return the string repr of a Tex node
    """
    result = []
    for tex_code in tex_tree:
        if isinstance(tex_code, TexEnv):
            result.append(tex_code.begin + str(tex_code.args), node_to_string(tex_code.all), tex_code.end)
        elif isinstance(tex_code, TexCmd):
            result.append("\\" + tex_code.name + str(tex_code.args))
        elif isinstance(tex_code,TexText):
            result.append(tex_code.text)
        elif isinstance(tex_code,TexGroup):
            result.append("{", node_to_string(TexSoup(tex_code.value).expr.all), "}")
        else:
            result.append(str(tex_code))

    return ''.join(result) if result else ''

def extract_math(latex_code):
    """
    extract all math equations from a Tex file
    """
    values = re.findall(r"\$(.*?)\$", latex_code)
    eqs = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", latex_code)
    #print(len(eqs))
    values = values + eqs
    eqs = re.findall(r"\\begin\{math\}(.*?)\\end\{math\}", latex_code)
    #print(len(eqs))
    values = values + eqs
    return values
    """
    t0 = time.time()
    soup = TexSoup(latex_code, tolerance=1)
    print(time.time() - t0)
    values = []
    for node in nodes(soup):
        if type(node.expr) == TexMathModeEnv:
            value = node.expr.string
        elif node.name == "math":
            value = ''.join([n for n in node.text])
        else:
            continue
        values.append(value)
    return values
    """

def nodes(s):
    """
    get all nodes and subnodes of a root node
    """
    for node in s.children:
        yield node
        yield from nodes(node)




if __name__ == "__main__":
    
    sink = wds.TarWriter("out.tar")
    i = 0
    for datum in parse_arxiv_shard_tar("arXiv_src_2203_094.tar"):
        datum['__key__'] = str(i)
        sink.write(datum)
        i += 1
    sink.close()

    #data = open("debug.tex").read()
    #soup = TexSoup(data, tolerance=1)
    #print(soup)
