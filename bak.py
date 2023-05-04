import re
import time
import torch
import webdataset as wds
import io
from clize import run
import sys
from TexSoup import TexSoup, read
from TexSoup.data import TexMathModeEnv, TexEnv, TexCmd, TexText, TexGroup
import os
import tarfile
import fsspec
from pdf2image import convert_from_bytes

import pprofile

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


def parse_arxiv_shard_tar(path, extract=["figure_captions"]):
    """
    arxiv dump is divided into multiple shards (tar format),
    and in each  shard there are multiple papers
    """
    of = fsspec.open(path)
    with of as fd:
        data = fd.read()
    of.close()
    fd = io.BytesIO(data)
    tar = tarfile.open(fileobj=fd)

    for member in tar.getmembers():
        if member.name.endswith(".gz"):
            #print(member.name)
            data = tar.extractfile(member).read()
            fd_gz = io.BytesIO(data)
            #print(f"Extraction of {member.name} done in {dl} seconds")
            # process each paper of the shard
            yield from parse_arxiv_paper_tar_gz(fd_gz, member.name, extract=extract)

def extract_figure_caption_pairs(data, filelist):
    """
    extract all figure-caption pairs by using a) latex file content and b) filelist of an existing paper (to extract image contents)
    
    """
    figs = re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure(.*?)\}", data, flags=re.DOTALL)
    filelist_no_ext = [os.path.splitext(f)[0] for f in filelist]
    for fig in figs:
        caption_global = None
        fname_global = None
        subfigs = []
        try:
            soup = TexSoup("\\begin{figure}" + fig[0] + "\\end{figure}")
        except Exception:
            with open("debug.txt", "w") as fd:
                fd.write(fig[0])
            raise
        for node in soup.children[0].children:
            if node.name == "subfigure":
                fname = None
                caption = None
                for subfig_node in node.children:
                    if subfig_node.name == "caption":
                        caption = everything_s(subfig_node)
                    elif subfig_node.name == "includegraphics":
                        for arg in subfig_node.args:
                            if arg.string in filelist or arg.string in filelist_no_ext:
                                fname = str(arg.string)
                    elif subfig_node.name == "epsfbox":
                        fname = everything_s(subfig_node)
                if fname:
                    subfigs.append((fname, caption))
            elif node.name == "caption":
                caption_global = everything_s(node)
            elif node.name == "includegraphics":
                 for arg in node.args:
                    if arg.string in filelist or arg.string in filelist_no_ext:
                        fname_global = str(arg.string)
            elif node.name == "epsfbox":
                fname = everything_s(node)
        #print(len(subfigs), fname_global, caption_global)
        for subfig in subfigs:
            fname, caption = subfig
            if caption is None and caption_global:
                caption = caption_global
            if fname and caption:
                yield fname, caption

        if fname_global and caption_global:
            yield fname_global, caption_global

def parse_arxiv_paper_tar_gz(fd, url, extract=("figure_captions",)):
    # process a single paper (usually a .tar.gz file) from a file description
    t0 = time.time()
    try:
        tar = tarfile.open(fileobj=fd, mode='r:gz')
    except Exception as ex:
        return
    
    filelist = []
    latex_files = []
    members = {}
    for member in tar.getmembers():
        filelist.append(member.name)
        if member.name.endswith(".tex"):
            data = (tar.extractfile(member).read()).decode()
            latex_files.append(data)
        members[member.name] = member
    pairs = []
    nb = 0
    if "math" in extract:
        for latex in latex_files:
            for eq in extract_math(latex):
                yield {"equation": eq}
    
    if "figure_captions" in extract:
        nb_actual_imgs  = 0
        for latex in latex_files:
            t0 = time.time()
            nb_actual_imgs += len(re.findall("includegraphics", latex))
            pairs.extend(list(extract_figure_caption_pairs(latex, filelist)))
        for img_path, caption in pairs:
            member = members[img_path]
            name, ext = os.path.splitext(member.name)
            data = (tar.extractfile(member).read())
            
            if ext == ".pdf" and False:
                t0 = time.time()
                images_pil = convert_from_bytes(data)
                if len(images_pil) != 1:
                    continue
                image_pil = images_pil[0]
                fd_image = io.BytesIO()
                image_pil.save(fd_image, format='PNG')
                data = fd_image.getvalue()
                full_name = name + ".png"
            else:
                full_name = member.name
            #print(full_name)
            yield {"img_content": data, "caption": caption, "img_path": full_name, "url": url}
            nb += 1
    tar.close()
    print(f"Finished {url} in {time.time() - t0} in {os.getpid()} with {nb} pairs, but there are {nb_actual_imgs} includegraphics")

def everything_s(tex_tree):
    """
    Return the string repr of a Tex node
    """
    result = []
    for tex_code in tex_tree:
        if isinstance(tex_code, TexEnv):
            result.append(tex_code.begin + str(tex_code.args), everything(tex_code.all), tex_code.end)
        elif isinstance(tex_code, TexCmd):
            result.append("\\" + tex_code.name + str(tex_code.args))
        elif isinstance(tex_code,TexText):
            result.append(tex_code.text)
        elif isinstance(tex_code,TexGroup):
            result.append("{", everything(TexSoup(tex_code.value).expr.all), "}")
        else:
            result.append(str(tex_code))

    return ''.join(result) if result else ''

def extract_math(latex_code):
    """
    extract all math equations from a Tex file
    """
    values = re.findall(r"\$(.*?)\$", latex_code)
    eqs = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", latex_code)
    print(len(eqs))
    values = values + eqs
    eqs = re.findall(r"\\begin\{math\}(.*?)\\end\{math\}", latex_code)
    print(len(eqs))
    values = values + eqs
    return values


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

def nodes(s):
    """
    get all nodes and subnodes of a root node
    """
    for node in s.children:
        yield node
        yield from nodes(node)




if __name__ == "__main__":
    """
    sink = wds.TarWriter("out.tar")
    i = 0
    for datum in parse_arxiv_shard_tar("arXiv_src_2203_094.tar"):
        datum['__key__'] = str(i)
        sink.write(datum)
        i += 1
    sink.close()
    run([extract_figure_caption_pairs])
    """

    data = open("debug.tex").read()
    soup = TexSoup(data, tolerance=1)
    print(soup)