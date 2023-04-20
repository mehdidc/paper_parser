import torch
import webdataset as wds
import io
from clize import run
from TexSoup import TexSoup, read
from TexSoup.data import TexMathModeEnv, TexEnv, TexCmd, TexText, TexGroup
import os
import tarfile
import fsspec
from pdf2image import convert_from_bytes


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
                yield from parse_arxiv_shard_tar(fs)
            except Exception as ex:
                print(ex)



def parse_arxiv_shard_tar(path):
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
            print(member.name)
            data = tar.extractfile(member).read()
            fd_gz = io.BytesIO(data)
            # process each paper of the shard
            yield from parse_arxiv_paper_tar_gz(fd_gz, member.name)

def extract_figure_caption_pairs(data, filelist):
    """
    extract all figure-caption pairs by using a) latex file content and b) filelist of an existing paper (to extract image contents)
    """
    try:
        soup = TexSoup(data, tolerance=1)
    except Exception as ex:
        #"$" env expecting $. Reached end of file
        print(ex)
        print(data)
        fd = open("debug.tex", "w")
        fd.write(data)
        fd.close()
        sys.exit(0)
        return []
    pairs = []
    for node in nodes(soup):
        if node.name == "caption" and node.parent.name == "figure":
            parent = node.parent
            fname = None
            for n in nodes(parent):
                if n.name == "epsfbox":
                    fname = everything_s(n)
                    #print(n, fname)
                elif n.name == "includegraphics":
                    for arg in n.args:
                        if arg.string in filelist:
                            fname = str(arg.string)
                            break
                else:
                    # what else?
                    pass
                if fname:
                    break
            if not fname:
                continue
            parent_text = str(parent)
            pairs.append((fname, everything_s(node) ))
    return pairs

def parse_arxiv_paper_tar_gz(fd, url):
    # process a single paper (usually a .tar.gz file) from a file description
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
    for latex in latex_files:
        pairs.extend(extract_figure_caption_pairs(latex, filelist))
    for img_path, caption in pairs:
        member = members[img_path]
        name, ext = os.path.splitext(member.name)
        data = (tar.extractfile(member).read())
        if ext == ".pdf":
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
        print(full_name)
        yield {"img_content": data, "caption": caption, "img_path": full_name, "url": url}
    tar.close()

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
    soup = TexSoup(latex_code, tolerance=1)
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