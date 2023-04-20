import webdataset as wds
import io
from clize import run
from TexSoup import TexSoup, read
from TexSoup.data import TexMathModeEnv, TexEnv, TexCmd, TexText, TexGroup
import os
import tarfile
import fsspec

def parse_arxiv_shard_tar(path):
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
            yield from parse_arxiv_paper_tar_gz(fd_gz)
            break

def parse_arxiv_paper_tar_gz(fd):
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
        ext = os.path.splitext(member.name)[1].replace(".", "")
        data = (tar.extractfile(member).read())
        yield {ext: data, "txt": caption}
    tar.close()

def everything_s(tex_tree):
    """
    Accepts a list of Union[TexNode,Token] and returns a nested list
    of strings of the entire source document.
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
    for node in s.children:
        yield node
        yield from nodes(node)


def extract_figure_caption_pairs(data, filelist):
    try:
        soup = TexSoup(data, tolerance=1)
    except Exception as ex:
        #"$" env expecting $. Reached end of file
        print(ex)
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
                            print(arg.string)
                            fname = str(arg.string)
                            break
                if fname:
                    break
            if not fname:
                continue
            parent_text = str(parent)
            pairs.append((fname, everything_s(node) ))
    return pairs

sink = wds.TarWriter("out.tar")
i = 0
for datum in parse_arxiv_shard_tar("arXiv_src_2203_094.tar"):
    datum['__key__'] = str(i)
    sink.write(datum)
    i += 1
sink.close()
#print(extract_figure_caption_pairs(open("2203/eccv2022submission.tex").read(), ["figures/num_rounds_local_epochs.png"]))