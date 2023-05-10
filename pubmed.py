import tarfile
import math
import torch
import time
import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from clize import run
import random
import os
import zipfile
import pubmed_parser as pp
import fsspec

from writer import TarWriter
import io
from joblib import Parallel, delayed


def parse_pubmed_to_list(path):
    return list(parse_pubmed(path))


def parse_pubmed(path):
    of = fsspec.open(path)
    with of as fd:
        data = fd.read()
    of.close()
    fd = io.BytesIO(data)
    t0 = time.time()
    tar = tarfile.open(fileobj=fd)
    #import gc
    members = tar.getmembers()
    member_by_name_jpg = {os.path.basename(m.name).replace('.jpg', ''): m for m in members if m.name.endswith("jpg")}
    member_by_name_png = {os.path.basename(m.name).replace('.png', ''): m for m in members if m.name.endswith("png")}
    for f in members:
        #print(f.name)
        if not f.name.endswith(".nxml"):
            continue
        mfd = tar.extractfile(f)
        xml_content = mfd.read()
        mfd.close()
        try:
            dicts_out = pp.parse_pubmed_caption(xml_content)
        except AttributeError:
            continue
        except ValueError:
            continue
        except Exception:
            continue
        if not dicts_out:
            continue
        for fig in dicts_out:
            caption = fig['fig_caption']
            graphic_ref = fig['graphic_ref']
            if graphic_ref is None:
                continue
            if caption is None:
                continue
            img_path  = graphic_ref
            if img_path in member_by_name_jpg:
                member = member_by_name_jpg[img_path]
                img_path = os.path.basename(member.name)
            elif img_path in member_by_name_png:
                member = member_by_name_png[img_path]
                img_path = os.path.basename(member.name)
            else:
                continue
            try:
                mfd = tar.extractfile(member)
                img_content = mfd.read()
                mfd.close()
            except Exception:
                continue
            datum = {"url": path, "caption": caption, "img_content": img_content, "img_path": img_path}
            yield datum 


class PubMedIterableDataset(torch.utils.data.IterableDataset):
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
        #print(self.start, self.end)
        for fs in self.filelist[self.start:self.end]:
            try:
                yield from parse_pubmed(fs)
            except Exception as ex:
                print(ex)

