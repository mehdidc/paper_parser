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


def parse_meca_to_list(path):
    return list(parse_meca(path))

def parse_meca(path):
    output = []
    of = fsspec.open(path)
    with of as fd:
        data = fd.read()
    of.close()
    fd = io.BytesIO(data)
    t0 = time.time()
    with zipfile.ZipFile(fd, mode='r') as zip_file:
        for f in zip_file.filelist:
            if not f.filename.startswith("content/"):
                continue
            if not f.filename.endswith(".xml"):
                continue
            with zip_file.open(f.filename) as xml_file:
                try:
                    dicts_out = pp.parse_pubmed_caption(xml_file)
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
                img_path  = "content/" + graphic_ref
                try:
                    with zip_file.open(img_path) as img_file:
                        img_content = img_file.read()
                except Exception:
                    continue
                datum = {"url": path, "caption": caption, "img_content": img_content, "img_path": img_path}
                #print(caption, len(img_content), img_path)
                #output.append(datum)
                yield datum 
    #print(time.time() - t0)
    #return output

class MecaDataset:

    def __init__(self, filelist):
        self.filelist = filelist

    def __getitem__(self, idx):
        path = self.filelist[idx]
        return parse_meca(path)

    def __len__(self):
        return len(self.filelist)

class MecaIterableDataset(torch.utils.data.IterableDataset):
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
                yield from parse_meca(fs)
            except Exception as ex:
                print(ex)

