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
import io

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
                with zip_file.open(img_path) as img_file:
                    img_content = img_file.read()
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
        print(self.start, self.end)
        for fs in self.filelist[self.start:self.end]:
            yield from parse_meca(fs)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def loader(filelist, num_workers=16):
    ds = MecaIterableDataset(filelist)
    bs = max(num_workers, 1)
    #dl = DataLoader(ds, num_workers=0, batch_size=bs, collate_fn=lambda x:x)
    dl = DataLoader(ds, num_workers=bs, batch_size=bs, collate_fn=lambda x:x, worker_init_fn=worker_init_fn)
    for batch in dl:
        for fig_i in batch:
            yield fig_i

class ShuffledIter:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        while True:
            random.shuffle(self.data)
            yield from self.data



def extract_figure_caption_pairs(filelist, *, nb_shards=1, path_shards=".", num_workers=1):
    filelist = [f.strip() for f in open(filelist).readlines()]
    filelist = filelist[0:1]
    fds = [
        fsspec.open(os.path.join(path_shards, f"shard-{i:05d}.tar"),"wb").open() for i in range(nb_shards)
    ]
    sinks = [wds.TarWriter(fd) for fd in fds]
    sink_iter = iter(ShuffledIter(sinks))
    nb = 0
    t0 = time.time()
    idx = 0
    for data in loader(filelist, num_workers=num_workers):
        #key = data["url"].replace("s3://", "").replace("/", "_") + data["img_path"].replace("/", "_").replace(".", "_")
        key = str(nb)
        ext = os.path.splitext(data["img_path"])[-1].replace(".", "")
        datum = {
            "__key__": key,
            ext: data["img_content"],
            "txt": data["caption"],
            "url": data["url"],
        }
        sink = next(sink_iter)
        sink.write(datum)
        nb += 1
        dt = time.time() - t0
        if nb % 10 == 0:
            print(nb, nb / dt)
    for s in sinks:
        s.close()
    for fd in fds:
        fd.close()

if __name__ == "__main__":
    run([extract_figure_caption_pairs])
