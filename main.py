import random
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

from meca import MecaIterableDataset, parse_meca_to_list
from arxiv import ArxivFigureCaptions, ArxivEquations, parse_arxiv_shard_tar_to_list
from pubmed import PubMedIterableDataset, parse_pubmed_to_list

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


def get_ds(filelist, processor):
    if processor in ("biorxiv", "medarxiv"):
        ds = MecaIterableDataset(filelist)  
    elif processor in ("pubmed",):
        ds = PubMedIterableDataset(filelist)
    elif processor == "arxiv_figure_captions":
        ds = ArxivFigureCaptions(filelist)
    elif processor == "arxiv_equations":
        ds = ArxivEquations(filelist)
    else:
        raise ValueError(processor)
    return ds


def get_fn(filelist, processor):
    if processor  in ("biorxiv", "medarxiv"):
        return parse_meca_to_list, {}
    elif processor == "arxiv_figure_captions":
        return parse_arxiv_shard_tar_to_list, {"extract": "figure_captions"}
    elif processor == "arxiv_equations":
        return parse_arxiv_shard_tar_to_list, {"extract": "math"}
    else:
        raise ValueError(processor)


def loader(filelist, num_workers=16, processor="meca"):


    ds = get_ds(filelist, processor)

    bs = max(num_workers, 1)
    #dl = DataLoader(ds, num_workers=0, batch_size=bs, collate_fn=lambda x:x)
    dl = DataLoader(ds, num_workers=num_workers, batch_size=bs, collate_fn=lambda x:x, worker_init_fn=worker_init_fn)
    for batch in dl:
        for fig_i in batch:
            yield fig_i


def loader2(filelist, num_workers=16, processor="meca"):
    fn, kw = get_fn(filelist, processor)
    with Parallel(n_jobs=num_workers, backend="multiprocessing") as parallel:
        for i in range(0, len(filelist), num_workers):
            fs = filelist[i:i+num_workers]
            results = parallel(delayed(fn)(f, **kw) for f in fs)
            for r in results:
                yield from r


class ShuffledIter:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        while True:
            random.shuffle(self.data)
            yield from self.data


def extract(filelist, *, start=0, nb:int=None, nb_shards=1, path_shards=".", num_workers=1, processor="medarxiv", writer="wds", total:int=None, chunk_size:int=None, seed=42, shard_prefix="shard", resume=None):
    random.seed(seed)
    filelist = [f.strip() for f in open(filelist).readlines()]
    if nb is None:
        end = len(filelist)
    else:
        end = start + nb
    filelist = filelist[start:end]
    if resume:
        import re
        names = re.findall("'(.+)'", open(resume).read())
        names = set(names)
        filelist = [fs for fs in filelist if fs not in names]
    if not len(filelist):
        return
    random.shuffle(filelist)
    fds = [
        fsspec.open(os.path.join(path_shards, f"{shard_prefix}-{i:05d}.tar"),"wb").open() for i in range(nb_shards)
    ]
    sinks = [TarWriter(fd, append=False) for fd in fds]
    sink_iter = iter(ShuffledIter(sinks))
    nb = 0
    t0 = time.time()
    BS = chunk_size if chunk_size else len(filelist)
    for i in range(0, len(filelist), BS):
        print(f"Processing filelist chunk from {i} to {i+BS},  current elapsed time = {time.time()-t0} seconds.")
        for data in loader(filelist[i:i+BS], processor=processor, num_workers=num_workers):
            key = str(nb)
            if "img_content" in data:
                ext = os.path.splitext(data["img_path"])[-1].replace(".", "")
                datum = {
                    "__key__": key,
                    ext: data["img_content"],
                    "txt": data["caption"],
                    "url": data["url"],
                }
            else:
                datum = data
                datum['__key__'] = key
            sink = next(sink_iter)
            sink.write(datum)
            nb += 1
            if total and nb == total:
                print("Total reached")
                break
            dt = time.time() - t0
            if nb % 1000 == 0:
                print(f"Number of samples written: {nb}, Speed: {nb/dt} samples/s")
        if total and nb == total:
            break
    for s in sinks:
        s.close()
    for fd in fds:
        fd.close()
    fs = str(filelist) if nb else None
    print(f"Finished {fs}, total samples written:", nb)


if __name__ == "__main__":
    run([extract])
