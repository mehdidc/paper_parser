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
    fd = io.BytesIO(data)
    with zipfile.ZipFile(fd, mode='r') as zip_file:
        for f in zip_file.filelist:
            if not f.filename.startswith("content/"):
                continue
            if not f.filename.endswith(".xml"):
                continue
            with zip_file.open(f.filename) as xml_file:
                dicts_out = pp.parse_pubmed_caption(xml_file)
            if not dicts_out:
                continue
            for fig in dicts_out:
                caption = fig['fig_caption']
                graphic_ref = fig['graphic_ref']
                img_path  = "content/" + graphic_ref
                with zip_file.open(img_path) as img_file:
                    img_content = img_file.read()
                #print(caption, len(img_content), img_path)
                output.append({"url": path, "caption": caption, "img_content": img_content, "img_path": img_path})
    return output

class MecaDataset:

    def __init__(self, filelist):
        self.filelist = filelist

    def __getitem__(self, idx):
        path = self.filelist[idx]
        try:
            return parse_meca(path)
        except Exception as ex:
            return []

    def __len__(self):
        return len(self.filelist)

def loader(filelist, num_workers=8):
    ds = MecaDataset(filelist)
    bs = num_workers
    dl = DataLoader(ds, num_workers=num_workers, batch_size=bs, collate_fn=lambda x:x)
    for batch in dl:
        for xi in batch:
            for fig_i in xi:
                yield fig_i

class ShuffledIter:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        while True:
            random.shuffle(self.data)
            yield from self.data



def extract_figure_caption_pairs(filelist, *, nb_shards=1, path_shards="."):
    filelist = [f.strip() for f in open(filelist).readlines()]
    nb_shards = 1
    sinks = [wds.TarWriter(os.path.join(path_shards, f"shard-{i:05d}.tar")) for i in range(nb_shards)]
    sink_iter = iter(ShuffledIter(sinks))
    for data in loader(filelist):
        key = data["url"].replace("s3://", "").replace("/", "_") + data["img_path"].replace("/", "_")
        ext = os.path.splitext(data["img_path"])[-1].replace(".", "")
        datum = {
            "__key__": key,
            ext: data["img_content"],
            "txt": data["caption"],
        }
        sink = next(sink_iter)
        sink.write(datum)
        break
    for s in sinks:
        s.close()

if __name__ == "__main__":
    run([extract_figure_caption_pairs])
