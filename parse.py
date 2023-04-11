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

            for fig in dicts_out:
                caption = fig['fig_caption']
                graphic_ref = fig['graphic_ref']
                img_path  = "content/" + graphic_ref
                with zip_file.open(img_path) as img_file:
                    img_content = img_file.read()
                print(caption, len(img_content), img_path)
                output.append({"caption": caption, "img_content": img_content, "img_path": img_path})
    return output

#parse_meca("s3://s-laion/papers/medarxiv/Current_Content/September_2022/ff7dc401-6d36-1014-9d1e-9b47c4401385.meca")
parse_meca("s3://s-laion/papers/biorxiv/Current_Content/September_2022/ffd8bd05-6f20-1014-b384-e0cf8b3eb6fb.meca")
