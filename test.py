import webdataset as wds
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
import braceexpand
import random
import sys
def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value



class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        #urls = wds.shardlists.expand_urls(urls)
        urls = list(braceexpand.braceexpand(urls))
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        #print(epoch)
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


train_transform = transforms.Compose([
        #transforms.Resize((256,256)),
        transforms.Resize(512),
        #transforms.Re(512, scale=(0.95, 1.0), interpolation=3),
    ])
#pipeline = [ResampledShards2("/p/fastdata/datasets/pubmed/figure-captions2/{00000..02499}.tar")]
pipeline = [ResampledShards2("medarxiv_figure_captions/shard-{00000..02499}.tar")]
#pipeline = [ResampledShards2("shard-00000.tar")]

pipeline.extend([
    wds.split_by_node,
    wds.split_by_worker,
    wds.tarfile_to_samples(),
])
pipeline.extend([
    wds.decode("pilrgb"),
    wds.rename(image="tif"),
    wds.map_dict(image=train_transform),
    wds.to_tuple("image","txt"),
    wds.batched(1, partial=False),
])
dataset = wds.DataPipeline(*pipeline)
data_loader = wds.WebLoader(
    dataset,
    batch_size=None,
    shuffle=False,
    num_workers=1,
    persistent_workers=True,
)

for x,y in data_loader:
    for xi, yi in zip(x, y):
        xi.save("out.jpg")
        print(yi)
    break