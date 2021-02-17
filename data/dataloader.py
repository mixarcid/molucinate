import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

class Collatable:

    def __init__(self, example=None, collate_len=None):
        self.collate_len = collate_len
    
    def recurse(self):
        for attr in dir(self):
            member = getattr(self, attr)
            if isinstance(member, torch.Tensor) or isinstance(member, Collatable):
                yield attr

    def to(self, device):
        kwargs = {
            attr: getattr(self, attr).to(device)
            for attr in self.recurse()
        }
        return type(self)(example=self, **kwargs)

    def __len__(self):
        if self.collate_len is None:
            raise Exception("Trying to determine the len of a non-collated object")
        return self.collate_len

def collate(batch):
    example = batch[0]
    if isinstance(example, Collatable):
        kwargs = {
            attr: collate([getattr(obj, attr) for obj in batch])
            for attr in example.recurse()
        }
        return type(example)(example=example,
                             collate_len=len(batch),
                             **kwargs)
    else:
        return default_collate(batch)

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=collate, **kwargs
        )
