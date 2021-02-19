import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate

class Collatable:
    
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
        return type(self)(**kwargs)

    def __len__(self):
        for attr in self.recurse():
            l = len(getattr(self, attr))
            if l is not None: return l
        return None

    def __getitem__(self, idx):
        kwargs = { attr: getattr(self, attr)[idx] for attr in self.recurse() }
        return type(self)(**kwargs)
    
def collate(batch):
    example = batch[0]
    if isinstance(example, Collatable):
        kwargs = {
            attr: collate([getattr(obj, attr) for obj in batch])
            for attr in example.recurse()
        }
        return type(example)(**kwargs)
    else:
        return default_collate(batch)

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=collate, **kwargs
        )
