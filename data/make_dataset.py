from .zinc import ZincDataset
from .qm9 import QM9Dataset

def make_dataset(cfg, is_train):
    return {
        "zinc": ZincDataset,
        "qm9": QM9Dataset
    }[cfg.dataset](cfg, is_train)
