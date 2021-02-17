from .zinc import ZincDataset

def make_dataset(cfg, is_train):
    return {
        "zinc": ZincDataset
    }[cfg.dataset](cfg, is_train)
