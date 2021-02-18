import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger

from data.mol_callback import MolCallback
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model

def flatten_dict(params):
    ret = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for key2, val2 in flatten_dict(value).items():
                ret[f"{key}.{key2}"] = val2
        else:
            ret[key] = value
    return ret

def get_params(cfg):
    params = OmegaConf.to_container(cfg, True)
    del params["platform"]
    params = flatten_dict(params)
    return params

@hydra.main(config_path='cfg', config_name="config")
def train(cfg):
    is_test = cfg.platform.stop_at is not None
    n_workers = 0 if is_test else cfg.platform.num_workers

    train_d = make_dataset(cfg, True)
    test_d = make_dataset(cfg, False)

    train_loader = DataLoader(train_d, batch_size=cfg.batch_size, num_workers=n_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=cfg.batch_size, num_workers=n_workers, pin_memory=True, shuffle=True)

    model = make_model(cfg)

    tags = []
    params = get_params(cfg)

    if is_test:
        logger = None
    else:
        logger = NeptuneLogger(project_name="mixarcid/molucinate",
                               experiment_name=cfg.name,
                               params=params,
                               tags=tags)

    checkpoint_callback = None
    mol_cb = MolCallback(cfg)
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),
                         checkpoint_callback=checkpoint_callback,
                         callbacks = [mol_cb],
                         logger=logger)
    trainer.fit(model, train_loader, test_loader)
    
if __name__ == '__main__':
    train()
