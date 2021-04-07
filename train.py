import hydra
from collections import defaultdict
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torchviz import make_dot
import os
import random
import numpy as np

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TMCfg
from data.mol_callback import MolCallback
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model

def count_keys(params):
    ret = defaultdict(int)
    for key, value in params.items():
        if isinstance(value, dict):
            for key2, count in count_keys(value).items():
                ret[key2] += count
        else:
            ret[key] += 1
    return ret

def flatten_dict(params, key_counts=None):
    if key_counts is None:
        key_counts = count_keys(params)
    ret = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for key2, val2 in flatten_dict(value).items():
                if key_counts[key2] == 1 and not '.' in key2:
                    ret[key2] = val2
                else:
                    ret[f"{key}.{key2}"] = val2
        else:
            ret[key] = value
    return ret

def get_params(cfg):
    params = OmegaConf.to_container(cfg, True)
    for key in ["debug", "platform"]:
        del params[key]
    params = flatten_dict(params)
    return params

@hydra.main(config_path='cfg', config_name="config")
def train(cfg):

    TMCfg.set_cfg(cfg.data)
    
    is_test = (cfg.debug.stop_at is not None) or cfg.debug.save_img
    n_workers = 0 if is_test else cfg.platform.num_workers

    train_d = make_dataset(cfg, True)
    test_d = make_dataset(cfg, False)

    train_loader = DataLoader(train_d, batch_size=cfg.batch_size,
                              num_workers=n_workers, pin_memory=True,
                              shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_d, batch_size=2,
                             num_workers=n_workers, pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)

    model = make_model(cfg)

    tags = []
    params = get_params(cfg)

    if is_test:
        logger = None
        print("Neptune logging disabled. The parameters are:")
        for key, val in params.items():
            print(f"\t{key}: {val}")
    else:
        logger = NeptuneLogger(project_name="mixarcid/molucinate",
                               experiment_name=cfg.name,
                               params=params,
                               tags=tags)

    if cfg.debug.save_img:
        batch = next(iter(train_loader))
        mu, logvar = model(batch)
        y = model.decode(mu)
        dot = make_dot(y.molgrid)
        out_fname = 'model.pdf'
        print(os.path.abspath(out_fname))
        f = dot.render('.'.join(out_fname.split('.')[:-1]), format='pdf')
        return

    checkpoint_callback = None
    mol_cb = MolCallback(cfg)
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),
                         checkpoint_callback=checkpoint_callback,
                         callbacks = [mol_cb],
                         logger=logger,
                         gradient_clip_val=cfg.grad_clip)
    trainer.fit(model, train_loader, test_loader)
    
if __name__ == '__main__':
    train()
