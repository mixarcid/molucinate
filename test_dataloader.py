from rdkit import Chem
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchviz import make_dot
import os
import random
import numpy as np
from tqdm import tqdm

assert(torch.__version__ >= '1.7.1')

#torch.multiprocessing.set_sharing_strategy('file_system')

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
from data.checkpoint_callback import CheckpointCallback
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model
from utils.cfg import get_params, get_checkpoint_cfg

def train():

    cfg = OmegaConf.create({
        "debug": {
            "profile": False
        },
        "data": {
            "grid_dim": 32,
            "grid_step": 0.5,
            "max_atoms": 32,
            "max_valence": 4,
            "kekulize": False,
            "randomize_smiles": False,
            "use_kps": True,
            "pos_randomize_std": 0.5,
            "atom_randomize_prob": 0.1,
        },
        "dataset": "zinc",
        "platform": {
            "zinc_dir": "/home/boris/Data/Zinc/"
        }
    })
    
    TMCfg.set_cfg(cfg.data)

    batch_size = 4
    n_workers = 0
    train_d = make_dataset(cfg, True)
    train_loader = DataLoader(train_d, batch_size=batch_size,
                              num_workers=n_workers, #pin_memory=True,
                              shuffle=True, worker_init_fn=seed_worker)

    for i, batch in enumerate(train_loader):
        print(i)
        if i > 20: break
    
if __name__ == '__main__':
    train()
