import neptune.new as neptune
import hydra
import torch
import random
import numpy as np
from omegaconf import OmegaConf

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from tqdm import tqdm

from pytorch_lightning.utilities.cloud_io import load as pl_load

from data.tensor_mol import TMCfg
from data.render import *
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model
from models.metrics import get_gen_metrics
from utils.cfg import get_checkpoint_cfg

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    TMCfg.set_cfg(cfg.data)

    
    batch_size = 64
    test_recon = False
    num_gen = 20

    n_workers = cfg.platform.num_workers
    test_d = make_dataset(cfg, False)
    
    test_loader = DataLoader(test_d, batch_size=batch_size,
                             num_workers=n_workers, #pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)

    train_smiles_set = make_dataset(cfg, True).get_smiles_set()
    print(f"There are {len(train_smiles_set)} molecules in the training set")
    num_valid = 0
    num_novel = 0
    tot = 0

    gen_fname = f"{cfg.platform.results_path}{cfg.test.run_id}_gen.txt"
    unique_set = set()
    novel_set = set()
    with open(gen_fname, 'r') as f:
        for line in tqdm(f.readlines()):
            tot += 1
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                unique_set.add(smiles)
                num_valid += 1
                if smiles not in train_smiles_set:
                    novel_set.add(smiles)

    num_unique = len(unique_set)
    num_novel = len(novel_set)
    print(f"Validity: {num_valid/tot}")
    print(f"Uniqueness: {num_unique/num_valid}")
    print(f"Novelty: {num_novel/num_unique}")
    print(f"Novel/Sample: {num_novel/tot}")
    
if __name__ == "__main__":
    test()
