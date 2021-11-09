import neptune.new as neptune
import hydra
import torch
import random
import numpy as np
from copy import deepcopy
import cv2
from omegaconf import OmegaConf

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from tqdm import tqdm

from pytorch_lightning.utilities.cloud_io import load as pl_load

from data.tensor_mol import TMCfg, TensorMol
from data.render import *
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model
from models.metrics import get_gen_metrics, rmsd_single
from utils.cfg import get_checkpoint_cfg

from models.vae import VAE

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_z(cfg, batch_size, device):
    return torch.normal(torch.zeros((batch_size, cfg.model.latent_size)), 1).to(device)

def gen_molecules(cfg, model, idx, batch_size):
    while True:
        z = create_z(cfg, batch_size, model.device)
        gen = model.decode(z)
        i = 0
        tmol = gen[i].argmax()
        mol = tmol.get_mol()
        
        try:
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol)
        except:
            continue

        try:
            mol_uff = deepcopy(mol)
            AllChem.UFFOptimizeMolecule(mol_uff, 500)
            rmsd = Chem.rdMolAlign.AlignMol(mol_uff, mol)
        except RuntimeError:
            continue
            
        return render_tmol(tmol, mol_uff=TensorMol(mol_uff))
            

@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    TMCfg.set_cfg(cfg.data)

    
    batch_size = 1
    num_rows = 2
    num_cols = 8

    n_workers = cfg.platform.num_workers
    test_d = make_dataset(cfg, False)
    
    test_loader = DataLoader(test_d, batch_size=batch_size,
                             num_workers=n_workers, #pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)


    run_id = cfg.test.run_id
    cfg, ckpt_path = get_checkpoint_cfg(cfg, run_id, cfg.test.use_cache)

    print(f"Loading model for {run_id}")
    model = make_model(cfg)
    checkpoint = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()

    imgs = []
    for i in range(num_rows):
        imgs.append([])
        for j in tqdm(range(num_cols)):
            img = gen_molecules(cfg, model, i, batch_size)
            imgs[-1].append(img)
    export_multi("slides/generate.png", imgs)

if __name__ == "__main__":
    test()
