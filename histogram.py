import neptune.new as neptune
import hydra
import torch
import random
import numpy as np
from omegaconf import OmegaConf

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import QED

from tqdm import tqdm
import matplotlib.pyplot as plt
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

    test_smiles_set = make_dataset(cfg, False).get_smiles_set()
    print(f"There are {len(test_smiles_set)} molecules in the testing set")
    gen_fname = f"{cfg.platform.results_path}{cfg.test.run_id}_gen.txt"
    gen_qeds = []
    test_qeds = []
    with open(gen_fname, 'r') as f:
        for line in tqdm(f.readlines()):
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue
            gen_qeds.append(QED.qed(mol))

    
    for i, smiles in enumerate(test_smiles_set):
        if i > len(gen_qeds): continue
        mol = Chem.MolFromSmiles(smiles)
        test_qeds.append(QED.qed(mol))
        

    n_bins = 20
    fig, axs = plt.subplots(1, 1)
    axs.hist(gen_qeds, bins=n_bins, alpha=0.5, label='generated')
    axs.hist(test_qeds, bins=n_bins, alpha=0.5, label='ZINC')

    axs.set_xlabel("QED Score", size=14)
    axs.set_ylabel("Number of Molecules", size=14)
    axs.set_title("Generated vs ZINC QED")
    plt.legend(loc='upper right')
    plt.savefig("figures/qed_hist.png")
    #plt.show()
    
if __name__ == "__main__":
    test()
