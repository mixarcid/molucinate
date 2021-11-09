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

def recon_metrics(cfg, model, batch_idx, batch):
    
    (batch, _), rad_gyr = batch
    batch = batch.to(model.device)
    mu, logvar = model(batch)
    recon = model.decode(mu)

    rmsds = []
    for i in range(batch.atom_types.size(0)):
        recon_max = recon[i].argmax()
        mol1 = batch[i].get_mol()
        mol2 = recon_max.get_mol()
        if Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2):
            rmsds.append(rmsd_single(recon_max, batch[i]))

    return rmsds

def gen_molecules(cfg, model, f, idx, batch_size, prefix):
    z = create_z(cfg, batch_size, model.device)
    gen = model.decode(z)
    for i in range(batch_size):
        mol = gen[i].argmax().get_mol()
        smiles = Chem.MolToSmiles(mol)
        f.write(smiles)
        f.write('\n')
        out_sd = f"{prefix}{idx*batch_size + i}.sdf"
        w = Chem.SDWriter(out_sd)
        w.write(mol)
        w.close()
            
def test_batch(cfg, model, batch_idx, batch):
    batch = batch.to(model.device)
    mu, logvar = model(batch)
    recon = model.decode(mu)
    z = create_z(cfg, model.device)
    gen = model.decode(z)

    #print(get_gen_metrics(gen))
    
    for i in range(cfg.batch_size):
        #break
        render_kp_rt(gen[i].argmax())
        #gen_mg_img = render_tmol(gen[i])
        #mg_img = render_tmol(batch[i], recon[i])
        #recon_mg_img = render_tmol(recon[i])
            

@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    TMCfg.set_cfg(cfg.data)

    
    batch_size = 16
    test_recon = True
    num_gen = 1000

    n_workers = cfg.platform.num_workers
    test_d = make_dataset(cfg, False)

    dataset_name = cfg.dataset
    test_loader = DataLoader(test_d, batch_size=batch_size,
                             num_workers=n_workers, #pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)

    # path = f"{cfg.platform.results_path}{cfg.test.run_id}_weights.pt"
    # cfg_path = f"{cfg.platform.results_path}{cfg.test.run_id}_cfg.yaml"
    # if not cfg.test.use_cache:
    #     print(f"Downloading latest {cfg.test.run_id} weights")
    #     run = neptune.init(project="mixarcid/molucinate",
    #                        run=cfg.test.run_id)
    #     run["artifacts/weights.pt"].download(path)
    #     run["artifacts/cfg.yaml"].download(cfg_path)

    run_id = cfg.test.run_id
    cfg, ckpt_path = get_checkpoint_cfg(cfg, run_id, cfg.test.use_cache)

    print(f"Loading model for {run_id}")
    model = make_model(cfg)
    checkpoint = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()

    if test_recon:
        print(f"Testing recon on {dataset_name}")
        rmsds = []
        tot = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                rmsds += recon_metrics(cfg, model, i, batch)
                tot += batch_size
                if tot > 1000:
                    break

        num_correct = len(rmsds)
        rmsd = np.mean(rmsds)
        mean = num_correct / tot
        print(f"Recon acc: {mean}")
        print(f"RMSD: {rmsd}")
        return

    out_fname = f"{cfg.platform.results_path}{run_id}_gen.txt"
    prefix = f"{cfg.platform.results_path}{run_id}_gen_"
    print(f"Generating molecules to {out_fname}")
    with open(out_fname, 'w') as f:
        for i in tqdm(range(num_gen)):
            gen_molecules(cfg, model, f, i, batch_size, prefix)

if __name__ == "__main__":
    test()
