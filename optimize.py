import neptune.new as neptune
import hydra
import torch
import random
import numpy as np
from omegaconf import OmegaConf
import cv2

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit import Chem
from rdkit.Chem import rdMolTransforms, Descriptors3D
from tqdm import tqdm

from pytorch_lightning.utilities.cloud_io import load as pl_load

from data.tensor_mol import TMCfg
from data.render import *
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model
from models.metrics import get_gen_metrics
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

def render_optim(tmol, rad, rec_max):
    orig = tmol != rec_max
    img = render_tmol(tmol, rec_max)
    og = (10, 10)
    font_scale = 0.75
    thickness = 1
    text = "Orig." if orig else "Optim."
    img = cv2.putText(img, "{} radius: {:.2f}".format(text, float(rad)), (10, 590), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA) 
    return img
                      
def optimize_mols(cfg, model, batch_idx, batch, shrink):
    
    (batch, _), rad_gyr = batch
    batch = batch.to(model.device)
    z, logvar = model(batch)
    gy_pred = model.prop_pred(z)
    print(f"original pred: {gy_pred} (actual {rad_gyr})")

    z = torch.nn.Parameter(z, requires_grad=True)
    optim = torch.optim.SGD([z], lr=5e-1)
    for i in range(500):
        new_pred = model.prop_pred(z)
        if not shrink:
            new_pred = -new_pred
        new_pred.backward()
        optim.step()
        optim.zero_grad()
        #print(f"new pred at {i}: {new_pred}")

    with torch.no_grad():
        recon = model.decode(z)

    img_list = []
    for i in range(batch.atom_types.size(0)):
         mol1 = batch[i].get_mol()
         rec_max = recon[i].argmax()
         mol2 = rec_max.get_mol()
         try:
             Chem.SanitizeMol(mol2)
         except:
             return None, None
         
         rad_gyr_new = Descriptors3D.RadiusOfGyration(mol2)
         print(f"Final actual: {rad_gyr_new}")
         img_list += ([render_optim(batch[i], rad_gyr[i], rec_max),
                       render_optim(rec_max, rad_gyr_new, rec_max)])
         #render_kp_rt(batch[i])
         #render_kp_rt(recon[i].argmax())
    return img_list[0], img_list[1]
         

@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    TMCfg.set_cfg(cfg.data)

    
    batch_size = 1
    test_recon = True
    num_gen = 1000

    n_workers = cfg.platform.num_workers
    test_d = make_dataset(cfg, False)
    
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

    cfg, ckpt_path = get_checkpoint_cfg(cfg, cfg.test.run_id, cfg.test.use_cache)

    run_id = cfg.test.run_id
    print(f"Loading model for {cfg.test.run_id}")
    model = make_model(cfg)
    checkpoint = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    #model.cuda()
    model.eval()

    tot = 0
    shrink = False
    num_mols = 8
    out_fname = f"figures/optim_{'small' if shrink else 'large'}.png"
    img1_list = []
    img2_list = []
    for i, batch in enumerate(tqdm(test_loader)):
        img1, img2 = optimize_mols(cfg, model, i, batch, shrink)
        if img1 is None: continue
        img1_list.append(img1)
        img2_list.append(img2)
        tot += batch_size
        if tot > num_mols:
            break

    print(f"Saving to {out_fname}")
    export_multi(out_fname, [img1_list, img2_list])

if __name__ == "__main__":
    test()
