import neptune.new as neptune
import hydra
import torch
import random
import numpy as np

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TMCfg
from data.render import *
from data.dataloader import DataLoader
from data.make_dataset import make_dataset
from models.make_model import make_model
from models.metrics import get_gen_metrics

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_z(cfg, device):
    return torch.normal(torch.zeros((cfg.batch_size, cfg.model.latent_size)), 1).to(device)

def test_batch(cfg, model, batch_idx, batch):
    batch = batch.to(model.device)
    mu, logvar = model(batch)
    recon = model.decode(mu)
    z = create_z(cfg, model.device)
    gen = model.decode(z)

    print(get_gen_metrics(gen))
    
    for i in range(cfg.batch_size):
        break
        render_kp_rt(gen[i].argmax())
        #gen_mg_img = render_tmol(gen[i])
        #mg_img = render_tmol(batch[i], recon[i])
        #recon_mg_img = render_tmol(recon[i])
            

@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    TMCfg.set_cfg(cfg.data)

    n_workers = cfg.platform.num_workers
    test_d = make_dataset(cfg, False)
    test_loader = DataLoader(test_d, batch_size=cfg.batch_size,
                             num_workers=n_workers, #pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)

    path = f"{cfg.platform.results_path}{cfg.test.run_id}_weights.pt"
    if not cfg.test.use_cache:
        print(f"Downloading latest {cfg.test.run_id} weights")
        run = neptune.init(project="mixarcid/molucinate",
                           run=cfg.test.run_id)
        run["artifacts/weights.pt"].download(path)

    print(f"Loading model for {cfg.test.run_id}")
    model = make_model(cfg)
    model.load_state_dict(torch.load(path))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            test_batch(cfg, model, i, batch)
            break

    

if __name__ == "__main__":
    test()
