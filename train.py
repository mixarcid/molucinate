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

        
@hydra.main(config_path='cfg', config_name="config")
def train(cfg):

    TMCfg.set_cfg(cfg.data)

    run_id_path = None
    run_id = None
    ckpt_path = None
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ['SLURM_JOB_ID']
        run_id_path = f"{cfg.platform.results_path}{job_id}_run_id.txt"
        try:
            with open(run_id_path, 'r') as f:
                run_id = f.read().strip()
        except FileNotFoundError:
            pass
    
    if run_id is not None:
        cfg, ckpt_path = get_checkpoint_cfg(cfg, run_id)
    
    is_test = (cfg.debug.stop_at is not None) or cfg.debug.save_img
    #if is_test:
    #    torch.autograd.set_detect_anomaly(True)
    n_workers = 0 if is_test else cfg.platform.num_workers
    # n_workers = cfg.platform.num_workers
    
    train_d = make_dataset(cfg, True)
    test_d = make_dataset(cfg, False)

    train_loader = DataLoader(train_d, batch_size=cfg.batch_size,
                              num_workers=n_workers, #pin_memory=True,
                              shuffle=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_d, batch_size=cfg.batch_size,
                             num_workers=n_workers, #pin_memory=True,
                             shuffle=True, worker_init_fn=seed_worker)

    model = make_model(cfg)

    tags = []
    params = get_params(cfg)

    if is_test or not cfg.use_neptune:
        logger = None
        print("Neptune logging disabled. The parameters are:")
        for key, val in params.items():
            print(f"\t{key}: {val}")
    else:
        logger = NeptuneLogger(project_name=cfg.neptune_project,
                               experiment_name=cfg.name,
                               params=params,
                               tags=tags,
                               experiment_id=run_id)
        if run_id_path is not None:
            with open(run_id_path, 'w') as f:
                f.write(logger.experiment.id)

    if cfg.debug.save_img:
        batch = next(iter(train_loader))
        mu, logvar = model(batch)
        y = model.decode(mu, batch)
        dot = make_dot(y.kps_1h)
        out_fname = 'model.pdf'
        print(os.path.abspath(out_fname))
        f = dot.render('.'.join(out_fname.split('.')[:-1]), format='pdf')
        return

    checkpoint_callback = None
    mol_cb = MolCallback(cfg)
    checkpoint_cb = CheckpointCallback(cfg)

    callbacks = [mol_cb, checkpoint_cb]
    if logger is not None:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),
                         checkpoint_callback=checkpoint_callback,
                         callbacks = callbacks,
                         logger=logger,
                         max_epochs=1 if cfg.debug.profile else None,
                         num_sanity_val_steps=0, #0 if cfg.debug.profile else 1,
                         gradient_clip_val=cfg.grad_clip,
                         resume_from_checkpoint=ckpt_path)

    cfg_file = f'{cfg.platform.results_path}{cfg.name}.yaml'
    with open(cfg_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    if trainer.logger:
        trainer.logger.experiment.log_artifact(cfg_file)
    
    trainer.fit(model, train_loader, test_loader)
    
if __name__ == '__main__':
    train()
