import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import *

class CheckpointCallback(Callback):

    def __init__(self, gcfg):
        self.name = gcfg.name
        self.checkpoint_n_batches = gcfg.debug.checkpoint_n_batches
        self.results_path = gcfg.platform.results_path
        self.n = 0

    def cb(self, trainer, pl_module):
        if trainer.logger:
            path = f"{self.results_path}{self.name}.pt"
            torch.save(trainer.model.state_dict(), path)
            trainer.logger.experiment.log_artifact(path, "weights.pt")
    
    def on_batch_end(self, trainer, pl_module):
        if self.checkpoint_n_batches is None: return
        if self.n > self.checkpoint_n_batches:
            self.n = 0
            self.cb(trainer, pl_module)
        else:
            self.n += 1

    def on_epoch_end(self, trainer, pl_module):
        self.cb(trainer, pl_module)
