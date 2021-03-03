import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import *

from .render import *

class MolCallback(Callback):

    def __init__(self, gcfg):
        super(MolCallback, self).__init__()
        self.name = gcfg.name
        self.cb_n_batches = gcfg.debug.cb_n_batches
        self.batch_size = 2#gcfg.batch_size
        self.latent_size = gcfg.model.latent_size
        self.results_path = gcfg.platform.results_path
        self.n = 0

    def checkpoint_imgs(self, trainer, log_name, fname, img_list):
        img_path = self.results_path + fname + "_" + log_name + ".png"
        export_multi(img_path, img_list)
        if trainer.logger: trainer.logger.experiment.log_image(log_name, img_path)

    def create_z(self, device):
        return torch.normal(torch.zeros((self.batch_size, self.latent_size)), 1).to(device)

    def cb_batch(self, model, batch_idx, batch, trainer):
        batch = batch.to(model.device)
        mu, logvar = model(batch)
        recon = model.decode(mu)
        z = self.create_z(model.device)
        gen = model.decode(z)
        for i in range(self.batch_size):
            if i > 0: break
            gen_mg_img = render_molgrid(gen[i])
            mg_img = render_molgrid(batch[i])
            recon_mg_img = render_molgrid(recon[i])
            fname = "{}_epoch_{}_{}".format(self.name,
                                            trainer.current_epoch,
                                            batch_idx*self.batch_size+i)
            self.checkpoint_imgs(trainer, "gen", fname, [[gen_mg_img]])
            self.checkpoint_imgs(trainer, "recon", fname, [[mg_img, recon_mg_img]])

    def cb(self, trainer, pl_module):
        trainer.model.eval()
        val_data = trainer.val_dataloaders[0]
        try:
            with torch.no_grad():
                for i, batch in enumerate(val_data):
                    self.cb_batch(trainer.model, i, batch, trainer)
                    break
        finally:
            trainer.model.train()
    
    def on_batch_end(self, trainer, pl_module):
        if self.cb_n_batches is None: return
        if self.n > self.cb_n_batches:
            self.n = 0
            self.cb(trainer, pl_module)
        else:
            self.n += 1

    def on_epoch_end(self, trainer, pl_module):
        self.cb(trainer, pl_module)