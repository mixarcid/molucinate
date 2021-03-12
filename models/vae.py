import pytorch_lightning as pl
import torch
import torch.nn as nn
from .nets.make_net import make_encoder, make_decoder
from .losses import get_loss_fn

class VAE(pl.LightningModule):

    def __init__(self, cfg, gcfg):
        super(VAE, self).__init__()
        self.should_reparam = (cfg.loss.kl_lambda > 0)
        self.optim_name = cfg.optimizer
        self.learn_rate = cfg.learn_rate
        self.latent_size = cfg.latent_size
        self.hidden_size = cfg.hidden_size
        self.encoder = make_encoder(self.hidden_size, cfg, gcfg)
        self.decoder = make_decoder(self.latent_size, cfg, gcfg)
        # hidden => mu
        self.fc1 = nn.Linear(self.hidden_size, self.latent_size)
        # hidden => logvar
        self.fc2 = nn.Linear(self.hidden_size, self.latent_size)
        self.loss_fn = get_loss_fn('vae', cfg.loss)

    def encode(self, tmol):
        h = self.encoder(tmol)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z, tmol=None):
        return self.decoder(z, tmol)

    def reparameterize(self, mu, logvar):
        if self.training and self.should_reparam:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, tmol):
        return self.encode(tmol)

    def training_step(self, batch, batch_idx):
        mu, logvar = self(batch)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch)
        loss, terms = self.loss_fn(recon, batch, mu, logvar)
        self.log('train_loss', loss, prog_bar=True)
        for name, term in terms.items():
            self.log(f'train_{name}_loss', term)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optim_class = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD
        }[self.optim_name]
        optimizer = optim_class(self.parameters(), lr=self.learn_rate)
        return optimizer

    def shared_eval(self, batch, batch_idx, prefix):
        mu, logvar = self(batch)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        loss, terms = self.loss_fn(recon, batch, mu, logvar)
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        for name, term in terms.items():
            self.log(f'{prefix}_{name}_loss', term)
        return loss
