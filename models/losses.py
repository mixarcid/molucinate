import torch
from torch.nn import functional as F

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(recon, x, mu, logvar, cfg):
    kl = kl_loss(mu, logvar)
    l2 = ((recon-x)**2).mean()
    ret = kl*cfg.kl_lambda + l2*cfg.l2_lambda
    return ret, { 'kl': kl, 'l2': l2 }
