import torch
from torch.nn import functional as F

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(recon, x, mu, logvar, cfg):
    kl = kl_loss(mu, logvar)
    l2 = ((recon.molgrid-x.molgrid)**2).mean()
    ret = 0
    terms = {}
    if cfg.kl_lambda:
        ret += kl*cfg.kl_lambda
        terms['kl'] = kl
    if cfg.l2_lambda:
        ret += l2*cfg.l2_lambda
        terms['l2'] = l2
    return ret, terms
