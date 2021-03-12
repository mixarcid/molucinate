import torch
from torch.nn import functional as F

def kl_loss(recon, x, mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def l2_loss(recon, x, mu, logvar):
    return ((recon.molgrid-x.molgrid)**2).mean()

def combine_losses(loss_fns, cfg, *args):
    ret = 0
    terms = {}
    for fn in loss_fns:
        loss = fn(*args)
        name = fn.__name__.split('_')[0]
        lam = getattr(cfg, name + "_lambda")
        if lam:
            ret += lam*loss
            terms[name] = loss
    return ret, terms

def get_loss_fn(model_name, cfg):
    loss_fns = {
        'vae': [ kl_loss, l2_loss ]
    }[model_name]
    return lambda *args: combine_losses(loss_fns, cfg, *args)
