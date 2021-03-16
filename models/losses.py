import torch
from torch.nn import functional as F

import sys
sys.path.insert(0, '../..')
from data.chem import *

def kl_loss(recon, x, mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def l2_loss(recon, x, mu, logvar):
    return ((recon.molgrid-x.molgrid)**2).mean()

def atom_ce_loss(recon, x, mu, logvar):
    recon = recon.atom_types
    x = x.atom_types
    return F.cross_entropy(
        recon[:, :-1].contiguous().view(-1, recon.size(-1)),
        x[:, 1:].contiguous().view(-1),
        ignore_index=ATOM_TYPE_HASH["_"]
    )

def combine_losses(loss_fns, cfg, *args):
    ret = 0
    terms = {}
    for fn in loss_fns:
        loss = fn(*args)
        name = '_'.join(fn.__name__.split('_')[:-1])
        lam = getattr(cfg, name + "_lambda")
        if lam:
            ret += lam*loss
            terms[name] = loss
    return ret, terms

def get_loss_fn(model_name, cfg):
    loss_fns = {
        'vae': [ atom_ce_loss ]
    }[model_name]
    return lambda *args: combine_losses(loss_fns, cfg, *args)
