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
        recon.contiguous().view(-1, recon.size(-1)),
        x.contiguous().view(-1),
        #ignore_index=ATOM_TYPE_HASH["_"]
    )

def valence_ce_loss(recon, x, mu, logvar):
    idxs = x.atom_types != ATOM_TYPE_HASH["_"]
    recon = recon.atom_valences[idxs]
    x = x.atom_valences[idxs]
    return F.cross_entropy(
        recon.contiguous().view(-1, recon.size(-1)),
        x.contiguous().view(-1),
    )

def kp_ce_loss(recon, x, mu, logvar):
    idxs = x.atom_types != ATOM_TYPE_HASH["_"]
    x = torch.unsqueeze(x.kps_1h[idxs], 0)
    x = torch.argmax(x.contiguous().view(x.size(0), x.size(1), -1), -1)
    recon = torch.unsqueeze(recon.kps_1h[idxs], 0)
    return F.cross_entropy(
        recon.contiguous().view(recon.size(0)*recon.size(1), -1),
        x.contiguous().view(-1)
    )

def bond_ce_loss(recon, x, mu, logvar):
    idxs = (x.atom_types != ATOM_TYPE_HASH["_"]).float()
    recon_bonds = recon.bonds.data
    x_bonds = x.bonds.data
    recon_bonds = recon_bonds.permute(0, 2, 3, 1)#.contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = x.bonds.data.permute(0, 2, 3, 1)#.contiguous().view(-1, NUM_BOND_TYPES)
    multi_idxs = torch.bmm(idxs.unsqueeze(-1), idxs.unsqueeze(-2)).bool().unsqueeze(1).expand(-1, NUM_BOND_TYPES, -1, -1).permute(0, 2, 3, 1)
    recon_bonds = recon_bonds[multi_idxs].contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = x_bonds[multi_idxs].contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = torch.argmax(x_bonds, -1)
    return F.cross_entropy(
        recon_bonds,
        x_bonds
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
        'vae': [ atom_ce_loss, valence_ce_loss, kp_ce_loss, kl_loss, bond_ce_loss ]
    }[model_name]
    return lambda *args: combine_losses(loss_fns, cfg, *args)
