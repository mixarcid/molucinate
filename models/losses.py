import torch
from torch.nn import functional as F

import sys
sys.path.insert(0, '../..')
from data.chem import *
from data.tensor_mol import TMCfg

def one_hot(labels,
            num_classes,
            device = None,
            dtype = None,
            eps = 1e-6):

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def focal_loss(input, target, alpha = 1.0, gamma = 2.0, eps = 1e-8):

    n = input.size(0)
    out_size = (n,) + input.size()[2:]

    # compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    
    loss = torch.mean(loss_tmp)
    
    return loss


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

def bond_loss(recon, x, mu, logvar, fn):
    idxs = (x.atom_types != ATOM_TYPE_HASH["_"]).float()
    recon_bonds = recon.bonds.data
    x_bonds = x.bonds.data
    recon_bonds = recon_bonds.permute(0, 2, 3, 1)#.contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = x.bonds.data.permute(0, 2, 3, 1)#.contiguous().view(-1, NUM_BOND_TYPES)
    multi_idxs = torch.bmm(idxs.unsqueeze(-1), idxs.unsqueeze(-2)).bool()
    for i in range(TMCfg.max_atoms):
        multi_idxs[:,i,i] = False
    multi_idxs = multi_idxs.unsqueeze(1).expand(-1, NUM_BOND_TYPES, -1, -1).permute(0, 2, 3, 1)
    recon_bonds = recon_bonds[multi_idxs].contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = x_bonds[multi_idxs].contiguous().view(-1, NUM_BOND_TYPES)
    x_bonds = torch.argmax(x_bonds, -1)
    return fn(
        recon_bonds,
        x_bonds
    )
    
def bond_ce_loss(recon, x, mu, logvar):
    return bond_loss(recon, x, mu, logvar, F.cross_entropy)

def bond_focal_loss(recon, x, mu, logvar):
    return bond_loss(recon, x, mu, logvar, focal_loss)

def bv_ce_loss(recon, x, mu, logvar):
    idxs = x.atom_types != ATOM_TYPE_HASH["_"]
    recon = recon.bonds.data[idxs]
    x = x.bonds.data[idxs]
    return F.cross_entropy(
        recon.contiguous().view(-1, recon.size(-1)),
        x.contiguous().view(-1),
    )

def bv_focal_loss(recon, x, mu, logvar):
    idxs = x.atom_types != ATOM_TYPE_HASH["_"]
    recon = recon.bonds.data[idxs]
    x = x.bonds.data[idxs]
    return focal_loss(
        recon.contiguous().view(-1, recon.size(-1)),
        x.contiguous().view(-1),
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
        'vae': [ atom_ce_loss, kp_ce_loss, kl_loss, bv_ce_loss, bv_focal_loss ]
    }[model_name]
    return lambda *args: combine_losses(loss_fns, cfg, *args)
