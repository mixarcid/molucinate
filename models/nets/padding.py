import torch

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

def get_padded_atypes(tmol, device, batch_size, truncate):
    start_idx = torch.tensor([[ATOM_TYPE_HASH['^']]]*batch_size,
                             device=device,
                             dtype=torch.long)
    if len(tmol.atom_types.shape) == 1:
        return start_idx
    padded = torch.cat((start_idx, tmol.atom_types), 1)
    if truncate:
        return padded[:,:-1]
    return padded

def get_padded_valences(tmol, device, batch_size, truncate):
    start_idx = torch.tensor([[[TMCfg.max_valence-1]*NUM_ACT_BOND_TYPES]]*batch_size,
                             device=device,
                             dtype=torch.long)
    if len(tmol.atom_types.shape) == 1:
        return start_idx
    padded = torch.cat((start_idx, tmol.atom_valences), 1)
    if truncate:
        return padded[:,:-1]
    return padded

def get_padded_kps(tmol, device, batch_size, truncate):
    sz = TMCfg.grid_size
    start = torch.ones((batch_size, 1, sz, sz, sz), device=device)
    if len(tmol.atom_types.shape) == 1:
        return torch.unsqueeze(start, 2)
    padded = torch.unsqueeze(torch.cat((start, tmol.kps), 1), 2)
    if truncate:
        return padded[:,:-1]
    return padded
