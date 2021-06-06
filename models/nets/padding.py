import torch

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

def get_padded_atypes(tmol, device):
    batch_size = tmol.atom_types.shape[0]
    start_idx = torch.tensor([[ATOM_TYPE_HASH['^']]]*batch_size,
                             device=device,
                             dtype=torch.long)
    padded = torch.cat((start_idx, tmol.atom_types), 1)[:,:-1]
    return padded

def get_padded_kps(tmol, device):
    batch_size = tmol.atom_types.shape[0]
    sz = TMCfg.grid_size
    start = torch.ones((batch_size, 1, sz, sz, sz), device=device)
    padded = torch.unsqueeze(torch.cat((start, tmol.kps), 1), 2)[:,:-1]
    return padded
