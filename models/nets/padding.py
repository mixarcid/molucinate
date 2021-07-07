import torch

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TensorBonds, TMCfg
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

def get_padded_bonds(tmol, device, batch_size, truncate):
    start_idx = torch.tensor([[[0]*TMCfg.max_valence]]*batch_size,
                             device=device,
                             dtype=torch.long)
    val_start_idx = torch.tensor([[[0]*NUM_ACT_BOND_TYPES]]*batch_size,
                             device=device,
                             dtype=torch.long)
    if len(tmol.atom_types.shape) == 1:
        return TensorBonds(None, start_idx, start_idx, val_start_idx)
    padded_types = torch.cat((start_idx, tmol.bonds.bond_types), 1)
    padded_atoms = torch.cat((start_idx, tmol.bonds.bonded_atoms), 1)
    padded_valences = torch.cat((val_start_idx, tmol.bonds.atom_valences), 1)
    if truncate:
        return TensorBonds(None, padded_types[:,:-1], padded_atoms[:,:-1], padded_valences[:,:-1])
    return TensorBonds(None, padded_types, padded_atoms, padded_valences)

def get_padded_valences(tmol, device, batch_size, truncate):
    start_idx = torch.tensor([[[TMCfg.max_valence-1]*NUM_ACT_BOND_TYPES]]*batch_size,
                             device=device,
                             dtype=torch.long)
    if len(tmol.atom_types.shape) == 1:
        return start_idx
    padded = torch.cat((start_idx, tmol.bonds.atom_valences), 1)
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
