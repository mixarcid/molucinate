import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')
from data.chem import *

def get_bond_attention_size_btypes(atn_sz):
    return atn_sz*NUM_ACT_BOND_TYPES

def get_bond_attention_size(atn_sz):
    return atn_sz

def bond_attention_given_btypes(atn_outputs, bonds):
    out = torch.zeros(*atn_outputs.shape[:2],
                      NUM_ACT_BOND_TYPES,
                      *atn_outputs.shape[2:])
    for batch, start, end, bond in bonds.get_all_indexes():
        print(start, end)
        out[batch, end, bond-1] += atn_outputs[batch, start]
    return out.contiguous().view(*atn_outputs.shape[:2],
                                 NUM_ACT_BOND_TYPES*atn_outputs.shape[2],
                                 *atn_outputs.shape[3:])

def bond_attention_given(atn_outputs, bonds):
    out = torch.zeros_like(atn_outputs)
    out += atn_outputs
    for batch, start, end, bond in bonds.get_all_indexes():
        out[batch, end] += atn_outputs[batch, start]
    return out

class BondAttentionFixed(nn.Module):

    def __init__(self, one_way):
        super().__init__()
        self.one_way = one_way

    def forward(self, x, bonds):
        out = torch.zeros_like(x)
        for batch, start, end, bond in bonds.get_all_indexes():
            out[batch, end] += x[batch, start]
            if not self.one_way:
                out[batch, start] += x[batch, end]
        print(out.shape, torch.cat([out, x], 2).shape)
        return torch.cat([out, x], 2)
        
