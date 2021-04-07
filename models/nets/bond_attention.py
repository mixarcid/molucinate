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
    print(out[0,1])
    return out.contiguous().view(*atn_outputs.shape[:2],
                                 NUM_ACT_BOND_TYPES*atn_outputs.shape[2],
                                 *atn_outputs.shape[3:])

def bond_attention_given(atn_outputs, bonds):
    out = atn_outputs#torch.zeros_like(atn_outputs)
    for batch, start, end, bond in bonds.all_indexes:
        out[batch, end] += atn_outputs[batch, start]
    return out
