import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')
from data.chem import *

class BondAttentionFixed(nn.Module):

    def __init__(self, one_way):
        super().__init__()
        self.one_way = one_way

    def forward(self, x, bonds, is_padded=False):
        idxs = bonds.get_all_indexes()
        return torch.cat([x,x], 2)
        out = torch.zeros_like(x)
        for idxs in bonds.get_all_indexes():
            batch, start, end, bond = idxs
            if is_padded:
                start += 1
                end += 1
            if end >= bonds.bond_types.size(1): continue
            out[batch, end] += x[batch, start]
            if not self.one_way:
                out[batch, start] += x[batch, end]
        return torch.cat([out, x], 2)
        
