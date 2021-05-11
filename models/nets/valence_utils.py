import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TensorBonds, TMCfg
from data.chem import *

class ValenceEmbedding(nn.Module):

    def __init__(self, embed_size):
        super().__init__()
        self.embed = nn.Embedding(TMCfg.max_valence**NUM_ACT_BOND_TYPES, embed_size)

    def forward(self, valences, device):
        mult = torch.tensor([TMCfg.max_valence**i for i in range(NUM_ACT_BOND_TYPES)], device=device)
        to_embed = torch.einsum('bav,v->ba', valences, mult)
        return self.embed(to_embed)
        

class ValenceDecoder(nn.Module):

    def __init__(self, cls, hidden_size, *args):
        super().__init__()
        self.fcs = nn.ModuleList()
        for i in range(NUM_ACT_BOND_TYPES):
            self.fcs.append(cls(hidden_size,
                                TMCfg.max_valence,
                                *args))

    def forward(self, *args):
        outs = []
        for i in range(NUM_ACT_BOND_TYPES):
            outs.append(torch.unsqueeze(self.fcs[i](*args), -2))
        return torch.cat(outs, -2)
