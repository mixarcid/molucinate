import torch
import torch.nn as nn

from .time_distributed import TimeDistributed
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorBonds, TMCfg
from data.chem import *

class BondPredictor(nn.Module):

    def __init__(self, in_filters, hid_filters):
        super().__init__()
        self.in_filters = in_filters
        self.hid_filters = hid_filters
        self.key_enc = TimeDistributed(
            nn.Sequential(
                nn.Linear(in_filters, hid_filters),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )
        self.query_enc = TimeDistributed(
            nn.Sequential(
                nn.Linear(in_filters, hid_filters*TMCfg.max_valence),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )
        self.type_out = TimeDistributed(
            nn.Linear(in_filters, NUM_BOND_TYPES*TMCfg.max_valence),
            axis=2
        )

    def forward(self, x):
        batch = x.size(0)
        atoms = x.size(1)
        qshape = (batch, atoms, self.hid_filters, TMCfg.max_valence)
        keys = self.key_enc(x)
        queries = self.query_enc(x).reshape(qshape)
        bdata = torch.einsum('abc,afcd->afdb', keys, queries)
        #bdata = torch.einsum('abc,afcde->abdef', keys, queries)
        mask = torch.ones((atoms, atoms), dtype=bool, device=x.device)
        mask = torch.triu(mask).reshape((1, atoms, 1, atoms))
        mask = mask.repeat(batch, 1, TMCfg.max_valence, 1)
        bdata[mask] = float('-inf')
        types = self.type_out(x).reshape((batch, atoms, TMCfg.max_valence, NUM_BOND_TYPES))
        return TensorBonds(bond_types=types, bonded_atoms=bdata)
        
