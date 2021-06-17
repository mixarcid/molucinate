import torch
import torch.nn as nn

from .time_distributed import TimeDistributed
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorBonds, TensorBondsValence, TMCfg
from data.chem import *

class BondPredictor(nn.Module):

    def __init__(self, in_filters, hid_filters):
        super().__init__()
        self.fcs = nn.ModuleList()
        for n in range(NUM_BOND_TYPES):
            self.fcs.append(TimeDistributed(
                nn.Sequential(
                    nn.Linear(in_filters, hid_filters),#, bias=False),
                    nn.BatchNorm1d(hid_filters),
                    nn.LeakyReLU(LEAK_VALUE)
                ),
                axis=2
            ))

    def forward(self, x):
        fc_outs = []
        for fc in self.fcs:
            fc_outs.append(fc(x))
        fc_outs = torch.stack(fc_outs, 1)
        bdata = torch.einsum('bfij,bfkj->bfik', fc_outs, fc_outs)
        #bdata = torch.cat((bdata, torch.zeros_like(bdata[:,:1])), 1)
        return TensorBonds(data=bdata)

class MultiHeadedBondPredictor(nn.Module):

    def __init__(self, in_filters, hid_filters, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hid_filters = hid_filters
        self.encoder = TimeDistributed(
            nn.Sequential(
                nn.Linear(in_filters, num_heads*hid_filters*2),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )
        self.decoder = nn.Linear(num_heads, NUM_BOND_TYPES)

    def forward(self, x):
        batch = x.size(0)
        atoms = x.size(1)
        enc = self.encoder(x)
        enc = enc.reshape((batch, atoms, 2, self.hid_filters, self.num_heads))
        mat =  torch.einsum('bihn,bjhn->bijn', enc[:,:,0], enc[:,:,1])
        dec = self.decoder(mat.reshape(-1, self.num_heads))
        bdata = dec.reshape((batch, atoms, atoms, NUM_BOND_TYPES)).permute(0, 3, 1, 2)
        return TensorBonds(data=bdata)

class BondValencePredictor(nn.Module):

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
                nn.Linear(in_filters, hid_filters*NUM_ACT_BOND_TYPES*TMCfg.max_valence),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )

    def forward(self, x):
        batch = x.size(0)
        atoms = x.size(1)
        qshape = (batch, atoms, self.hid_filters, NUM_ACT_BOND_TYPES, TMCfg.max_valence)
        keys = self.key_enc(x)
        queries = self.query_enc(x).reshape(qshape)
        #bdata = torch.einsum('abc,afcde->afdeb', keys, queries)
        bdata = torch.einsum('abc,afcde->abdef', keys, queries)
        mask = torch.ones((atoms, atoms), dtype=bool, device=x.device)
        mask = torch.triu(mask, 1).reshape((1, atoms, 1, 1, atoms))
        mask = mask.repeat(batch, 1, NUM_ACT_BOND_TYPES, TMCfg.max_valence, 1)
        #ret = torch.zeros_like(bdata)
        #ret[mask] += bdata[mask]
        bdata[mask] = float('-inf')
        return TensorBondsValence(data=bdata)
        
