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
