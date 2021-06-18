import torch
import torch.nn as nn

from .time_distributed import TimeDistributed
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorBonds, TensorBondsValence, TMCfg
from data.chem import *

class BondEncoder(nn.Module):

    def __init__(self, in_filters, out_rnn_filters, out_filters):
        super().__init__()
        self.rnn = nn.GRU(in_filters,
                          out_rnn_filters,
                          num_layers=1,
                          batch_first=True)
        self.fc = TimeDistributed(
            nn.Sequential(
                nn.Linear(out_rnn_filters*NUM_ACT_BOND_TYPES*TMCfg.max_valence, out_filters),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )

    def forward(self, x, bonds):
        rnn_out, _ = self.rnn(x)
        bflat = bonds.data.reshape(
            (bonds.data.size(0),
             bonds.data.size(1),
             -1))
        fc_in = torch.zeros((bflat.size(0), bflat.size(1), bflat.size(-1), rnn_out.size(-1)), device=bflat.device)
        for batch in range(bflat.size(0)):
            fc_in[batch] = rnn_out[batch][bflat[batch]]
        #for batch in range(bflat.size(0)):
        #    for atom in range(bflat.size(1)):
        #        for bv in range(bflat.size(2)):
        #            fc_in[batch][atom][bv] = rnn_out[batch][bflat[batch][atom][bv]]

        fc_in = fc_in.reshape((fc_in.size(0), fc_in.size(1), -1))
        
        return self.fc(fc_in)
