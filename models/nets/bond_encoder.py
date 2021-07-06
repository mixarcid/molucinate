import torch
import torch.nn as nn

from .time_distributed import TimeDistributed
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorBonds, TMCfg
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
                nn.Linear(out_rnn_filters*TMCfg.max_valence, out_filters),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )

    def forward(self, x, bonds):
        rnn_out, _ = self.rnn(x)
        ba = bonds.bonded_atoms
        fc_in = torch.zeros((ba.size(0), ba.size(1), ba.size(-1), rnn_out.size(-1)), device=ba.device)
        for batch in range(ba.size(0)):
            fc_in[batch] = rnn_out[batch][ba[batch]]
        for batch in range(ba.size(0)):
            for atom in range(ba.size(1)):
                for bv in range(ba.size(2)):
                    fc_in[batch][atom][bv] = rnn_out[batch][ba[batch][atom][bv]]

        fc_in = fc_in.reshape((fc_in.size(0), fc_in.size(1), -1))
        
        return self.fc(fc_in)
