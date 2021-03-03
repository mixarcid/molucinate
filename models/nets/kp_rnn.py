import torch
import torch.nn as nn

from .ms_conv import MSConv
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class KpRnnEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()
        filter_list = [
            cfg.init_filters,
            cfg.init_filters*2,
            cfg.init_filters*4,
            cfg.init_filters*8
        ]
        flat_in_size = TMCfg.max_atoms*NUM_ATOM_TYPES
        self.init_conv = conv3(TMCfg.max_atoms, filter_list[0])
        self.init_flat = Flatten()
        self.ms_conv = MSConv(filter_list, flat_in_size, hidden_size)

    def forward(self, tmol):
        x_grid = tmol.kps
        x_flat = self.init_flat(tmol.atom_types)
        x_grid = self.init_conv(x_grid)
        out_grid, out_flat = self.ms_conv(x_grid, x_flat)
        return out_flat
