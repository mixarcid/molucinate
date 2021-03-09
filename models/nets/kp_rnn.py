import torch
import torch.nn as nn

from .ms_conv import MsConv
from .ms_rnn import MsRnn
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class KpRnnEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()
        hid_filter_list = [
            cfg.init_filters,
            cfg.init_filters*2,
            cfg.init_filters*4,
            cfg.init_filters*8,
            cfg.init_filters*8,
            cfg.init_filters*8
        ]
        enc_filter_list = [
            0,
            32,
            64,
            64,
            128,
            128
        ]
        flat_in_size = NUM_ATOM_TYPES
        """self.init_flat = Flatten()
        self.ms_conv = MsConv(filter_list,
                              TMCfg.max_atoms,
                              1,
                              flat_in_size,
                              hidden_size)"""
        ih_flat_size = 32
        hid_flat_size = 64
        self.ms_rnn = MsRnn(hid_filter_list,
                            [],
                            hid_flat_size,
                            cfg.init_filters,
                            ih_flat_size,
                            flat_in_size)
        self.out = MsConv(enc_filter_list,
                          [],
                          hid_flat_size,
                          hidden_size,
                          self.ms_rnn.out_filter_list)
                          

    def forward(self, tmol):
        x_grid = tmol.kps
        """x_flat = self.init_flat(tmol.atom_types)
        out_grid, out_flat = self.ms_conv(x_grid, x_flat)
        return out_flat"""
        x_flat = tmol.atom_types
        hid_flat, hid_convs =  self.ms_rnn(x_grid, x_flat)
        grid, out, in_grids, out_grids = self.out(None, hid_flat, hid_convs)
        return out
