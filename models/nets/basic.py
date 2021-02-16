import torch
import torch.nn as nn

from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol
from data.chem import *
from data.utils import *

def get_linear_mul(filter_list, gcfg):
    final_width = int(get_grid_size(gcfg)/(2**(len(filter_list)-1)))
    return (final_width**3)

class BasicEncoder(nn.Module):
    
    def __init__(self, hidden_size, cfg, gcfg):
        filter_list = [
            NUM_ATOM_TYPES,
            cfg.init_filters,
            cfg.init_filters*2,
            cfg.init_filters*4,
            cfg.init_filters*8
        ]
        self.convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            self.convs.append(nn.Sequential(
                conv3(filt, filter_list[i+1]),
                downsample()
            ))
        mul = get_linear_mul(filter_list, gcfg)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(linear_mul*filter_list[-1], hidden_size),
            nn.LeakyReLU(LEAK_VALUE)
        )

    def forward(self, tmol):
        x = tmol.molgrid
        for conv in self.convs:
            x = conv(x)
        return self.fc(x)

class BasicDecoder(nn.Module):

        def __init__(self, hidden_size, cfg, gcfg):
        filter_list = [
            cfg.init_filters*8,
            cfg.init_filters*4,
            cfg.init_filters*2,
            cfg.init_filters,
            NUM_ATOM_TYPES
        ]
        mul = get_linear_mul(filter_list, gcfg)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, filter_list[0]*mul),
            nn.LeakyReLU(LEAK_VALUE),
            Unflatten()
        )
        self.convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            self.convs.append(nn.Sequential(
                conv3(filt, filter_list[i+1]),
            ))

    def forward(self, x):
        x = self.fc(x)
        for conv in self.convs:
            x = conv(x)
        return TensorMol(molgrid=x)
