import torch
import torch.nn as nn

from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol
from data.chem import *
from data.utils import *

def get_final_width(filter_list, gcfg):
    return int(get_grid_size(gcfg.data)/(2**(len(filter_list)-1)))

def get_linear_mul(filter_list, gcfg):
    final_width = get_final_width(filter_list, gcfg)
    return (final_width**3)

class BasicEncoder(nn.Module):
    
    def __init__(self, hidden_size, cfg, gcfg):
        super(BasicEncoder, self).__init__()
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
            nn.Linear(mul*filter_list[-1], hidden_size),
            nn.LeakyReLU(LEAK_VALUE)
        )

    def forward(self, tmol):
        x = tmol.molgrid
        for conv in self.convs:
            x = conv(x)
        return self.fc(x)

class BasicDecoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super(BasicDecoder, self).__init__()
        filter_list = [
            cfg.init_filters*8,
            cfg.init_filters*4,
            cfg.init_filters*2,
            cfg.init_filters,
            NUM_ATOM_TYPES
        ]
        width = get_final_width(filter_list, gcfg)
        mul = get_linear_mul(filter_list, gcfg)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, filter_list[0]*mul),
            nn.LeakyReLU(LEAK_VALUE),
            Unflatten((filter_list[0], width, width, width))
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
