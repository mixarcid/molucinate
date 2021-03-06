import torch
import torch.nn as nn

from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *
from data.utils import *

class BasicEncoder(nn.Module):
    
    def __init__(self, hidden_size, cfg, gcfg):
        super(BasicEncoder, self).__init__()
        filter_list = [
            TMCfg.max_atoms,
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
        mul = get_linear_mul(filter_list)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(mul*filter_list[-1], hidden_size, bias=False),
            nn.LeakyReLU(LEAK_VALUE)
        )

    def forward(self, tmol, device):
        x = tmol.kps
        for conv in self.convs:
            x = conv(x)
        return self.fc(x)

class BasicDecoder(nn.Module):

    def __init__(self, latent_size, cfg, gcfg):
        super(BasicDecoder, self).__init__()
        filter_list = [
            cfg.init_filters*8,
            cfg.init_filters*4,
            cfg.init_filters*2,
            cfg.init_filters,
            cfg.init_filters
        ]
        width = get_final_width(filter_list)
        mul = get_linear_mul(filter_list)
        self.fc = nn.Sequential(
            nn.Linear(latent_size, filter_list[0]*mul),
            nn.LeakyReLU(LEAK_VALUE),
            Unflatten((filter_list[0], width, width, width))
        )
        self.convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            filt_next = filter_list[i+1]
            self.convs.append(nn.Sequential(
                upsample(filt, filt_next),
                conv3(filt_next, filt_next)
            ))
        self.final_conv = nn.Conv3d(filter_list[-1], NUM_ATOM_TYPES,
                                    kernel_size=1, bias=True)

    def forward(self, x, tmol, device):
        x = self.fc(x)
        for conv in self.convs:
            x = conv(x)
        x = self.final_conv(x)
        return TensorMol(molgrid=x)
