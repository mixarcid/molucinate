import torch
import torch.nn as nn

from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class MgDecoder(nn.Module):

    def __init__(self, latent_size, cfg, gcfg):
        super().__init__()

        filter_list = [
            512,
            256,
            256,
            64,
            32,
            32,
        ]
        width = get_final_width(filter_list)
        mul = get_linear_mul(filter_list)
        self.mg_fc = nn.Sequential(
            nn.Linear(latent_size, filter_list[0]*mul),
            nn.LeakyReLU(LEAK_VALUE),
            Unflatten((filter_list[0], width, width, width))
        )
        self.mg_convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            filt_next = filter_list[i+1]
            self.mg_convs.append(nn.Sequential(
                upsample(filt, filt_next),
                conv3(filt_next, filt_next)
            ))
        self.final_mg_conv = nn.Conv3d(filter_list[-1], NUM_ATOM_TYPES,
                                    kernel_size=1, bias=True)


    def forward(self, z, device):
        x = self.mg_fc(z)
        for conv in self.mg_convs:
            x = conv(x)
        x = self.final_mg_conv(x)
        return x
        
