import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TMCfg

LEAK_VALUE = 0.1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)

def linear(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False),
        nn.BatchNorm1d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )

def conv3(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, kernel_size=3, bias=False, padding=1),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )

def conv1(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, kernel_size=1, bias=False),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )
    
def downsample():
    return nn.MaxPool3d(2)

def upsample(in_f, out_f):
    return nn.Sequential(
        nn.ConvTranspose3d(in_f, out_f, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )

def get_final_width(filter_list):
    return int(TMCfg.grid_size/(2**(len(filter_list)-1)))

def get_linear_mul(filter_list):
    final_width = get_final_width(filter_list)
    return (final_width**3)
