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

class IterativeSequential(nn.Module):

    def __init__(self, cls, filter_list, *args):
        super().__init__()
        self.mod_list = nn.ModuleList()
        for f, f_next in zip(filter_list, filter_list[1:]):
            self.mod_list.append(cls(f, f_next, *args))

    def forward(self, x, *args):
        for mod in self.mod_list:
            x = mod(x, *args)
        return x

def linear(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False),
        #nn.BatchNorm1d(out_f),
        nn.LayerNorm(out_f),
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

def get_final_width_len(length):
    return int(TMCfg.grid_size/(2**(length-1)))

def get_final_width(filter_list):
    return get_final_width_len(len(filter_list))

def get_linear_mul(filter_list):
    final_width = get_final_width(filter_list)
    return (final_width**3)
