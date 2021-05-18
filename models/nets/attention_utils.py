import torch
import torch.nn as nn

from .nn_utils import *
from .time_distributed import TimeDistributed
from .bond_attention import *

class AtnDownConv(nn.Module):

    def __init__(self, in_f, out_f, one_way):
        super().__init__()
        self.atn = BondAttentionFixed(one_way)
        self.conv = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(in_f, out_f, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm3d(out_f),
                nn.LeakyReLU(LEAK_VALUE),
                downsample(),
            ),
            axis=2
        )

    def forward(self, x, bonds):
        x = self.atn(x, bonds)
        return self.conv(x)

class AtnUpConv(nn.Module):

    def __init__(self, in_f, out_f, one_way):
        super().__init__()
        self.atn = BondAttentionFixed(one_way)
        self.conv = TimeDistributed(
            nn.Sequential(
                upsample(in_f, out_f),
                nn.Conv3d(out_f, out_f, kernel_size=3, bias=False, padding=1),
                nn.BatchNorm3d(out_f),
                nn.LeakyReLU(LEAK_VALUE),
            ),
            axis=2
        )

    def forward(self, x, bonds):
        x = self.atn(x, bonds)
        return self.conv(x)

class AtnFlat(nn.Module):

    def __init__(self, in_filters, out_filters, atn_cls, *args):
        super().__init__()
        self.atn = atn_cls(*args)
        self.linear = TimeDistributed(
            nn.Sequential(
                nn.Linear(in_filters, out_filters, bias=False),
                nn.BatchNorm1d(out_filters),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )

    def forward(self, x, *args):
        x = self.atn(x, *args)
        return self.linear(x)
