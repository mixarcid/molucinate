import torch
import torch.nn as nn

from .nn_utils import *

class MSConv(nn.Module):

    def __init__(self, filter_list, in_flat_size, out_flat_size):
        super().__init__()
        self.in_convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            self.in_convs.append(nn.Sequential(
                conv3(filt, filter_list[i+1]),
                downsample()
            ))
            
        width = get_final_width(filter_list)
        mul = get_linear_mul(filter_list)
        lin_conv_size = mul*filter_list[-1]
        hidden_size = lin_conv_size + in_flat_size
        out_filter_list = list(reversed(filter_list))
        
        self.flatten = Flatten()
        self.fc_flat = linear(hidden_size, out_flat_size)
        self.fc_conv = nn.Sequential(
            linear(hidden_size, lin_conv_size),
            Unflatten((out_filter_list[0], width, width, width))
        )
        
        self.out_convs = nn.ModuleList()
        for i, filt in enumerate(out_filter_list[:-1]):
            filt_next = out_filter_list[i+1]
            self.out_convs.append(nn.Sequential(
                upsample(filt, filt_next),
                conv3(filt_next, filt_next)
            ))

    def forward(self, grid, flat):
        for conv in self.in_convs:
            grid = conv(grid)
        fgrid = self.flatten(grid)
        hidden = torch.cat((fgrid, flat), 1)
        out_flat = self.fc_flat(hidden)
        out_grid = self.fc_conv(hidden)
        for conv in self.out_convs:
            out_grid = conv(out_grid)
        return out_grid, out_flat
