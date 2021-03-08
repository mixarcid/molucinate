import torch
import torch.nn as nn

from .nn_utils import *

class MsConv(nn.Module):

    def __init__(self,
                 in_filter_list,
                 out_filter_list,
                 in_flat_size,
                 out_flat_size,
                 aux_filter_list=None):
        super().__init__()

        width = get_final_width(in_filter_list)
        mul = get_linear_mul(in_filter_list)
        lin_conv_size = mul*in_filter_list[-1]
        hidden_size = lin_conv_size + in_flat_size
        
        self.in_convs = nn.ModuleList()
        for i, filt in enumerate(in_filter_list[:-1]):
            filt2 = aux_filter_list[i] if aux_filter_list is not None else 0
            self.in_convs.append(nn.Sequential(
                conv3(filt + filt2, in_filter_list[i+1]),
                downsample()
            ))
        
        self.flatten = Flatten()
        self.fc_flat = linear(hidden_size, out_flat_size)
        if len(out_filter_list) > 0:
            self.fc_conv = nn.Sequential(
                linear(hidden_size, lin_conv_size),
                Unflatten((out_filter_list[0], width, width, width))
            )
        else:
            self.fc_conv = None
        
        self.out_convs = nn.ModuleList()
        self.out_ups = nn.ModuleList()
        for i, filt in enumerate(out_filter_list[:-1]):
            filt_next = out_filter_list[i+1]
            filt_in = in_filter_list[-i-2]
            self.out_ups.append(upsample(filt, filt_next))
            self.out_convs.append(conv3(filt_next + filt_in, filt_next))

    def forward(self, grid, flat, aux_grids=None):
        in_grids = []
        for i, conv in enumerate(self.in_convs):
            in_grids.append(grid)
            if aux_grids is not None:
                grid = torch.cat((grid, aux_grids[i]), 1)
            grid = conv(grid)
        fgrid = self.flatten(grid)
        hidden = torch.cat((fgrid, flat), 1)
        out_flat = self.fc_flat(hidden)

        out_grids = []
        if self.fc_conv is not None:
            out_grid = self.fc_conv(hidden)
            for up, conv, igrid in zip(self.out_ups,
                                       self.out_convs,
                                       reversed(in_grids)):
                out_grid = up(out_grid)
                out_grid = torch.cat((out_grid, igrid), 1)
                out_grid = conv(out_grid)
                out_grids.append(out_grid)
        else:
            out_grid = None
        return out_grid, out_flat, in_grids, out_grids
