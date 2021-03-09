import torch
import torch.nn as nn

from .ms_conv import MsConv
from .nn_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TMCfg


class MsRnn(nn.Module):

    def __init__(self,
                 in_filter_list,
                 dec_filter_list,
                 hid_flat_size,
                 ih_conv_size,
                 ih_flat_size,
                 i_flat_size):
        super().__init__()

        self.filt_len = len(in_filter_list)
        self.filt_len_diff = len(in_filter_list) - len(dec_filter_list)

        self.init_hid_convs = nn.ParameterList()
        self.out_filter_list = []
        for i, filt in enumerate(in_filter_list):
            if i < self.filt_len_diff:
                hid_size = filt
            else:
                hid_size = dec_filter_list[self.filt_len_diff - i]
            self.out_filter_list.append(hid_size)
            sz = get_final_width_len(i+1)
            hid_shape = (hid_size, sz, sz, sz)
            self.init_hid_convs.append(nn.Parameter(torch.zeros(hid_shape)))
        
        self.init_hid_flat = nn.Parameter(torch.zeros(hid_flat_size))
        self.init_conv = conv3(1, ih_conv_size)
        self.init_flat = linear(i_flat_size, ih_flat_size)
        self.rnn = MsConv(in_filter_list,
                          dec_filter_list,
                          ih_flat_size + hid_flat_size,
                          hid_flat_size,
                          self.out_filter_list)


    def forward(self, grid, flat):
        assert(grid.shape[1] == flat.shape[1])
        batch_sz = flat.shape[0]
        hid_convs = []
        for ihc in self.init_hid_convs:
            hid_convs.append(ihc.repeat(batch_sz, 1, 1, 1, 1))
        hid_flat = self.init_hid_flat.repeat(batch_sz, 1)
        for i in range(grid.shape[1]):
            grid_in = self.init_conv(grid[:,i:i+1])
            flat_in1 = self.init_flat(flat[:,i])
            flat_in = torch.cat((hid_flat, flat_in1), 1)
            _, hid_flat, in_grids, out_grids = self.rnn(grid_in, flat_in, hid_convs)
            hid_convs = in_grids
            for i, out_grid in enumerate(out_grids):
                hid_convs[-i-1] = out_grid
            
        return hid_flat, hid_convs
            
