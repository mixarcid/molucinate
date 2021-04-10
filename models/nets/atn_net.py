import torch
import torch.nn as nn

from .nn_utils import *
from .mg_decoder import MgDecoder
from .time_distributed import TimeDistributed
from .bond_attention import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class AtnDownConv(nn.Module):

    def __init__(self, in_f, out_f):
        super().__init__()
        self.atn = BondAttentionFixed()
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
        

class AtnNetEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, cfg.atom_embed_size)
        self.atom_enc = AtnFlat(cfg.atom_embed_size,
                                cfg.atom_enc_size,
                                BondAttentionFixed)

        self.kp_init_enc = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(1, cfg.kp_filter_list[0],
                          kernel_size=3, bias=False, padding=1),
                nn.BatchNorm3d(cfg.kp_filter_list[0]),
                nn.LeakyReLU(LEAK_VALUE)
            ),
            axis=2
        )
        self.kp_enc = IterativeSequential(
            AtnDownConv, cfg.kp_filter_list
        )
        mul = get_linear_mul(cfg.kp_filter_list)
        self.kp_flat_enc = AtnFlat(mul*cfg.kp_filter_list[-1],
                                   cfg.kp_enc_size,
                                   BondAttentionFixed)

        self.final_enc = AtnFlat(cfg.atom_enc_size + cfg.kp_enc_size,
                                 cfg.final_enc_size,
                                 BondAttentionFixed)

        self.rnn = nn.GRU(cfg.final_enc_size,
                          hidden_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True)

    def forward(self, tmol, device):
        aenc = self.atom_embed(tmol.atom_types)
        aenc = self.atom_enc(aenc, tmol.bonds)

        kpenc = torch.unsqueeze(tmol.kps, 2)
        kpenc = self.kp_init_enc(kpenc)
        kpenc = self.kp_enc(kpenc, tmol.bonds)
        kpenc = kpenc.contiguous().view(kpenc.size(0), kpenc.size(1), -1)
        kpenc = self.kp_flat_enc(kpenc, tmol.bonds)

        enc = torch.cat((aenc, kpenc), 2)
        enc = self.final_enc(enc, tmol.bonds)
        _, hidden = self.rnn(enc)
        return hidden[0]
        
