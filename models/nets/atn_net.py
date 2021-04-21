import torch
import torch.nn as nn

from .nn_utils import *
from .mg_decoder import MgDecoder
from .time_distributed import TimeDistributed
from .bond_attention import *
from .bond_predictor import BondPredictor

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

class AtnUpConv(nn.Module):

    def __init__(self, in_f, out_f):
        super().__init__()
        self.atn = BondAttentionFixed()
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
                          cfg.enc_rnn_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True,
                          bidirectional=True)

        self.flat = linear(cfg.enc_rnn_size*2, hidden_size)

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
        hidden = torch.cat((hidden[0], hidden[1]), 1)
        return self.flat(hidden)

class AtnNetDecoder(nn.Module):

    def __init__(self, latent_size, cfg, gcfg):
        super().__init__()
        self.lat_fc = nn.Sequential(
            nn.Linear(latent_size, cfg.dec_lat_fc_size, bias=False),
            nn.BatchNorm1d(cfg.dec_lat_fc_size),
            nn.LeakyReLU(LEAK_VALUE)
        )
        self.rnn = nn.GRU(cfg.dec_lat_fc_size,
                          cfg.dec_rnn_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True,
                          bidirectional=True)
        self.bond_pred = BondPredictor(cfg.dec_rnn_size*2,
                                       cfg.bond_pred_filters)
        self.initial_dec = AtnFlat(cfg.dec_rnn_size*2,
                                   cfg.initial_dec_size,
                                   BondAttentionFixed)
        self.atom_out = AtnFlat(cfg.initial_dec_size,
                                NUM_ATOM_TYPES,
                                BondAttentionFixed)
        filter_list = list(reversed(cfg.kp_filter_list))
        mul = get_linear_mul(filter_list)
        width = get_final_width(filter_list)
        self.kp_flat_dec = AtnFlat(cfg.initial_dec_size,
                                   mul*filter_list[0],
                                   BondAttentionFixed)
        self.kp_reshape = Unflatten((TMCfg.max_atoms, filter_list[0], width, width, width))
        self.kp_dec = IterativeSequential(
            AtnUpConv, filter_list
        )
        self.kp_out = TimeDistributed(
            nn.Conv3d(filter_list[-1], 1,
                      kernel_size=3, bias=False, padding=1),
            axis=2
        )

    def forward(self, z, tmol, device):

        if tmol is None:
            return TensorMol()
        
        rnn_in = self.lat_fc(z).unsqueeze(1).repeat(1, TMCfg.max_atoms, 1)
        dec, _ = self.rnn(rnn_in)

        bond_pred = self.bond_pred(dec)
        dec = self.initial_dec(dec, tmol.bonds)

        out_atom = self.atom_out(dec, tmol.bonds)
        
        kp_out = self.kp_flat_dec(dec, tmol.bonds)
        kp_out = self.kp_reshape(kp_out)
        kp_out = self.kp_dec(kp_out, tmol.bonds)
        kp_out = self.kp_out(kp_out).squeeze()

        return TensorMol(atom_types=out_atom,
                         kps_1h=kp_out,
                         bonds=tmol.bonds)
                          
        
