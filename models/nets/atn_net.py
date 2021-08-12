import torch
import torch.nn as nn

from .nn_utils import *
from .mg_decoder import MgDecoder
from .time_distributed import TimeDistributed
from .bond_attention import *
from .bond_predictor import BondPredictor
from .attention_utils import *
from .valence_utils import *
from .bond_encoder import BondEncoder

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class AtnNetEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, cfg.atom_embed_size)
        self.atom_enc = AtnFlat(cfg.atom_embed_size,
                                cfg.atom_enc_size,
                                BondAttentionFixed,
                                False)

        self.predict_valence = gcfg.predict_valence
        if self.predict_valence:
            self.valence_embed = ValenceEmbedding(cfg.valence_embed_size)
            self.valence_enc = AtnFlat(cfg.valence_embed_size,
                                       cfg.valence_enc_size,
                                       BondAttentionFixed,
                                       False)


        self.use_kps = gcfg.data.use_kps
        if self.use_kps:
            self.kp_init_enc = TimeDistributed(
                nn.Sequential(
                    downsample(),
                    nn.Conv3d(1, cfg.kp_filter_list[1],
                              kernel_size=cfg.kernel_size, bias=False, padding=cfg.padding),
                    nn.BatchNorm3d(cfg.kp_filter_list[1]),
                    nn.LeakyReLU(LEAK_VALUE)
                ),
                axis=2
            )
            self.kp_enc = IterativeSequential(
                AtnDownConv, cfg.kp_filter_list[1:], cfg.kernel_size, cfg.padding, False
            )
            mul = get_linear_mul(cfg.kp_filter_list)
            self.kp_flat_enc = AtnFlat(mul*cfg.kp_filter_list[-1],
                                       cfg.kp_enc_size,
                                       BondAttentionFixed, False)
            kp_enc_size = cfg.kp_enc_size
        else:
            kp_enc_size = 0

        self.has_bond_type_enc = cfg.bond_embed_size > 0
        if self.has_bond_type_enc:
            self.bond_type_enc = nn.Embedding(NUM_BOND_TYPES, cfg.bond_embed_size)

        final_enc_size = cfg.atom_enc_size + kp_enc_size + cfg.bond_embed_size*TMCfg.max_valence + cfg.valence_enc_size
        self.final_enc = AtnFlat(final_enc_size,
                                 cfg.final_enc_size,
                                 BondAttentionFixed, False)

        self.bond_enc = BondEncoder(cfg.final_enc_size,
                                    cfg.bond_enc_filters,
                                    cfg.bond_pred_filters)

        self.final_final_enc = AtnFlat(cfg.final_enc_size + cfg.bond_pred_filters,
                                       cfg.final_enc_size,
                                       BondAttentionFixed, False)

        #flat_filter_list = [final_enc_size] + [cfg.kp_enc_size]*6
        #self.final_enc = IterativeSequential(AtnFlat, flat_filter_list, BondAttentionFixed, True)

        self.bidirectional = (cfg.num_gru_directions==2)
        self.rnn = nn.GRU(cfg.final_enc_size,
                          cfg.enc_rnn_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.flat = linear(cfg.enc_rnn_size*cfg.num_gru_directions, hidden_size)

    def forward(self, tmol, device):
        aenc = self.atom_embed(tmol.atom_types)
        aenc = self.atom_enc(aenc, tmol.bonds)

        if self.predict_valence:
            venc = self.valence_embed(tmol.bonds.atom_valences, device)
            venc = [self.valence_enc(venc, tmol.bonds)]
        else:
            venc = []

        if self.use_kps:
            kpenc = torch.unsqueeze(tmol.kps, 2)
            kpenc = self.kp_init_enc(kpenc)
            kpenc = self.kp_enc(kpenc, tmol.bonds)
            kpenc = kpenc.contiguous().view(kpenc.size(0), kpenc.size(1), -1)
            kpenc = self.kp_flat_enc(kpenc, tmol.bonds)

        batch, atoms = tmol.atom_types.shape
        if self.has_bond_type_enc:
            bond_type_encs = [self.bond_type_enc(tmol.bonds.bond_types[:,:,i]) for i in range(TMCfg.max_valence)]
        else:
            bond_type_encs = []
        
        if self.use_kps:
            enc = torch.cat((aenc, kpenc, *bond_type_encs, *venc), 2)
        else:
            enc = torch.cat((aenc, *bond_type_encs, *venc), 2)
            
        enc = self.final_enc(enc, tmol.bonds)

        bond_enc = self.bond_enc(enc, tmol.bonds)
        enc = torch.cat((enc, bond_enc), 2)
        enc = self.final_final_enc(enc, tmol.bonds)
        
        _, hidden = self.rnn(enc)
        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), 1)
        else:
            hidden = hidden[-1]
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
                          bidirectional=(cfg.num_gru_directions==2))
        self.bond_pred = BondPredictor(cfg.dec_rnn_size*cfg.num_gru_directions,
                                       cfg.bond_pred_filters)
        self.initial_dec = AtnFlat(cfg.dec_rnn_size*cfg.num_gru_directions,
                                   cfg.initial_dec_size,
                                   BondAttentionFixed, False)
        #self.atom_out = AtnFlat(cfg.initial_dec_size,
        #                        NUM_ATOM_TYPES,
        #                        BondAttentionFixed, False)
        #self.valence_out = ValenceDecoder(AtnFlat, cfg.initial_dec_size,
        #                                  BondAttentionFixed, False)

        self.atom_out = TimeDistributed(
            nn.Linear(cfg.dec_rnn_size*cfg.num_gru_directions, NUM_ATOM_TYPES),
            axis=2
        )
        #self.valence_out = ValenceDecoder(lambda hid, val: TimeDistributed(nn.Linear(hid, val), 2), cfg.dec_rnn_size*cfg.num_gru_directions)
        
        filter_list = list(reversed(cfg.kp_filter_list))
        mul = get_linear_mul(filter_list)
        width = get_final_width(filter_list)
        self.kp_flat_dec = AtnFlat(cfg.initial_dec_size,
                                   mul*filter_list[0],
                                   BondAttentionFixed, False)
        self.kp_reshape = Unflatten((TMCfg.max_atoms, filter_list[0], width, width, width))
        self.kp_dec = IterativeSequential(
            AtnUpConv, filter_list, cfg.kernel_size, cfg.padding, False
        )
        self.kp_out = TimeDistributed(
            nn.Conv3d(filter_list[-1], 1,
                      kernel_size=cfg.kernel_size, bias=False, padding=cfg.padding),
            axis=2
        )

    def forward(self, z, tmol, device):

        #if tmol is None:
        #    return TensorMol()
        
        rnn_in = self.lat_fc(z).unsqueeze(1).repeat(1, TMCfg.max_atoms, 1)
        dec, _ = self.rnn(rnn_in)

        bond_pred = self.bond_pred(dec)
        #out_valences = self.valence_out(dec)
        out_atom = self.atom_out(dec)

        if tmol is None:
            bonds = bond_pred.argmax(torch.argmax(out_atom, -1), torch.argmax(out_valences, -1))
        else:
            bonds = tmol.bonds
        
        dec = self.initial_dec(dec, bonds)
        
        kp_out = self.kp_flat_dec(dec, bonds)
        kp_out = self.kp_reshape(kp_out)
        kp_out = self.kp_dec(kp_out, bonds)
        kp_out = self.kp_out(kp_out).squeeze()

        return TensorMol(atom_types=out_atom,
                         #atom_valences=out_valences,
                         kps_1h=kp_out,
                         bonds=bond_pred)
                          
        
