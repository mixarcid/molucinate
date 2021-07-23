import torch
import torch.nn as nn

from .nn_utils import *
from .mg_decoder import MgDecoder
from .time_distributed import TimeDistributed
from .bond_attention import *
from .bond_predictor import BondPredictor
from .attention_utils import *
from .valence_utils import *
from .padding import get_padded_kps, get_padded_atypes, get_padded_valences, get_padded_bonds

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TensorBonds, TMCfg
from data.chem import *

class ArNetDecoder(nn.Module):

    def __init__(self, latent_size, cfg, gcfg):
        super().__init__()

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, cfg.atom_embed_size)
        self.atom_enc = AtnFlat(cfg.atom_embed_size,
                                cfg.atom_enc_size,
                                BondAttentionFixed,
                                True)

        self.predict_valence = gcfg.predict_valence
        if self.predict_valence:
            self.valence_embed = ValenceEmbedding(cfg.valence_embed_size)
            self.valence_enc = AtnFlat(cfg.valence_embed_size,
                                       cfg.valence_enc_size,
                                       BondAttentionFixed,
                                       True)

        self.use_kps = gcfg.data.use_kps
        if self.use_kps:
            self.kp_init_enc = TimeDistributed(
                nn.Sequential(
                    nn.Conv3d(1, cfg.kp_filter_list[0],
                              kernel_size=cfg.kernel_size, bias=False, padding=cfg.padding),
                    nn.BatchNorm3d(cfg.kp_filter_list[0]),
                    nn.LeakyReLU(LEAK_VALUE)
                ),
                axis=2
            )
            self.kp_enc = IterativeSequential(
                AtnDownConv, cfg.kp_filter_list, True
            )
            mul = get_linear_mul(cfg.kp_filter_list)
            self.kp_flat_enc = AtnFlat(mul*cfg.kp_filter_list[-1],
                                       cfg.kp_enc_size,
                                       BondAttentionFixed, True)
            kp_enc_size = cfg.kp_enc_size
        else:
            kp_enc_size = 0

        self.bond_type_enc = nn.Embedding(NUM_BOND_TYPES, cfg.bond_embed_size)

        final_enc_size = cfg.atom_enc_size + kp_enc_size + cfg.bond_embed_size*TMCfg.max_valence + cfg.valence_enc_size
        self.final_enc = AtnFlat(final_enc_size,
                                 cfg.final_enc_size,
                                 BondAttentionFixed, True)

        # self.bond_enc = BondEncoder(cfg.final_enc_size,
        #                             cfg.bond_enc_filters,
        #                             cfg.bond_pred_filters)

        # self.final_final_enc = AtnFlat(cfg.final_enc_size + cfg.bond_pred_filters,
        #                                cfg.final_enc_size,
        #                                BondAttentionFixed, True)
        
        self.lat_fc = nn.Sequential(
            nn.Linear(latent_size, cfg.dec_lat_fc_size, bias=False),
            nn.BatchNorm1d(cfg.dec_lat_fc_size),
            nn.LeakyReLU(LEAK_VALUE)
        )
        self.rnn = nn.GRU(cfg.dec_lat_fc_size + cfg.final_enc_size,
                          cfg.dec_rnn_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True,
                          bidirectional=False)

        print(cfg.self_attention_layers)
        self.self_attentions = nn.ModuleList([ SelfAttention(cfg.dec_rnn_size, cfg.self_attention_heads) for n in range(cfg.self_attention_layers) ])
        
        self.bond_pred = BondPredictor(cfg.dec_rnn_size,
                                       cfg.bond_pred_filters)
        self.initial_dec = AtnFlat(cfg.dec_rnn_size,
                                   cfg.initial_dec_size,
                                   BondAttentionFixed, True)

        self.atom_out = TimeDistributed(
            nn.Linear(cfg.dec_rnn_size, NUM_ATOM_TYPES),
            axis=2
        )
        if self.predict_valence:
            self.valence_out = ValenceDecoder(lambda hid, val: TimeDistributed(nn.Linear(hid, val), 2), cfg.dec_rnn_size)

        if self.use_kps:
            filter_list = list(reversed(cfg.kp_filter_list))
            mul = get_linear_mul(filter_list)
            width = get_final_width(filter_list)
            self.kp_flat_dec = AtnFlat(cfg.initial_dec_size,
                                       mul*filter_list[0],
                                       BondAttentionFixed, True)
            self.kp_reshape = Unflatten((-1, filter_list[0], width, width, width))
            self.kp_dec = IterativeSequential(
                AtnUpConv, filter_list, True
            )
            self.kp_out = TimeDistributed(
                nn.Conv3d(filter_list[-1], 1,
                          kernel_size=cfg.kernel_size, bias=False, padding=cfg.padding),
                axis=2
            )

    def forward(self, z, tmol, device, use_tmol_bonds=True, truncate_atoms=True):
        batch_size = z.size(0)

        if tmol is None:
            return self.generate(z, device)

        atypes = get_padded_atypes(tmol, device, batch_size, truncate_atoms)
        aenc = self.atom_embed(atypes)
        aenc = self.atom_enc(aenc, tmol.bonds, True)

        bonds = get_padded_bonds(tmol, device, batch_size, truncate_atoms)
        bond_type_encs = [self.bond_type_enc(bonds.bond_types[:,:,i]) for i in range(TMCfg.max_valence)]

        if self.predict_valence:
            venc = self.valence_embed(bonds.atom_valences, device)
            venc = [self.valence_enc(venc, tmol.bonds, True)]
        else:
            vence = []
        
        if self.use_kps:
            kpenc = get_padded_kps(tmol, device, batch_size, truncate_atoms)
            kpenc = self.kp_init_enc(kpenc)
            kpenc = self.kp_enc(kpenc, tmol.bonds, True)
            kpenc = kpenc.contiguous().view(kpenc.size(0), kpenc.size(1), -1)
            kpenc = self.kp_flat_enc(kpenc, tmol.bonds, True)

            kpenc = [kpenc]
        else:
            kpenc = []

        enc = torch.cat((aenc, *kpenc, *bond_type_encs, *venc), 2)
        enc = self.final_enc(enc, tmol.bonds, True)

        #bond_enc = self.bond_enc(enc, tmol.bonds)
        #enc = torch.cat((enc, bond_enc), 2)
        #enc = self.final_final_enc(enc, tmol.bonds)

        lat_in = self.lat_fc(z).unsqueeze(1).repeat(1, atypes.size(1), 1)
        rnn_in = torch.cat([lat_in, enc], 2)
        dec, _ = self.rnn(rnn_in)

        mask = torch.ones((dec.size(1), dec.size(1)), device=device, dtype=bool)
        mask = torch.triu(mask, diagonal=1)

        for atn in self.self_attentions:
            dec = atn(dec, mask)

        bond_pred = self.bond_pred(dec)
        out_valences = self.valence_out(dec)
        bond_pred.atom_valences = out_valences
        out_atom = self.atom_out(dec)

        if use_tmol_bonds:
            bonds = tmol.bonds
        else:
            bonds = bond_pred.argmax(torch.argmax(out_atom, -1))#, torch.argmax(out_valences, -1))

        if self.use_kps:
            dec = self.initial_dec(dec, bonds)
            kp_out = self.kp_flat_dec(dec, bonds)
            kp_out = self.kp_reshape(kp_out)
            kp_out = self.kp_dec(kp_out, bonds)
            kp_out = self.kp_out(kp_out).squeeze(2)
        else:
            kp_out = None

        return TensorMol(atom_types=out_atom,
                         #atom_valences=out_valences,
                         kps_1h=kp_out,
                         bonds=bond_pred)

    def generate(self, z, device):
        if self.use_kps:
            kps = torch.tensor([], device=device, dtype=torch.float)
        else:
            kps = None
        mol = TensorMol(atom_types=torch.tensor([], device=device, dtype=torch.long),
                        kps_1h=kps,
                        kps=kps,
                        bonds=TensorBonds(bond_types=torch.tensor([], device=device, dtype=torch.long),
                                          bonded_atoms=torch.tensor([], device=device, dtype=torch.long),
                                          atom_valences=torch.tensor([], device=device, dtype=torch.long)))
        for i in range(TMCfg.max_atoms):
            mol = self(z, mol.argmax(), device, False, False)
        return mol
