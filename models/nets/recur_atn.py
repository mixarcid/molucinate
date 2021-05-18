import torch
import torch.nn as nn

from .nn_utils import *
from .mg_decoder import MgDecoder
from .time_distributed import TimeDistributed
from .bond_attention import *
from .bond_predictor import BondPredictor
from .attention_utils import *

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

class RecurAtnDecoder(nn.Module):

    def __init__(self, latent_size, cfg, gcfg):
        super().__init__()
        self.lat_fc = nn.Sequential(
            nn.Linear(latent_size, cfg.dec_lat_fc_size, bias=False),
            nn.BatchNorm1d(cfg.dec_lat_fc_size),
            nn.LeakyReLU(LEAK_VALUE)
        )
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, cfg.atom_embed_size)
        self.atom_enc = AtnFlat(cfg.atom_embed_size,
                                cfg.atom_enc_size,
                                BondAttentionFixed,
                                True)

        self.final_enc = AtnFlat(cfg.atom_enc_size + self.lat_fc_size,
                                 cfg.dec_final_enc_size,
                                 BondAttentionFixed,
                                 True)
        
        self.atom_dec = nn.GRU(cfg.dec_final_enc_size,
                               NUM_ATOM_TYPES,
                               num_layers=cfg.num_gru_layers,
                               batch_first=True,
                               bidirectional=(cfg.num_gru_directions==2))

    def forward(self, z, tmol, device):
        if tmol is None:
            return TensorMol()
