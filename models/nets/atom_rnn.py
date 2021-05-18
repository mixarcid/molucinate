import torch
import torch.nn as nn

from .ms_conv import MsConv
from .ms_rnn import MsRnn
from .nn_utils import *
from .padding import get_padded_atypes

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *
    
class AtomRnnEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()
        self.embedding = nn.Embedding(NUM_ATOM_TYPES, cfg.gru_embed_size)
        self.rnn = nn.GRU(cfg.gru_embed_size,
                          hidden_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True)
        
    def forward(self, tmol, device):
        padded = get_padded_atypes(tmol, device)
        embedded = self.embedding(padded)
        _, hidden = self.rnn(embedded)
        assert(hidden.shape[0] == 1)
        return hidden[0]

class AtomRnnDecoder(nn.Module):
    def __init__(self, latent_size, cfg, gcfg):
        super().__init__()
        self.embedding = nn.Embedding(NUM_ATOM_TYPES, cfg.gru_embed_size)
        self.rnn = nn.GRU(cfg.gru_embed_size + cfg.gru_hidden_size,
                          cfg.gru_hidden_size,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True)
        self.lat_fc = nn.Sequential(
            nn.Linear(latent_size, cfg.gru_hidden_size)
        )
        self.out_fc = nn.Sequential(
            nn.Linear(cfg.gru_hidden_size, NUM_ATOM_TYPES)
        )

    def forward(self, z, tmol, device):
        if tmol is None:
            return self.generate(z, device)

        x = get_padded_atypes(tmol, device)
        x_emb = self.embedding(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)

        h_0 = self.lat_fc(z)
        h_0 = h_0.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)

        output, _ = self.rnn(x_input, h_0)
        y = self.out_fc(output)
        
        return TensorMol(atom_types=y[:,1:])

    def generate(self, z, device):
        
        n_batch = z.shape[0]
        z_0 = z.unsqueeze(1)
        h = self.lat_fc(z)
        h = h.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)

        w = torch.tensor(ATOM_TYPE_HASH["^"]).repeat(n_batch).to(device)
        x = torch.tensor([ATOM_TYPE_HASH["_"]]).repeat(n_batch, TMCfg.max_atoms+1).to(device)
        x[:, 0] = ATOM_TYPE_HASH["^"]
        end_pads = torch.tensor([TMCfg.max_atoms+1]).repeat(n_batch).to(device)
        eos_mask = torch.zeros(n_batch, dtype=torch.long).to(device)

        # Generating cycle
        for i in range(1, TMCfg.max_atoms+1):
            x_emb = self.embedding(w).unsqueeze(1)
            x_input = torch.cat([x_emb, z_0], dim=-1)

            o, h = self.rnn(x_input, h)
            y = self.out_fc(o.squeeze(1))
            w = torch.argmax(y, 1)
            x[:,i] = w

        return TensorMol(atom_types=x[:,1:])
