import torch
import torch.nn as nn

from .ms_conv import MsConv
from .ms_rnn import MsRnn
from .nn_utils import *
from .conv_gru import *
from .time_distributed import TimeDistributed

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg
from data.chem import *

def get_padded_atypes(tmol, device):
    batch_size = tmol.atom_types.shape[0]
    start_idx = torch.tensor([[ATOM_TYPE_HASH['^']]]*batch_size,
                             device=device,
                             dtype=torch.long)
    padded = torch.cat((start_idx, tmol.atom_types), 1)
    return padded
    
class AtomKpRnnEncoder(nn.Module):

    def __init__(self, hidden_size, cfg, gcfg):
        super().__init__()
        self.embedding = nn.Embedding(NUM_ATOM_TYPES, cfg.gru_embed_size)
        atom_out_sz = 128
        self.rnn = nn.GRU(cfg.gru_embed_size,
                          atom_out_sz,
                          num_layers=cfg.num_gru_layers,
                          batch_first=True)
        sz = TMCfg.grid_size
        filter_list = [4, 8, 16, 32]
        self.kp_rnns = nn.ModuleList()
        self.downs = nn.ModuleList()
        for f, f_prev in zip(filter_list, [1] + filter_list):
            self.kp_rnns.append(ConvGRU((sz, sz, sz),
                                        f_prev,
                                        f,
                                        (3,3,3),
                                        1,
                                        batch_first=True))
            self.downs.append(TimeDistributed(nn.Sequential(
                nn.BatchNorm3d(f),
                nn.LeakyReLU(LEAK_VALUE),
                downsample()
            ), axis=2))
            sz //= 2
            fc_sz = f*(sz**3)
        self.flat = Flatten()
        self.fc = nn.Sequential(
            nn.Linear(fc_sz + atom_out_sz, hidden_size),
            nn.LeakyReLU(LEAK_VALUE)
        )
        
    def forward(self, tmol, device):
        kps = torch.unsqueeze(tmol.kps, 2)
        padded = get_padded_atypes(tmol, device)
        embedded = self.embedding(padded)
        _, hidden = self.rnn(embedded)
        kph = kps
        for kp_rnn, down in zip(self.kp_rnns, self.downs):
            kp_hidden, _ = kp_rnn(kph, device=device)
            kph = down(kp_hidden[0])
        kph = self.flat(kph[:,-1])
        out = self.fc(torch.cat((kph, hidden[0]), -1))
        return out

class AtomKpRnnDecoder(nn.Module):
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

        filter_list = [
            512,
            256,
            256,
            64,
            32,
            32,
        ]
        width = get_final_width(filter_list)
        mul = get_linear_mul(filter_list)
        self.mg_fc = nn.Sequential(
            nn.Linear(latent_size, filter_list[0]*mul),
            nn.LeakyReLU(LEAK_VALUE),
            Unflatten((filter_list[0], width, width, width))
        )
        self.mg_convs = nn.ModuleList()
        for i, filt in enumerate(filter_list[:-1]):
            filt_next = filter_list[i+1]
            self.mg_convs.append(nn.Sequential(
                upsample(filt, filt_next),
                conv3(filt_next, filt_next)
            ))
        self.final_mg_conv = nn.Conv3d(filter_list[-1], NUM_ATOM_TYPES,
                                    kernel_size=1, bias=True)

    def get_molgrid(self, z, device):
        x = self.mg_fc(z)
        for conv in self.mg_convs:
            x = conv(x)
        x = self.final_mg_conv(x)
        return x

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
        
        return TensorMol(atom_types=y[:,1:],
                         molgrid=self.get_molgrid(z, device))

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

        return TensorMol(atom_types=x[:,1:],
                         molgrid=self.get_molgrid(z, device))
