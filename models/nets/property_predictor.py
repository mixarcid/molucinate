import torch
import torch.nn as nn

from .nn_utils import *

class PropertyPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        def linear_bn(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f, bias=False),
                nn.BatchNorm1d(out_f),
                # todo: no
                nn.LayerNorm(out_f),
                nn.LeakyReLU(LEAK_VALUE)
            )
        
        self.net = IterativeSequential(
            linear_bn, [cfg.latent_size] + list(cfg.prop_pred_filters)
        )

        self.final = nn.Linear(cfg.prop_pred_filters[-1], 1)

    def forward(self, x):
        x = self.net(x)
        return self.final(x).squeeze()
