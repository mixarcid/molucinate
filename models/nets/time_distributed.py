import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, axis):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.axis = axis

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.shape[self.axis:])  # (samples * timesteps, input_size)

        y = self.module(x_reshape)
        y = y.contiguous().view(*x.shape[:self.axis], *y.shape[1:])
        return y
