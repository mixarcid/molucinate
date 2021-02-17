import torch
import torch.nn as nn

LEAK_VALUE = 0.1

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)

def conv3(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, kernel_size=3, bias=False, padding=1),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )

def conv1(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, kernel_size=1, bias=False),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )
    
def downsample():
    return nn.MaxPool3d(2)

def upsample(in_f, out_f):
    return nn.Sequential(
        nn.ConvTranspose3d(in_f, out_f, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm3d(out_f),
        nn.LeakyReLU(LEAK_VALUE)
    )
