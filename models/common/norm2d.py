from torch import nn


def Norm2d(in_channels):
    return nn.BatchNorm2d(in_channels)
