from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, groups, bias, use_cuda=True):
        super().__init__()
        self.separable_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.pointwise_conv(x)
        return x