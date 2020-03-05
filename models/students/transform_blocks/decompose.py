from torch import nn 


class DecompositionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, kernel_size, padding, dilation, groups, bias, use_cuda=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size, padding=padding, dilation=dilation,
                              bias=bias)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
                               bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x