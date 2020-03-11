import torch.nn as nn
import torch


class GateLayer(nn.Module):
    def __init__(self, num_features):
        super(GateLayer, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, input):
        return input*self.weight.view(1, -1, 1, 1)
