import torch.nn as nn


class WeightedHintMSELoss(nn.Module):

    def __init__(self, reduction='mean', num_classes=19):
        super(WeightedHintMSELoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets, filter_weight):
        sq_diff = (inputs - targets) ** 2
        spatial_reduced = sq_diff.mean(dim=(-1, -2))
        weighted = (filter_weight*spatial_reduced).sum(dim=-1) / filter_weight.sum(dim=-1)
        return weighted.mean()