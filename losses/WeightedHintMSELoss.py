import torch.nn as nn
import torch


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


class TopkHintMSELoss(nn.Module):

    def __init__(self, reduction='mean', num_classes=19, topk=0.5):
        super(TopkHintMSELoss, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.topk = topk

    def forward(self, inputs, targets):
        if len(targets.size()) == 4:
            norm = targets.norm(p=2, dim=(-1, -2))
        else:
            norm = targets

        idx_ascending = norm.argsort(dim=-1, descending=True)
        num_channels = idx_ascending.size(-1)
        idx_pivot = int(self.topk*num_channels)
        mask = torch.zeros(targets.size(0), num_channels)
        idx_keep = idx_ascending[:, :idx_pivot]

        for x, y in zip(mask, idx_keep):
            x[y] = 1.0

        sq_diff = (inputs - targets) ** 2
        spatial_reduced = sq_diff.mean(dim=(-1, -2))
        masked_loss = spatial_reduced*mask
        reduced_loss = masked_loss.sum() / mask.sum()
        return reduced_loss
