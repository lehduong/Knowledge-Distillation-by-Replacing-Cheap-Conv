import torch.nn as nn


class MSELoss(nn.Module):
    """
    MSE loss
    """

    def __init__(self, reduction='mean', num_classes=19):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # balance this loss vs crossentropy loss since ce loss doesn't divided by num_classes
        return self.mse_loss(inputs, targets)*self.num_classes
