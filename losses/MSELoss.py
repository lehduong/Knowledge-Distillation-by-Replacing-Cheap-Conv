import torch.nn as nn


class MSELoss(nn.Module):
    """
    MSE loss
    """

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, targets):
        return self.mse_loss(inputs, targets)*inputs.shape[1]
