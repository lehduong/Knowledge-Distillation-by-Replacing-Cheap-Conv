import torch.nn.functional as F
import torch.nn as nn
import torch


class JSDivergenceLoss(nn.Module):
    """
    Jensen-Shanon Divergence loss between 2 tensor
    return the JS divergence between distributions
    :param temperature - float:
    input:
        inputs - torch.Tensor: the predictions of 1 model. The shape of this tensor should be batchsize x C x H x W
        targets - torch.Tensor: the target of
    """

    def __init__(self, temperature=1):
        super(JSDivergenceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        q = 0.5 * (F.softmax(targets / self.temperature, dim=1) + F.softmax(inputs / self.temperature, dim=1))

        return self.temperature * self.temperature * 0.5 * (
                    F.kl_div(torch.log(q), F.softmax(targets / self.temperature, dim=1), size_average=False)/targets.shape[0] +
                    F.kl_div(torch.log(q), F.softmax(inputs / self.temperature, dim=1), size_average=False)/targets.shape[0]
        )
