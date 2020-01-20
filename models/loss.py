import torch.nn.functional as F
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence loss between 2 tensor
    return the KL divergence between distributions
    :param temperature - float:
    input:
        inputs - torch.Tensor: the predictions of 1 model. The shape of this tensor should be batchsize x C x H x W
        targets - torch.Tensor: the target of
    """

    def __init__(self, temperature=1):
        self.temperature = temperature
        self.kldiv = nn.KLDivLoss()

    def forward(self, inputs, targets):
        return self.temperature * self.temperature * self.kldiv(F.log_softmax(inputs / self.temperature, dim=1),
                                                                F.softmax(targets / self.temperature, dim=1))


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
        self.temperature = temperature
        self.kldiv = nn.KLDivLoss()

    def forward(self, inputs, targets):
        q = 0.5 * (F.softmax(targets / self.temperature, dim=1) + F.softmax(inputs / self.temperature, dim=1))

        return self.temperature * self.temperature * 0.5 * (
                    self.kldiv(torch.log(q), F.softmax(targets / self.temperature, dim=1)) +
                    self.kldiv(torch.log(q), F.softmax(inputs / self.temperature, dim=1))
        )


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
