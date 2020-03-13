import torch.nn.functional as F
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    """
    Kullback-Leibler Divergence loss between 2 tensor
    return the KL divergence between distributions
    :param temperature - float:
    input:
        inputs - torch.Tensor: the predictions of 1 model. The shape of this tensor should be batchsize x C x H x W
        targets - torch.Tensor: soft prediction of ensemble models
    """

    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, inputs, targets):
        p_s = F.log_softmax(inputs, dim=1)
        p_t = targets
        loss = F.kl_div(p_s, p_t) *targets.shape[1]
        return loss
