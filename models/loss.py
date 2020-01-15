import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def js_div(output, target):
    m = (output+target)/2
    return 0.5*(F.kl_div(output, m)+F.kl_div(target, m))