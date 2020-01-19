from .deeplabv3 import DeepWV3Plus
from torch import nn
from collections import OrderedDict

def get_distil_model(model):
    """
    :param model: a pretrained DeepWV3Plus model
    :return: distil version of input
    """
    mods = model.module

    # Keep module 1
    mod1 = mods.mod1

    ##################################################################
    # Distil module 2
    ##################################################################
    mod2_list = []

    ## keep first 2 blocks (freeze their weights)
    block1 = mods.mod2.block1
    for p in block1.parameters():
        p.requires_grad = False
    mod2_list.append(('block1', mods.mod2.block1))

    block2 = mods.mod2.block2
    for p in block2.parameters():
        p.requires_grad = False
    mod2_list.append(('block2', mods.mod2.block2))

    ## remove the third blocks and add deepwise-separable dilated conv
    mod2_list.append(('block3', nn.Sequential(
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, groups=128),
        nn.Conv2d(128, 128, kernel_size=1)
    )))

    mod2 = nn.Sequential(OrderedDict(mod2_list))

    ##################################################################
    # Distil module 3
    ##################################################################
    mod3_list = []
    mod3 = mods.mod3

    ## keep first 2 blocks (freeze their weights)
    block1 = mod3.block1
    for p in block1.parameters():
        p.requires_grad = False
    mod3_list.append(('block1', mod3.block1))

    block2 = mod3.block2
    for p in block2.parameters():
        p.requires_grad = False
    mod3_list.append(('block2', mod3.block2))

    ## remove the third blocks and add deepwise-separable dilated conv
    mod3_list.append(('block3', nn.Sequential(
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, groups=128),
        nn.Conv2d(256, 256, kernel_size=1)
    )))

    mod3 = nn.Sequential(OrderedDict(mod3_list))


