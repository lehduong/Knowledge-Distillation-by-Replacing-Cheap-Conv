import copy
from collections import OrderedDict
from torch import nn

from .deeplabv3 import DeepWV3Plus
from ..common.initialization import initialize_weights


def get_distil_model(teacher):
    teacher = copy.deepcopy(teacher)
    for param in teacher.parameters():
        param.requires_grad = False

    block3_mod2 = nn.Sequential(
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, groups=128),
        nn.Conv2d(128, 128, kernel_size=1)
    )

    block3_mod3 = nn.Sequential(
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, groups=256),
        nn.Conv2d(256, 256, kernel_size=1)
    )

    block3_mod4 = nn.Sequential(
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
        nn.Conv2d(512, 512, kernel_size=1)
    )

    block6_mod4 = nn.Sequential(
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
        nn.Conv2d(512, 512, kernel_size=1)
    )

    block3_mod5 = nn.Sequential(
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=2, dilation=2, groups=1024),
        nn.Conv2d(1024, 1024, kernel_size=1)
    )

    added_bl = [block3_mod2, block3_mod3, block3_mod4, block6_mod4, block3_mod5]
    _ = [initialize_weights(bl) for bl in added_bl]

    teacher.module.mod2.block3 = block3_mod2
    teacher.module.mod3.block3 = block3_mod3
    teacher.module.mod4.block3 = block3_mod4
    teacher.module.mod4.block6 = block6_mod4
    teacher.module.mod5.block3 = block3_mod5

    return teacher