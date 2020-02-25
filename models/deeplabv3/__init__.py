import copy
from torch import nn

from .deeplabv3 import DeepWV3Plus, DeepR50V3PlusD_m1, DeepSRNX50V3PlusD_m1, DeepSRNX101V3PlusD_m1, DeepV3Plus
from ..common.initialization import initialize_weights
from models.students.base_student import DistillationArgs


def get_distil_model(teacher):
    """
    TODO: Remove this method
    Deprecated,
    :param teacher:
    :return:
    """
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


def get_distillation_args(level=1):
    """
    Get distillation arguments for StudentBase class. The required arguments have 3 properties: old_block_name, \
        new_block, new_block_name where old_block_name is a STR indicate the block that would be replaced, new_block \
        is nn.Module - the block that will substitute for aforementioned block, and new_block_name is STR represents
        name of new block
    :param level: int - represent the mediate stage of distillation procedure
    :return: list of DistillationArgs object
    """
    #TODO: Add initialization for new blocks
    ret = list()
    ret.append(DistillationArgs("mod2.block3",
                                nn.Sequential(
                                    nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, groups=128),
                                    nn.Conv2d(128, 128, kernel_size=1)
                                ),
                                "mod2.block3"))

    ret.append(DistillationArgs("mod3.block3",
                                nn.Sequential(
                                    nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, groups=256),
                                    nn.Conv2d(256, 256, kernel_size=1)
                                ),
                                "mod3.block3"))

    ret.append(DistillationArgs("mod4.block3",
                                nn.Sequential(
                                    nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
                                    nn.Conv2d(512, 512, kernel_size=1)
                                ),
                                "mod4.block3"))

    ret.append(DistillationArgs("mod4.block6",
                                nn.Sequential(
                                    nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
                                    nn.Conv2d(512, 512, kernel_size=1)
                                ),
                                "mod4.block6"))

    # ret.append(DistillationArgs("mod5.block3",
    #                             nn.Sequential(
    #                                 nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #                                 nn.ReLU(inplace=True),
    #                                 nn.Conv2d(1024, 1024, kernel_size=3, padding=2, dilation=2, groups=1024, bias=False),
    #                                 nn.Conv2d(1024, 1024, kernel_size=1, bias=False)
    #                             ),
    #                             "mod5.block3"))

    return ret


