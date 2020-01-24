from .deeplabv3 import DeepWV3Plus
from torch import nn
from collections import OrderedDict
from ..common.initialization import initialize_weights

# def get_distil_model(model):
#     """
#     :param model: a pretrained DeepWV3Plus model
#     :return: distil version of input
#     """
#     mods = model.module

#     # Keep module 1
#     mod1 = mods.mod1

#     ##################################################################
#     # Distil module 2
#     ##################################################################
#     mod2_list = []

#     ## keep first 2 blocks (freeze their weights)
#     block1 = mods.mod2.block1
#     for p in block1.parameters():
#         p.requires_grad = False
#     mod2_list.append(('block1', mods.mod2.block1))

#     block2 = mods.mod2.block2
#     for p in block2.parameters():
#         p.requires_grad = False
#     mod2_list.append(('block2', mods.mod2.block2))

#     ## remove the third blocks and add deepwise-separable dilated conv
#     mod2_list.append(('block3', nn.Sequential(
#         nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2, groups=128),
#         nn.Conv2d(128, 128, kernel_size=1)
#     )))

#     mod2 = nn.Sequential(OrderedDict(mod2_list))

#     ##################################################################
#     # Distil module 3
#     ##################################################################
#     mod3_list = []
#     mod3 = mods.mod3

#     ## keep first 2 blocks (freeze their weights)
#     block1 = mod3.block1
#     for p in block1.parameters():
#         p.requires_grad = False
#     mod3_list.append(('block1', mod3.block1))

#     block2 = mod3.block2
#     for p in block2.parameters():
#         p.requires_grad = False
#     mod3_list.append(('block2', mod3.block2))

#     ## remove the third blocks and add deepwise-separable dilated conv
#     mod3_list.append(('block3', nn.Sequential(
#         nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2, groups=256),
#         nn.Conv2d(256, 256, kernel_size=1)
#     )))

#     mod3 = nn.Sequential(OrderedDict(mod3_list))

#     ##################################################################
#     # Distil module 4
#     ##################################################################
#     mod4_list = []
#     mod4 = mods.mod4

#     ## keep first 2 blocks (freeze their weights)
#     block1 = mod4.block1
#     for p in block1.parameters():
#         p.requires_grad = False
#     mod4_list.append(('block1', mod4.block1))

#     block2 = mod4.block2
#     for p in block2.parameters():
#         p.requires_grad = False
#     mod4_list.append(('block2', mod4.block2))

#     ## remove the third blocks and add deepwise-separable dilated conv
#     mod4_list.append(('block3', nn.Sequential(
#         nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
#         nn.Conv2d(512, 512, kernel_size=1)
#     )))

#     ## keep next 2 blocks (freeze their weights)
#     block4 = mod4.block4
#     for p in block4.parameters():
#         p.requires_grad = False
#     mod4_list.append(('block4', mod4.block4))

#     block5 = mod4.block5
#     for p in block5.parameters():
#         p.requires_grad = False
#     mod4_list.append(('block5', mod4.block5))

#     ## remove the sixth blocks and add deepwise-separable dilated conv
#     mod4_list.append(('block6', nn.Sequential(
#         nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, groups=512),
#         nn.Conv2d(512, 512, kernel_size=1)
#     )))
#     mod4 = nn.Sequential(OrderedDict(mod4_list))

#     ##################################################################
#     # Distil module 5
#     ##################################################################
#     mod5_list = []
#     mod5 = mods.mod5

#     ## keep first 2 blocks (freeze their weights)
#     block1 = mod5.block1
#     for p in block1.parameters():
#         p.requires_grad = False
#     mod5_list.append(('block1', mod5.block1))

#     block2 = mod5.block2
#     for p in block2.parameters():
#         p.requires_grad = False
#     mod5_list.append(('block2', mod5.block2))

#     ## remove the third blocks and add deepwise-separable dilated conv
#     mod5_list.append(('block3', nn.Sequential(
#         nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(1024, 1024, kernel_size=3, padding=2, dilation=2, groups=1024),
#         nn.Conv2d(1024, 1024, kernel_size=1)
#     )))
#     mod5 = nn.Sequential(OrderedDict(mod5_list))

#     # Keep mod 6, 7
#     mod6 = mods.mod6
#     mod7 = mods.mod7
#     return [mod1, mod2, mod3, mod4, mod5, mod6, mod7]

def get_distil_model(teacher):
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

    modules_tc = teacher.module
    lst_block_tc = [modules_tc.mod2.block3, modules_tc.mod3.block3, modules_tc.mod4.block3, 
                        modules_tc.mod4.block6, modules_tc.mod5.block3]

    lst_block_st = [block3_mod2, block3_mod3, block3_mod4, block6_mod4, block3_mod5]

    modules_tc.mod2.block3 = block3_mod2
    modules_tc.mod3.block3 = block3_mod3
    modules_tc.mod4.block3 = block3_mod4
    modules_tc.mod4.block6 = block6_mod4
    modules_tc.mod5.block3 = block3_mod5

    return teacher, lst_block_tc, lst_block_st