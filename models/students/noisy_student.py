from .depthwise_student import DepthwiseStudent
from .transform_blocks import *
from models.encoders.wider_resnet import bnrelu
from torch import nn
import copy
import torch 
import gc 


class NoisyStudent(DepthwiseStudent):
    """
        To determine the redundant of a layers we suggest 
    """
    def __init__(self,teacher_model, config):
        super().__init__(teacher_model, config)

    def replace(self, block_names, **kwargs):
        """
        Replace a block with itself + dropout + 1x1 conv
        :param block_names: str
        :return:
        """
        # rate of dropout layers
        droprate = kwargs['droprate']
        
        for block_name in block_names:
            self.replaced_block_names.append(block_name)
            # get teacher block to retrieve information such as channel dim,...
            teacher_block = self.get_block(block_name, self.teacher)
            self.teacher_blocks.append(teacher_block)
            # replace student block with teacher block and heavy dropout
            cp_teacher_block = copy.deepcopy(teacher_block)
            # unfreeze the teacher weights
            for param in cp_teacher_block.parameters():
                param.requires_grad = True 
            # create larger teacher for knowledge expansion
            replace_block = nn.Sequential(cp_teacher_block,
                                          bnrelu(teacher_block.out_channels),
                                          nn.Dropout2d(droprate),
                                          nn.Conv2d(teacher_block.out_channels, 
                                                    teacher_block.out_channels, 
                                                    kernel_size=kwargs['kernel_size'],
                                                    padding=kwargs['padding'],
                                                    dilation=kwargs['dilation'],
                                                    bias=True)
                                         ).cuda()
            self.student_blocks.append(replace_block)
            self._set_block(block_name, replace_block, self.student)

        # free memory of unused layer i.e. the layer of student before replacing
        gc.collect()
        torch.cuda.empty_cache()
        