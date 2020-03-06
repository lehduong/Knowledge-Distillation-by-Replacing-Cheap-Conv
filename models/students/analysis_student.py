from .depthwise_student import DepthwiseStudent
from .transform_blocks import *
from torch import nn
import copy
import torch 
import gc 


class AnalysisStudent(DepthwiseStudent):
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
            # FIXME: bias true or false?
            replace_block = nn.Sequential(cp_teacher_block,
                                          nn.Dropout(droprate),
                                          nn.Conv2d(teacher_block.in_channels, teacher_block.out_channels, 1, bias=True)
                                         )
            self.student_blocks.append(replace_block)
            self._set_block(block_name, replace_block, self.student)

        # free memory of unused layer i.e. the layer of student before replacing
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self):
        # remove hint layers
        self._remove_hooks()
        self.logger.info('Remove all hint layers')
        # remove replaced layers
        while self.replaced_block_names:
            block_name = self.replaced_block_names.pop()
            teacher_block  = self.get_block(block_name, self.teacher)
            student_block = copy.deepcopy(teacher_block)
            self._set_block(block_name, student_block, self.student)
            self.logger.debug("Replaced block {} in student".format(block_name))
            
        self.logger.info('Remove all replaced layer')
        