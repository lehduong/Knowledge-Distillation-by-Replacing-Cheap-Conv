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
            # replace student block with teacher block and heavy dropout
            # the copied teacher block's parameters are fixed by default
            cp_teacher_block = copy.deepcopy(teacher_block)
            # FIXME: bias true or false?
            replace_block = nn.Sequential(cp_teacher_block,
                                          RandomMask2d(teacher_block.out_channels, droprate),
                                          nn.Conv2d(teacher_block.out_channels, 
                                                    teacher_block.out_channels, 
                                                    kernel_size=1, 
                                                    bias=False)
                                         ).cuda()
            self._set_block(block_name, replace_block, self.student)

        # free memory of unused layer i.e. the layer of student before replacing
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self):
        logger = self.config.get_logger('trainer', self.config['trainer']['verbosity'])
        # remove hint layers
        self._remove_hooks()
        logger.debug('Removing all hint layers...')
        # remove replaced layers
        while self.replaced_block_names:
            block_name = self.replaced_block_names.pop()
            teacher_block  = self.get_block(block_name, self.teacher)
            student_block = copy.deepcopy(teacher_block)
            self._set_block(block_name, student_block, self.student)
            logger.debug("Replace the layer {} back to teacher's block".format(block_name))
        logger.debug("Reset completed...")
            