from .depthwise_student import DepthwiseStudent
from .transform_blocks import *
from models.encoders.wider_resnet import bnrelu
from torch import nn
import copy
import torch 
import gc 


class EnsembleStudent(DepthwiseStudent):
    """
        To determine the redundant of a layers we suggest 
    """
    def __init__(self,teacher_model, config):
        super().__init__(teacher_model, config)
        self.studdents = nn.ModuleList()
    