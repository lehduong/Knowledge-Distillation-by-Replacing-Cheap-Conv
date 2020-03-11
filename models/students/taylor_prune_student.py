from .depthwise_student import DepthwiseStudent
from .transform_blocks import *
from torch import nn
import copy
import torch
import gc


class TaylorPruneStudent(DepthwiseStudent):
    """
        To determine the importance of a feature map in conv layer
    """

    def __init__(self, teacher_model, config):
        super().__init__(teacher_model, config)
        self.added_gates = dict()

    def replace(self, block_names, **kwargs):
        """
        Replace a block with itself + gate layer
        :param block_names: str
        :return:
        """
        # gate is a vector with "num_features" element
        num_features = kwargs['num_features']

        for block_name in block_names:
            self.replaced_block_names.append(block_name)
            # get teacher block to retrieve information such as channel dim,...
            teacher_block = self.get_block(block_name, self.teacher)
            # the copied teacher block's parameters are fixed by default
            cp_teacher_block = copy.deepcopy(teacher_block)
            # add gate to calculate importance
            gate_layer = GateLayer(num_features)
            self.added_gates[block_name] = gate_layer
            replace_block = nn.Sequential(cp_teacher_block, gate_layer).cuda()
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
            teacher_block = self.get_block(block_name, self.teacher)
            student_block = copy.deepcopy(teacher_block)
            self._set_block(block_name, student_block, self.student)
            logger.debug("Replace the layer {} back to teacher's block".format(block_name))
        logger.debug("Reset completed...")

    def get_gate_importance(self):
        importance_dict = ()
        for name, gate_layer in self.added_gates.items():
            gate_weight = gate_layer.weight
            gate_grad = gate_layer.weight.grad
            importance = (gate_weight*gate_grad)**2
            importance_dict[name] = importance.cpu().numpy()

        return importance_dict

