from models.students.aux_student import AuxStudent
from torch import nn
import gc
import torch


class IndependentStudent(AuxStudent):
    def __init__(self, model, distillation_args, aux_args, logger=None):
        """
        :param model: nn.Module object - pretrained model that need to be distilled
        :param distillation_args: LIST of DistillationArgs object - contains the distillation information
        """
        super().__init__(model, distillation_args, aux_args, logger)
        self.saved_student_blocks = nn.Module()
        self.saved_teacher_blocks = nn.Module()
        self.saved_distillation_args = list()

    def reset(self):
        """
        stop pruning current student layers and transfer back to original model
        :return:
        """
        is_learning = False
        if not self._teaching:
            self._assign_blocks(False)
            is_learning = True

        self._remove_hooks()
        for param in self.parameters():
            param.requires_grad = False
        self.teacher_blocks = nn.ModuleList()

        # saving student blocks for later usage
        for blocks in self.student_blocks:
            self.saved_student_blocks.append(blocks)
        for blocks in self.teacher_blocks:
            self.saved_teacher_blocks.append(blocks)
        self.saved_distillation_args += self.distillation_args
        self.student_blocks = nn.ModuleList()
        self.teacher_blocks = nn.ModuleList()
        self.distillation_args = []

        # flush memory
        gc.collect()
        torch.cuda.empty_cache()

        if is_learning:
            self._assign_blocks(True)

    def restore(self):
        self._remove_hooks()
        self.distillation_args = self.saved_distillation_args
        self.student_blocks = self.saved_student_blocks
        self.teacher_blocks = self.saved_teacher_blocks
        self._register_hooks()

        for param in self.model.parameters():
            param.requires_grad = True
