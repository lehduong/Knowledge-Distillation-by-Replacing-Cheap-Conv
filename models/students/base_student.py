import copy
import torch
import gc
import numpy as np
from collections import namedtuple
from functools import reduce
from torch import nn
from base import BaseModel
from beautifultable import BeautifulTable

BLOCKS_LEVEL_SPLIT_CHAR = '.'
DistillationArgs = namedtuple('DistillationArgs', ['old_block_name', 'new_block', 'new_block_name'])


class BaseStudent(BaseModel):
    def __init__(self, model, distillation_args, logger=None):
        """
        :param model: nn.Module object - pretrained model that need to be distilled
        :param distillation_args: LIST of DistillationArgs object - contains the distillation information
        """
        super().__init__()
        # deep cloning teacher model as we will change it later depends on training purpose
        self.model = copy.deepcopy(model)
        if self.model.training:
            self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._teaching = True  # teaching mode
        self.distillation_args = distillation_args
        # stored output of intermediate layers when
        self.student_hidden_outputs = list()
        self.teacher_hidden_outputs = list()

        self.student_blocks = list()
        self.teacher_blocks = list()
        self._prepare_blocks(distillation_args)

        self._student_hook_handlers = list()
        self._teacher_hook_handlers = list()

        self.training = False
        self.train()

    def _assign_blocks(self, student_mode=True):
        """
        change the distilled blocks to teacher blocks and vice versal
        :param student_mode: boolean - indicate the blocks
        :return:
        """
        if student_mode:
            for idx, block in enumerate(self.distillation_args):
                # self._set_block(block.old_block_name, block.new_block)
                self._set_block(block.old_block_name, self.student_blocks[idx])
            self._teaching = False
        else:
            for idx, block in enumerate(self.distillation_args):
                self._set_block(block.old_block_name, self.teacher_blocks[idx])
            self._teaching = True

    def _register_hooks(self):
        # register hook for saving student hidden outputs
        for block in self.student_blocks:
            handler = block.register_forward_hook(lambda m, inp, out: self.student_hidden_outputs.append(out))
            self._student_hook_handlers.append(handler)

        # register hook for saving teacher hidden outputs
        for block in self.teacher_blocks:
            handler = block.register_forward_hook(lambda m, inp, out: self.teacher_hidden_outputs.append(out))
            self._teacher_hook_handlers.append(handler)

    def _remove_hooks(self):
        while self._student_hook_handlers:
            handler = self._student_hook_handlers.pop()
            handler.remove()
        while self._teacher_hook_handlers:
            handler = self._teacher_hook_handlers.pop()
            handler.remove()

    def _prepare_blocks(self, distillation_args):
        # ATTENTION: Must run store teacher block before calling store student block
        self._store_teacher_blocks(distillation_args)
        self._store_student_blocks(distillation_args)
        if len(self.student_blocks) != len(self.teacher_blocks):
            raise Exception("Number of blocks in Student Network must be equal to Teacher Network")

    def _store_teacher_blocks(self, distillation_args):
        """
        store teacher blocks that are going to be replaced by distilled blocks
        :return: None
        """
        for block in distillation_args:
            block_name = block.old_block_name
            teacher_block = self.get_block(block_name)
            for param in teacher_block.parameters():
                param.requires_grad = False
            self.teacher_blocks.append(teacher_block)
        if not isinstance(self.teacher_blocks, nn.ModuleList):
            self.teacher_blocks = nn.ModuleList(self.teacher_blocks)

    def _store_student_blocks(self, distillation_args):
        """
        store newly initialized distilled blocks of student net
        :return: None
        """
        for block in distillation_args:
            student_block = block.new_block
            self.student_blocks.append(student_block)
        if not isinstance(self.student_blocks, nn.ModuleList):
            self.student_blocks = nn.ModuleList(self.student_blocks)

    def _set_block(self, block_name, block):
        """
        set a hidden block to particular object
        :param block_name: str
        :param block: nn.Module
        :return: None
        """
        block_name_split = block_name.split(BLOCKS_LEVEL_SPLIT_CHAR)
        # suppose the blockname is abc.def.ghk then get module self.teacher.abc.def and set that object's attribute \
        # (in this case 'ghk') to block value
        if len(block_name_split) == 1:
            setattr(self.model, block_name, block)
        else:
            obj = self.get_block(BLOCKS_LEVEL_SPLIT_CHAR.join(block_name_split[:-1]))
            attr = block_name_split[-1]
            setattr(obj, attr, block)

    def get_block(self, block_name):
        """
        get block from block name
        :param block_name: str - should be st like abc.def.ghk
        :return: nn.Module
        """
        def _get_block(acc, elem):
            if elem.isdigit():
                layer = acc[int(elem)]
            else:
                layer = getattr(acc, elem)
            return layer

        return reduce(lambda acc, elem: _get_block(acc, elem), block_name.split(BLOCKS_LEVEL_SPLIT_CHAR), self.model)

    def to_teacher(self):
        """
        promote all the student layers to teacher layer i.e. current student would become teacher assistant
        # https://arxiv.org/pdf/1902.03393.pdf
        :return:
        """
        # keep a flag to indicate the _teaching mode and revert the network to its mode before calling this function
        # to prevent unpredictable behaviors
        is_teaching = False
        if self._teaching:
            is_teaching = True
            self._assign_blocks(student_mode=True)

        self._remove_hooks()
        for param in self.model.parameters():
            param.requires_grad = False
        self.teacher_blocks = nn.ModuleList()
        self.student_blocks = nn.ModuleList()
        self.distillation_args = []

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

        if is_teaching:
            self._assign_blocks(student_mode=False)

    def update_pruned_layers(self, distillation_args):
        """
        Update the model to be compatible with new distillation args
        :param distillation_args: list of DistillationArgs
        :return: None
        """
        # remove all registered hooks in previous blocks as we're registering hooks all again
        self._remove_hooks()
        self.distillation_args += distillation_args
        # append the new block to student_blocks and teacher_blocks
        self._prepare_blocks(distillation_args)
        # registering hooks for all blocks
        self._register_hooks()

    def forward(self, x):
        # flush the output of last forward
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []

        # in training mode, the network has to forward 2 times, one for computing teacher's prediction \
        # and another for student's one
        self._assign_blocks(student_mode=False)
        with torch.no_grad():
            teacher_pred = self.model(x)
        self._assign_blocks(student_mode=True)
        student_pred = self.model(x)

        return student_pred, teacher_pred

    def inference(self, x):
        if self._teaching:
            self._assign_blocks(student_mode=True)
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []
        out = self.model(x)

        return out

    def eval(self):
        if not self.training:
            return

        self.training = False

        for block in self.student_blocks:
            block.eval()
        self._remove_hooks()

        return self

    def train(self):
        """
        The parameters of teacher's network will always be set to EVAL
        :return: self
        """
        if self.training:
            return

        self.training = True

        for block in self.student_blocks:
            block.train()

        if self._teaching:
            self._assign_blocks(student_mode=True)
        self._register_hooks()

        return self

    def get_distilled_network(self):
        """
        Get the distilled student network
        :return: nn.Module
        """
        # prevent side effect of this function
        flag = False
        if self._teaching:
            flag = True
            self._assign_blocks(student_mode=True)

        ret = copy.deepcopy(self.model)

        # turn the student network to teaching mode as before function call
        if flag:
            self._assign_blocks(student_mode=False)

        return ret

    @staticmethod
    def __get_number_param(mod):
        return sum(p.numel() for p in mod.parameters())

    @staticmethod
    def __dump_module_name(mod):
        ret = ""
        for param in mod.named_parameters():
            ret += str(param[0]) + "\n"
        return ret

    def dump_trainable_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return '\nTrainable parameters: {}'.format(params)

    def dump_student_teacher_blocks_info(self):
        table = BeautifulTable()
        table.column_headers = ["Block name", "old block",
                                "number params old blk", "new block",
                                "number params new blk"]

        table.left_padding_widths['Block name'] = 1
        table.right_padding_widths['Block name'] = 1
        table.left_padding_widths['old block'] = 1
        table.right_padding_widths['old block'] = 1
        table.left_padding_widths['number params old blk'] = 1
        table.right_padding_widths['number params old blk'] = 1
        table.left_padding_widths['new block'] = 1
        table.right_padding_widths['new block'] = 1
        table.left_padding_widths['number params new blk'] = 1
        table.right_padding_widths['number params new blk'] = 1

        for i in range(len(self.student_blocks)):
            table.append_row([self.distillation_args[i].old_block_name,
                              self.__dump_module_name(self.teacher_blocks[i]),
                              str(self.__get_number_param(self.teacher_blocks[i])),
                              self.__dump_module_name(self.student_blocks[i]),
                              str(self.__get_number_param(self.student_blocks[i]))])
        return str(table)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        table = self.dump_student_teacher_blocks_info()
        return super().__str__() + '\n' + table
