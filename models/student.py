import copy
from collections import namedtuple
from functools import reduce
import torch
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
        self._student_hidden_outputs = list()
        self._teacher_hidden_outputs = list()

        self.student_blocks = list()
        self.teacher_blocks = list()
        self._prepare_blocks()
        self._register_hook()
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

    def _register_hook(self):
        # register hook for saving student hidden outputs
        for block in self.student_blocks:
            block.register_forward_hook(lambda m, inp, out: self._student_hidden_outputs.append(out))

        # register hook for saving teacher hidden outputs
        for block in self.teacher_blocks:
            block.register_forward_hook(lambda m, inp, out: self._teacher_hidden_outputs.append(out))

    def _prepare_blocks(self):
        # ATTENTION: Must run store teacher block before calling store student block
        self._store_teacher_blocks()
        self._store_student_blocks()
        if len(self.student_blocks) != len(self.teacher_blocks):
            raise Exception("Number of blocks in Student Network must be equal to Teacher Network")

    def _store_teacher_blocks(self):
        """
        store teacher blocks that are going to be replaced by distilled blocks
        :return: None
        """
        for block in self.distillation_args:
            block_name = block.old_block_name
            self.teacher_blocks.append(self._get_block(block_name))
        self.teacher_blocks = nn.ModuleList(self.teacher_blocks)

    def _store_student_blocks(self):
        """
        store newly initialized distilled blocks of student net
        :return: None
        """
        for block in self.distillation_args:
            for param in block.new_block.parameters():
                param.requires_grad = True
            self.student_blocks.append(block.new_block)
        self.student_blocks = nn.ModuleList(self.student_blocks)

    def _get_block(self, block_name):
        """
        get block from block name
        :param block_name: str - should be st like abc.def.ghk
        :return: nn.Module
        """
        return reduce(lambda acc, elem: getattr(acc, elem), block_name.split(BLOCKS_LEVEL_SPLIT_CHAR), self.model)

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
        obj = self._get_block(BLOCKS_LEVEL_SPLIT_CHAR.join(block_name_split[:-1]))
        attr = block_name_split[-1]
        setattr(obj, attr, block)

    def forward(self, x):
        # flush the output of last forward
        self._student_hidden_outputs = []
        self._teacher_hidden_outputs = []

        # in training mode, the network has to forward 2 times, one for computing teacher's prediction \
        # and another for student's one
        self._assign_blocks(student_mode=False)
        with torch.no_grad():
            teacher_pred = self.model(x)
        self._assign_blocks(student_mode=True)
        student_pred = self.model(x)
        return student_pred, teacher_pred, self._student_hidden_outputs, self._teacher_hidden_outputs

    def inference(self, x):
        return self.model(x)

    def eval(self):
        for block in self.student_blocks:
            block.eval()

        return self

    def train(self):
        """
        The parameters of teacher's network will always be set to EVAL
        :return: self
        """
        for block in self.student_blocks:
            block.train()

        if self._teaching:
            self._assign_blocks(student_mode=True)

        return self

    @staticmethod
    def __get_number_param(mod):
        return sum(p.numel() for p in mod.parameters())

    @staticmethod
    def __str_module(mod):
        ret = ""
        for param in mod.named_parameters():
            ret += str(param[0]) + "\n"
        return ret

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
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
                              self.__str_module(self.teacher_blocks[i]),
                              str(self.__get_number_param(self.teacher_blocks[i])),
                              self.__str_module(self.student_blocks[i]),
                              str(self.__get_number_param(self.student_blocks[i]))])

        return super().__str__() + '\n' + str(table)
