import copy
import torch
import numpy as np
import gc
from collections import namedtuple
from functools import reduce
from torch import nn
from base import BaseModel
from beautifultable import BeautifulTable
from .transform_blocks import DepthwiseSeparableBlock
from utils import *

BLOCKS_LEVEL_SPLIT_CHAR = '.'


class DepthwiseStudent(BaseModel):
    def __init__(self, teacher_model, config):
        """
        :param teacher_model: nn.Module object - pretrained model that need to be distilled
        """
        super().__init__()
        self.config = config
        # deep cloning teacher model as we will change it later depends on training purpose
        self.teacher = copy.deepcopy(teacher_model)
        if self.teacher.training:
            self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        # create student net
        self.student = copy.deepcopy(self.teacher)

        # distillation args contain the distillation information such as block name, ...
        self.replaced_block_names = []
        # stored output of intermediate layers when
        self.student_hidden_outputs = list()
        self.teacher_hidden_outputs = list()
        # list of handlers for removing later
        self._student_hook_handlers = list()
        self._teacher_hook_handlers = list()

        # auxiliary layer
        self.aux_block_names = list()

        self.save_hidden = True 

    def register_hint_layers(self, block_names):
        """
        Register auxiliary layers for computing hint loss
        :param block_names: str
        :return:
        """
        # remove all added hint layers....
        if len(block_names) > 0:
            self._remove_hooks()
        # add new hint layers
        for block_name in block_names:
            self.aux_block_names.append(block_name)
            # get teacher and student block 
            teacher_block = self.get_block(block_name, self.teacher)
            student_block = self.get_block(block_name, self.student)

            # teacher's hook
            def teacher_handle(m, inp, out):
                if self.save_hidden:
                    self.teacher_hidden_outputs.append(out)

            teacher_handler = teacher_block.register_forward_hook(teacher_handle)
            self._teacher_hook_handlers.append(teacher_handler)

            # student's hook
            def student_handle(m, inp, out):
                if self.save_hidden:
                    self.student_hidden_outputs.append(out)

            student_handler = student_block.register_forward_hook(student_handle)
            self._student_hook_handlers.append(student_handler)
        gc.collect()
        torch.cuda.empty_cache()

    def unfreeze(self, block_names):
        for block_name in block_names:
            block = self.get_block(block_name, self.student)
            for param in block.parameters():
                param.requires_grad = True

    def replace(self, blocks, **kwargs):
        """
        Replace a block with depthwise conv
        :param block_names: list of dictionary, each dictionary should have following format:\
            {"name": 'abcd', "epoch": 5, "args"(optional): {"padding": 5, "dilation":3, "kernel_size":10}}
        :param **kwargs: other specifications such as dilation, padding,...
        :return:
        """
        for block in blocks:
            block_name = block['name']
            self.replaced_block_names.append(block_name)

            # get teacher block to retrieve information such as channel dim,...
            teacher_block = self.get_block(block_name, self.teacher)

            # replace student block with depth-wise separable block
            # if the argument is specified for each block than use that argument o.w. use default 
            #   i.e. config['pruning']['args']
            if "args" in block:
                kernel_size = block['args']['kernel_size']
                padding = block['args']['padding']
                dilation = block['args']['dilation']
            else:
                kernel_size = kwargs['kernel_size']
                padding = kwargs['padding']
                dilation = kwargs['dilation']
            # create atrous depthwise separable convolution             
            replace_block = DepthwiseSeparableBlock(in_channels=teacher_block.in_channels,
                                                    out_channels=teacher_block.out_channels,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    groups=teacher_block.in_channels,
                                                    bias=teacher_block.bias).cuda()
            # replaced that newly created layer to student network
            self._set_block(block_name, replace_block, self.student)

        gc.collect()
        torch.cuda.empty_cache()

    def _remove_hooks(self):
        while self._student_hook_handlers:
            handler = self._student_hook_handlers.pop()
            handler.remove()
        while self._teacher_hook_handlers:
            handler = self._teacher_hook_handlers.pop()
            handler.remove()

    def _set_block(self, block_name, block, model):
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
            setattr(model, block_name, block)
        else:
            obj = self.get_block(BLOCKS_LEVEL_SPLIT_CHAR.join(block_name_split[:-1]), model)
            attr = block_name_split[-1]
            setattr(obj, attr, block)

    def get_block(self, block_name, model):
        """
        get block from block name
        :param block_name: str - should be st like abc.def.ghk
        :param model: nn.Module - which model that block would be drawn from
        :return: nn.Module - required block
        """

        def _get_block(acc, elem):
            if elem.isdigit():
                layer = acc[int(elem)]
            else:
                layer = getattr(acc, elem)
            return layer

        return reduce(lambda acc, elem: _get_block(acc, elem), block_name.split(BLOCKS_LEVEL_SPLIT_CHAR), model)

    def forward(self, x):
        # flush the output of last forward
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []
        # in training mode, the network has to forward 2 times, one for computing teacher's prediction \
        # and another for student's one
        with torch.no_grad():
            teacher_pred = self.teacher(x)
        student_pred = self.student(x)
        return student_pred, teacher_pred

    def inference(self, x):
        # flush the output of last forward
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []

        student_pred = self.student(x)
        return student_pred

    def inference_test(self, data, args):
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []

        to_PIL = transforms.ToPILImage()
        # TODO: Fixing really bad code here
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        result = []
        images = [to_PIL(x.cpu()) for x in data]
        for image in images:
            image_data = scale_and_flip_image(image, mean_std, args['scales'])
            ori_size, mapping, tensors = get_crops_image(image_data, args['scales'], crop_size=args['crop_size'])
            results_model = self.student(tensors.cuda()).data.cpu().numpy()
            outputs = reverse_mapping(mapping, results_model, ori_size)
            outputs_mean_for_scales = np.mean(outputs, axis=0)
            result.append(np.expand_dims(outputs_mean_for_scales, axis=0))

        result_arr = np.concatenate(result, axis=0)
        return torch.from_numpy(result_arr).cuda()

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

        # get student teacher blocks 
        teacher_blocks = [self.get_block(block_name, self.teacher) for block_name in self.replaced_block_names]
        student_blocks = [self.get_block(block_name, self.student) for block_name in self.replaced_block_names]
        # get info of student/teacher blocks 
        for i in range(len(self.replaced_block_names)):
            block_name = self.replaced_block_names[i]
            table.append_row([self.replaced_block_names[i],
                              self.__dump_module_name(teacher_blocks[i]),
                              str(self.__get_number_param(teacher_blocks[i])),
                              self.__dump_module_name(student_blocks[i]),
                              str(self.__get_number_param(student_blocks[i]))])
        return str(table)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        table = self.dump_student_teacher_blocks_info()
        return super().__str__() + '\n' + table

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

    def train(self,mode=True):
        if mode:
            self.save_hidden = True
        else:
            self.save_hidden = False 
        super().train(mode)
        # teacher will always in eval mode
        self.teacher.eval()

        return self