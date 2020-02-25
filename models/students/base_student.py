import copy
import torch
import gc
import numpy as np
import cv2
from collections import namedtuple
from functools import reduce
from torch import nn
from base import BaseModel
from beautifultable import BeautifulTable
from torchvision import transforms
from PIL import Image
from math import ceil


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
        for param in self.parameters():
            param.requires_grad = False
        self.teacher_blocks = nn.ModuleList()
        self.student_blocks = nn.ModuleList()
        self.distillation_args = []

        # remove cache
        gc.collect()
        torch.cuda.empty_cache()

        if is_teaching:
            self._assign_blocks(student_mode=False)

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
        self.student_blocks = nn.ModuleList()
        self.distillation_args = []
        gc.collect()
        torch.cuda.empty_cache()

        if is_learning:
            self._assign_blocks(True)

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

    def inference_test(self, data, args):
        if self._teaching:
            self._assign_blocks(student_mode=True)
        self.student_hidden_outputs = []
        self.teacher_hidden_outputs = []

        to_PIL = transforms.ToPILImage()
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        result = []
        images = [to_PIL(x.cpu()) for x in data]
        for image in images:
            image_data = _scale_and_flip_image(image, mean_std, args['scales'])
            ori_size, mapping, tensors = _get_crops_image(image_data, args['scales'], crop_size=args['crop_size'])
            results_model = self.model(tensors.cuda()).data.cpu().numpy()
            outputs = self.reverse_mapping(mapping, results_model, ori_size)
            outputs_mean_for_scales = np.mean(outputs, axis=0)
            result.append(np.expand_dims(outputs_mean_for_scales, axis=0))

        result_arr = np.concatenate(result, axis=0)
        return torch.from_numpy(result_arr).cuda()

    def reverse_mapping(self, mapping, results, ori_size):
        idx = 0
        outputs = []
        for items in mapping:
            w, h = items[0], items[1]
            coordinates = items[2]
            n_slices = len(coordinates)
            probs_no_flip = self.collect_windows_result(w, h, coordinates, results[idx: idx + n_slices])
            probs_flipped = self.collect_windows_result(w, h, coordinates, results[idx + n_slices: idx + 2*n_slices])

            list_slices_restore = [np.expand_dims(np.fliplr(x), 0) for x in probs_flipped]
            probs_flipped_restored = np.concatenate(list_slices_restore, axis=0)

            probs_no_flip_rs = self.resize_output(probs_no_flip, ori_size)
            probs_flipped_rs = self.resize_output(probs_flipped_restored, ori_size)

            probs_mean = (probs_no_flip_rs + probs_flipped_rs) / 2
            outputs.append(np.expand_dims(probs_mean, axis=0))
            idx += 2 * n_slices

        return np.concatenate(outputs, axis=0)

    def resize_output(self, masks, ori_size):
        mask_rs = []
        for x in masks:
            img_rs = cv2.resize(x, ori_size, interpolation=cv2.INTER_LINEAR)
            mask_rs.append(np.expand_dims(img_rs, axis=0))

        result = np.concatenate(mask_rs, axis=0)
        return result

    def collect_windows_result(self, w, h, coordinates, windows):
        num_classes = windows.shape[1]
        full_probs = np.zeros((num_classes, h, w))
        count_predictions = np.zeros((num_classes, h, w))
        for i, coor in enumerate(coordinates):
            x1, y1, x2, y2 = coor
            count_predictions[y1:y2, x1:x2] += 1
            average = windows[i]
            if full_probs[:, y1: y2, x1: x2].shape != average.shape:
                average = average[:, :y2 - y1, :x2 - x1]

            full_probs[:, y1:y2, x1:x2] += average

        full_probs = full_probs / count_predictions.astype(np.float)
        return full_probs

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

def _scale_and_flip_image(image, mean_std, scales=[1.0]):
    w, h = image.size
    new_images = []
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*mean_std)])
    for scale in scales:
        tg_w, tg_h = int(w * scale), int(h * scale)
        scaled_image = image.resize((tg_w, tg_h), Image.BILINEAR)
        flipped_image = scaled_image.transpose(Image.FLIP_LEFT_RIGHT)
        scaled_image = img_transform(scaled_image)
        flipped_image = img_transform(flipped_image)
        new_images.append([scaled_image, flipped_image])

    return ((w, h), new_images)

def _get_crops_image(image_data, scales=[1.0], crop_size=512, overlap=1 / 3):
    # image_data[0] is size of original image
    new_images = image_data[1]
    result = []
    # result = [
    # tensor1(row*col*2, 3, crop_size, crop_size) for scale 1, 2 in "row*col*2" for no-flip and flip
    # ]
    mapping = []
    # mapping =[
    # [w1, h1, [(x1, x2, y1, y2), (x1, x2, y1, y2), ...]], for scale 1
    # [w2, h2, [(x1, x2, y1, y2), (x1, x2, y1, y2), ...]], for scale 2
    # ...
    # ]

    for i, scale in enumerate(scales):
        scaled_image, flipped_image = new_images[i]
        h, w = scaled_image.shape[1:]
        tile_size = (int(scale * crop_size), int(scale * crop_size))
        stride = ceil(tile_size[0] * (1 - overlap))
        tile_rows = int(ceil((w - tile_size[0]) / stride) + 1)
        tile_cols = int(ceil((h - tile_size[1]) / stride) + 1)
        windows_image = []
        windows_flipped_image = []
        coordinates = [w, h, []]
        for row in range(tile_rows):
            for col in range(tile_cols):
                y1 = int(col * stride)
                x1 = int(row * stride)
                x2 = min(x1 + tile_size[1], w)
                y2 = min(y1 + tile_size[0], h)
                x1 = int(x2 - tile_size[1])
                y1 = int(y2 - tile_size[0])
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                coordinates[2].append((x1, y1, x2, y2))
                img_ts = scaled_image[:, y1:y2, x1:x2].unsqueeze(0)
                fl_img_ts = flipped_image[:, y1:y2, x1:x2].unsqueeze(0)
                windows_image.append(img_ts)
                windows_flipped_image.append(fl_img_ts)

        windows_image = torch.cat(windows_image, dim=0)
        windows_flipped_image = torch.cat(windows_flipped_image, dim=0)
        result.append(torch.cat([windows_image, windows_flipped_image], dim=0))
        mapping.append(coordinates)

    tensor_result = torch.cat(result, dim=0)
    return (image_data[0], mapping, tensor_result)
