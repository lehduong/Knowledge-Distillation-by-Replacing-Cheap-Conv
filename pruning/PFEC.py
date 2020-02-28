# Pruning Filters for Efficient Convnets
# https://arxiv.org/abs/1608.08710
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from torch import nn
from base import BasePruner


class PFEC(BasePruner):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.config = config
        # pruning rate per layer using norm criterion
        self.compress_rate = self.config['pruning']['compress_rate']
        # transform block params
        self.dilation = self.config['pruning']['pruner']['dilation']
        self.kernel_size = self.config['pruning']['pruner']['kernel_size']
        self.padding = self.config['pruning']['pruner']['padding']

    def norm_based_pruning(self, layer, num_kept_filter):
        """
        Pruning conv and fc layer using norm criterion i.e. those filters that have smallest norm would be removed
        :param num_kept_filter: float - the ratio of number of kept filters and number of all filters
        :param layer: nn.Conv2d or nn.Linear - layer of network
        :return: a new layer
        """
        # construct new layer with identical weights with old layer but having smaller number of filters
        if type(layer) is nn.Conv2d:
            new_layer = nn.Conv2d(layer.in_channels, num_kept_filter, layer.kernel_size, layer.stride,
                                  layer.padding, layer.dilation, layer.groups, layer.bias, layer.padding_mode)
        elif type(layer) is nn.Linear:
            new_layer = nn.Linear(layer.in_features, num_kept_filter, layer.bias)
        else:
            raise Exception("Unsupported type of layer, expect it to be nn.Conv2d or nn.Linear but got: " +
                            str(type(layer)))

        weight = layer.weight.data
        weight_norm = torch.norm(weight.view(weight.shape[0], -1), 2, 1)

        # index of the top k norm filters
        idx_kept_filter = torch.topk(weight_norm, num_kept_filter, sorted=False)[1].cpu().numpy().astype(np.int32)

        # copy the weight
        new_layer.weight.data = weight[idx_kept_filter]

        if self.use_cuda:
            new_layer = new_layer.cuda()

        return new_layer

    def transform_block(self, inp_channels, layer):
        """
        create a block that transform a pruned layer to the same number of filter
        :param inp_channels: int - number of channels of input
        :param layer: nn.Module - the unpruned layer
        :return:
        """
        out_channels = layer.out_channels
        ret = nn.Sequential(
            nn.Conv2d(inp_channels, inp_channels, kernel_size=self.kernel_size, padding=self.padding,
                      dilation=self.dilation, groups=inp_channels),
            nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        )
        # self._initializing(layer, ret[0])
        # self._initializing(layer, ret[1])

        if self.use_cuda:
            ret = ret.cuda()
        return ret

    def _initializing(self, reference_layer, new_layer):
        mean = reference_layer.weight.data.mean().item()
        std = reference_layer.weight.data.std().item()
        new_layer.weight.data.normal_(mean, std)

    def prune(self, layer, compress_rate=None):
        """
        suppose
        :param compress_rate: float - the ratio of number of kept filters must range from 0 to 1
        :param layer: nn.Module - a Conv2d layer that need to be pruned
        :return: block which is replaceable for input layers.
        """
        if compress_rate is None:
            compress_rate = self.compress_rate

        num_kept_filter = int(compress_rate * layer.weight.shape[0])
        if num_kept_filter > 0:
            new_layer = self.norm_based_pruning(layer, num_kept_filter)
            # keep weight of pre-trained layer
            for param in new_layer.parameters():
                param.requires_grad = True
            transform_block = TransformBlock(num_kept_filter, layer, new_layer, self.kernel_size, self.padding,
                                             self.dilation, self.use_cuda)
            #transform_block = self.transform_block(num_kept_filter, layer)
            #return nn.Sequential(new_layer, transform_block)
        else:
            transform_block = DepthwiseSeparableBlock(layer.in_channels, layer.out_channels, self.kernel_size,
                                                      self.padding, self.dilation, layer.in_channels, False)
        return transform_block


class TransformBlock(nn.Module):
    def __init__(self, num_kept_filter, layer, new_layer, kernel_size, padding, dilation, use_cuda=True):
        super().__init__()
        self.num_kept_filter = num_kept_filter
        self.pruned_layer = new_layer
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.use_cuda = use_cuda
        self.residual_block = self.create_transform_block(self.pruned_layer.in_channels, self.pruned_layer)
        self.transform_block = self.create_transform_block(self.num_kept_filter, layer)
        self.relu = nn.LeakyReLU(0.5)

    def create_transform_block(self, inp_channels, layer):
        """
        create a block that transform a pruned layer to the same number of filter
        :param inp_channels: int - number of channels of input
        :param layer: nn.Module - the unpruned layer
        :return:
        """
        out_channels = layer.out_channels
        ret = nn.Sequential(
            nn.Conv2d(inp_channels, inp_channels, kernel_size=self.kernel_size, padding=self.padding,
                      dilation=self.dilation, groups=inp_channels),
            nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        )
        # self._initializing(layer, ret[0])
        # self._initializing(layer, ret[1])

        if self.use_cuda:
            ret = ret.cuda()
        return ret

    def _initializing(self, reference_layer, new_layer):
        mean = reference_layer.weight.data.mean().item()
        std = reference_layer.weight.data.std().item()
        new_layer.weight.data.normal_(mean, std)

    def forward(self, x):
        out1 = self.pruned_layer(x)
        out2 = self.residual_block(x)
        out = out1+out2
        # out = self.relu(out)
        out = self.transform_block(out)
        return out


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, groups, bias, use_cuda=True):
        super().__init__()
        self.separable_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation,
                                        groups=groups, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        if use_cuda:
            self.separable_conv.cuda()
            self.pointwise_conv.cuda()

    def initialize(self, reference_layer):
        mean = reference_layer.weight.data.mean().item()
        std = reference_layer.weight.data.std().item()
        self.separable_conv.weight.data.normal_(mean, std)
        self.pointwise_conv.weight.data.normal_(mean, std)

    def forward(self, x):
        x = self.separable_conv(x)
        x = self.pointwise_conv(x)
        return x
