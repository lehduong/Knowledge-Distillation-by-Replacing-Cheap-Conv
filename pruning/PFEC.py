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
    def __init__(self, model, config, compress_rate):
        super().__init__(model, config)
        self.config = config
        # pruning rate per layer using norm criterion
        self.compress_rate = compress_rate
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
        idx_kept_filter = torch.topk(weight_norm, num_kept_filter)[1].cpu().numpy().astype(np.int32)

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
        self._initializing(layer, ret[0])
        self._initializing(layer, ret[1])

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
        new_layer = self.norm_based_pruning(layer, num_kept_filter)
        # keep weight of pre-trained layer
        for param in new_layer.parameters():
            param.requires_grad = False
        transform_block = self.transform_block(num_kept_filter, layer)

        return nn.Sequential(new_layer, transform_block)
