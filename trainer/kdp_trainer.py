"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.students.base_student import DistillationArgs
from models import forgiving_state_restore
import copy
import collections
import torch
import torch.nn as nn


class KDPTrainer(KnowledgeDistillationTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)
        self.pruner = pruner
        self.pruning_plan = self.config['pruning']['pruning_plan']
        self.compress_rate = self.config['pruning']['compress_rate']

    def prune(self, epoch):
        # get ALL layers that will be pruned in this step
        to_be_pruned_layers = list(filter(lambda x: x['epoch'] == epoch, self.pruning_plan))

        # there isn't any layer would be pruned at this epoch
        if not to_be_pruned_layers:
            return
        else:
            # logging the layers being pruned
            self._ta_count = 1  # reset TA interval if using TA
            self.logger.info('Pruning layer(s): ' + str(list(map(lambda x: x['name'], to_be_pruned_layers))))

        # get all layers (nn.Module object) in to_be_pruned_layers list by their names
        layers = [self.model.get_block(layer['name']) for layer in to_be_pruned_layers]

        # prune above layers and get the new blocks
        new_layers = []
        for idx, layer in enumerate(layers):
            compress_rate = self.compress_rate
            if 'compress_rate' in to_be_pruned_layers[idx]:
                compress_rate = to_be_pruned_layers[idx]['compress_rate']
            print(str(layer) + " compress rate: " + str(compress_rate))
            new_layers.append(self.pruner.prune(layer, compress_rate=compress_rate))

        # create new Distillation args
        args = []
        for i, new_layer in enumerate(new_layers):
            layer_name = to_be_pruned_layers[i]['name']
            args.append(DistillationArgs(layer_name, new_layer, layer_name))

            # if lr is specified for each layer then use that lr otherwise use default lr of optimizer
            optimizer_arg = copy.deepcopy(self.config['optimizer']['args'])
            if 'lr' in to_be_pruned_layers[i]:
                optimizer_arg['lr'] = to_be_pruned_layers[i]['lr']
            # reset optimizer state otherwise the momentum of adam will update teacher blocks even though
            # the gradient is 0
            self.optimizer.state = collections.defaultdict(dict)

            # add new parameters to optimizer
            self.optimizer.add_param_group({'params': new_layer.parameters(),
                                            **optimizer_arg})
        # add new blocks to student model
        self.model.update_pruned_layers(args)
        self.logger.info('Number of trainable parameters after pruning: ' + str(self.model.dump_trainable_params()))
        self.logger.info(self.model.dump_student_teacher_blocks_info())

    def _train_epoch(self, epoch):
        self.prune(epoch)

        return super()._train_epoch(epoch)
