"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.students.base_student import DistillationArgs
import copy
import collections
import torch

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
            self.logger.info('Pruning layer(s): ' + str(list(lambda x: x['name'], to_be_pruned_layers)))

        # get all layers (nn.Module object) in to_be_pruned_layers list by their names
        layers = [self.model.get_block(layer['name']) for layer in to_be_pruned_layers]

        # prune above layers and get the new blocks
        new_layers = []
        for idx, layer in enumerate(layers):
            compress_rate = self.compress_rate
            if 'compress_rate' in to_be_pruned_layers[idx]:
                compress_rate = to_be_pruned_layers[idx]['compress_rate']
            print(str(layer) + " compress rate: " + str(compress_rate))
            new_layers.append(self.pruner.prune(layer, compress_rate))

        # create new Distillation args
        args = []
        for i, new_layer in enumerate(new_layers):
            layer_name = to_be_pruned_layers[i]['name']
            args.append(DistillationArgs(layer_name, new_layer, layer_name))
            # reset optimizer state otherwise the momentum of adam will update teacher blocks even though
            # the gradient is 0
            # TODO: generalize this line to prune mulitple blocks at a time
            self.optimizer = self.config.init_object('optimizer', torch.optim, new_layer.parameters())

            # if lr is specified for each layer then use that lr otherwise use default lr of optimizer
            if 'lr' in to_be_pruned_layers[i]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = to_be_pruned_layers[i]['lr']

        # add new blocks to student model
        self.model.update_pruned_layers(args)
        self.logger.info('Number of trainable parameters after pruning: ' + str(self.model.dump_trainable_params()))
        self.logger.info(self.model.dump_student_teacher_blocks_info())

    def load_weight(self, checkpoint, pruner=None, pruning_plan=None, epoch=None):
        """
        load weights of pretrained pruned student model
        :return:
        """
        # use self.pruner or self.pruning_plan in config if the arguments aren't set
        if pruner is None:
            pruner = self.pruner
        if pruning_plan is None:
            pruning_plan = self.pruning_plan

        if epoch is None:
            # if epoch is none, then prune all the layer in pruning plan
            epochs = list(map(lambda x: x['epoch'], pruning_plan))
        else:
            # if epoch is specified only prune layers (in pruning plan) that would be prune at
            # i-th epoch where i<= epoch
            epochs = list(map(lambda x: x['epoch'], filter(lambda x: x['epoch'] <= epoch, pruning_plan)))

        # get ALL layers that will be pruned in this step
        to_be_pruned_layers = list(filter(lambda x: x['epoch'] in epochs, pruning_plan))

        # there isn't any layer would be prune
        if not to_be_pruned_layers:
            raise Exception("Expect at least one layer to be pruned but got 0.")

        # get all layers in to_be_pruned_layers list by their names
        layers = [self.model.get_block(layer['name']) for layer in to_be_pruned_layers]

        # prune above layers and get the new blocks
        new_layers = []
        for idx, layer in enumerate(layers):
            compress_rate = self.compress_rate
            if 'compress_rate' in to_be_pruned_layers[idx]:
                compress_rate = to_be_pruned_layers[idx]['compress_rate']
            print(str(layer) + " compress rate: " + str(compress_rate))
            new_layers.append(pruner.prune(layer, compress_rate))

        # create new Distillation args
        args = []
        for i, new_layer in enumerate(new_layers):
            # check if this layer is trainable or not
            trainable = to_be_pruned_layers[i]['trainable']
            if not trainable:
                for param in new_layer.parameters():
                    param.requires_grad = False

            layer_name = to_be_pruned_layers[i]['name']
            args.append(DistillationArgs(layer_name, new_layer, layer_name))

            # if lr is specified for each layer then use that lr otherwise use default lr of optimizer
            if trainable:
                optimizer_arg = copy.deepcopy(self.config['optimizer']['args'])
                if 'lr' in to_be_pruned_layers[i]:
                    optimizer_arg['lr'] = to_be_pruned_layers[i]['lr']
                self.optimizer.add_param_group({'params': new_layer.parameters(),
                                                **optimizer_arg})

        # add new blocks to student model
        self.model.update_pruned_layers(args)
        self.logger.info(self.model.dump_trainable_params())
        self.logger.info(self.model.dump_student_teacher_blocks_info())

        self.model.load_state_dict(checkpoint)

    def _train_epoch(self, epoch):
        self.prune(epoch)

        return super()._train_epoch(epoch)

