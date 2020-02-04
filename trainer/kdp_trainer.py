"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.student import DistillationArgs, BLOCKS_LEVEL_SPLIT_CHAR
from functools import reduce
import copy


class KDPTrainer(KnowledgeDistillationTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None, len_epoch=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler, len_epoch)
        self.pruner = pruner
        self.pruning_plan = self.config['pruning']['pruning_plan']
        self.compress_rate = self.config['pruning']['compress_rate']

    def prune(self, epoch):
        # get ALL layers that will be pruned in this step
        to_be_pruned_layers = list(filter(lambda x: x['epoch'] == epoch, self.pruning_plan))

        # there isn't any layer would be pruned at this epoch
        if not to_be_pruned_layers:
            return

        # get all layers in to_be_pruned_layers list by their names
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

            # if lr is specified for each layer then use that lr otherwise use default lr of optimizer
            optimizer_arg = copy.deepcopy(self.config['optimizer']['args'])
            if 'lr' in to_be_pruned_layers[i]:
                optimizer_arg['lr'] = to_be_pruned_layers[i]['lr']
            self.optimizer.add_param_group({'params': new_layer.parameters(),
                                            **optimizer_arg})
        # add new blocks to student model
        self.model.update_pruned_layers(args)
        print(self.model.dump_student_teacher_blocks_info())

    def load_weight(self, checkpoint, pruner=None, pruning_plan=None):
        """
        load weights of pretrained pruned student model
        :return:
        """
        # use self.pruner or self.pruning_plan in config if the arguments aren't set
        if pruner is None:
            pruner = self.pruner
        if pruning_plan is None:
            pruning_plan = self.pruning_plan

        # get all new layers using above pruner & pruning_plan
        layers = []
        for i in range(len(pruning_plan)):
            layer_names = pruning_plan[i]
            layers += [self.model.get_block(name) for name in layer_names]
        new_layers = pruner.prune(layers)
        layer_names = reduce(lambda acc, elem: acc+elem, pruning_plan, [])

        # create distilation_args
        args = []
        for i in range(len(new_layers)):
            args.append(DistillationArgs(layer_names[i], new_layers[i], layer_names[i]))
            self.optimizer.add_param_group({'params': new_layers[i].parameters(),
                                            **self.config['optimizer']['args']})

        # update the model architecture
        self.model.update_pruned_layers(args)

        # load the checkpoint
        self.model.load_state_dict(checkpoint)

    def _train_epoch(self, epoch):
        self.prune(epoch)

        return super()._train_epoch(epoch)

