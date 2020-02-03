"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.student import DistillationArgs, BLOCKS_LEVEL_SPLIT_CHAR
from functools import reduce


class MKDPTrainer(KnowledgeDistillationTrainer):
    """
    Knowledge distillation via mimic last layers + filter pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None, len_epoch=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler, len_epoch)
        self.pruner = pruner
        self.pruning_plan = self.config['pruning']['pruning_plan']
        self.pruning_interval = self.config['pruning']['pruning_interval']

    def prune(self):
        # All layers in pruning plan have been pruned
        if not self.pruning_plan:
            return

        # get ALL layers that will be pruned in this step
        layer_names = self.pruning_plan.pop()
        layers = [self.model.get_block(name) for name in layer_names]

        # prune assigned layers by keep a fixed number of filters and combine new 1x1 conv to retain the old shape
        new_layers = self.pruner.prune(layers)

        # create new Distillation args
        args = []
        for i in range(len(layer_names)):
            args.append(DistillationArgs(layer_names[i], new_layers[i], layer_names[i]))
            self.optimizer.add_param_group({'params': new_layers[i].parameters(),
                                            **self.config['optimizer']['args']})
        self.model.update_pruned_layers(args)
        print(self.model)

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
        if (epoch-1) % self.pruning_interval == 0:
            self.prune()

        return super()._train_epoch(epoch)

