"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.student import DistillationArgs, BLOCKS_LEVEL_SPLIT_CHAR


class KDPTrainer(KnowledgeDistillationTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, len_epoch)
        self.pruner = pruner
        self.pruning_plan = self.config['pruning']['pruning_plan']
        self.pruning_interval = self.config['pruning']['pruning_interval']

    def prune(self):
        # All layers in pruning plan have been pruned
        if not self.pruning_plan:
            return
        layer_names = self.pruning_plan.pop()
        layers = [self.model.get_block(name) for name in layer_names]
        new_layers = self.pruner.prune(layers)

        # create new Distillation args
        args = []
        for i in range(len(layer_names)):
            args.append(DistillationArgs(layer_names[i], new_layers[i], layer_names[i]))
            self.optimizer.add_param_group({'params': new_layers[i].parameters(),
                                            **self.config['optimizer']['args']})
        self.model.update_pruned_layers(args)
        print(self.model)

    def _train_epoch(self, epoch):
        if (epoch-1) % self.pruning_interval == 0:
            self.prune()

        return super()._train_epoch(epoch)

