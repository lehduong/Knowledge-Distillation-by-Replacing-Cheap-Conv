"""
Knowledge distillation via Pruning i.e. KDP
"""
from .kd_trainer import KnowledgeDistillationTrainer
from models.student import DistillationArgs, BLOCKS_LEVEL_SPLIT_CHAR


class KDPTrainer(KnowledgeDistillationTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """

    def __init__(self, model, criterions, pruner, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(self, model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, len_epoch)
        self.pruner = pruner
        self.pruning_plan = self.config['pruning']['pruning_plan']
        self.pruning_interval = self.config['pruning']['pruning_interval']

    def get_layers(self, layer_name):
        split_name = layer_name.split('.')
        layer = self.model
        for name in split_name:
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer,name)
        return layer

    def prune(self):
        # All layers in pruning plan have been pruned
        if not self.pruning_plan:
            return
        layer_names = self.pruning_plan.pop()
        layers = [self.get_layers(name) for name in layer_names]
        new_layer = self.pruner.prune(layers)

        # create new Distillation args
        args = []
        for i in range(len(layer_names)):
            args.append(DistillationArgs(layer_names[i], new_layer[i], layer_names[i]))

        self.model.update_pruned_layers(args)

    def _train_epoch(self, epoch):
        if epoch % self.pruning_interval == 0:
            self.prune()
        if self.do_validation:
            print('after pruning:')
            val_log = self._valid_epoch(epoch)

        super()._train_epoch(epoch)

