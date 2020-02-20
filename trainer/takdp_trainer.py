"""
Knowledge distillation via Pruning with Teacher Assistant
"""
from .kdp_trainer import KDPTrainer
from models.students.base_student import DistillationArgs
from models import forgiving_state_restore
import numpy as np
import torch


class TAKDPTrainer(KDPTrainer):
    """
    Knowledge distillation with teacher assistant + filter pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)
        self.ta_interval = self.config['teaching_assistant']['interval'] # maximum number of epochs for training a TA
        self.ta_tol = self.config['teaching_assistant']['tol'] # error tolerance for TA promotion
        self._ta_count = 1 # interval for TA promotion
        self._trained_ta_layers = []  # name of TA layers that have been trained and promoted to TA
        self._training_ta_layers = []  # name of pruning layers and haven't been promoted to TA yet
        if 'resume_path' in self.config['trainer']:
            self._resume_checkpoint(self.config['trainer']['resume_path'])

    def _train_epoch(self, epoch):
        if (self._teacher_student_iou_gap < self.ta_tol) or ((self._ta_count % self.ta_interval) == 0):
            # transfer student to teaching assistant
            trained_ta_layers = list(map(lambda x: x.old_block_name, self.model.distillation_args))
            self._trained_ta_layers += trained_ta_layers
            self.model.to_teacher()

            # find the soonest layer that will be pruned and prune it now
            prune_epoch_to_now = np.array(list(map(lambda x: x['epoch'], self.pruning_plan)))-epoch
            idx = -1
            min = np.inf
            for i in range(len(prune_epoch_to_now)):
                if min > prune_epoch_to_now[i] >= 0:
                    idx = i
                    min = prune_epoch_to_now[i]
            if idx < 0:
                print('Early stop as student mIoU is close enough to teacher')
                return {}

            self.pruning_plan[idx]['epoch'] = epoch

            # dump the new teacher:
            self.logger.debug('Promoted Student to Teaching Assistant')
            number_of_param = sum(p.numel() for p in self.model.parameters())
            self.logger.debug('Number of parameters: ' + str(number_of_param))

            self._ta_count = 0
            self.weight_scheduler.reset()

        self._ta_count += 1
        return super()._train_epoch(epoch)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        training_ta_layers = list(map(lambda x: x.old_block_name, self.model.distillation_args))
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'trained_ta_layers': self._trained_ta_layers,
            'training_ta_layers': training_ta_layers
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # first, transform the student to have identical to checkpoint architecture
        # TODO: Generalize pruner
        pruner = self.pruner
        pruning_plan = checkpoint['config']['pruning']['pruning_plan']
        pruning_plan_dict = {elem["name"]: elem["compress_rate"] for elem in pruning_plan}
        trained_ta_layers = [self.model.get_block(layer) for layer in checkpoint['trained_ta_layers']]
        training_ta_layers = [self.model.get_block(layer) for layer in checkpoint['training_ta_layers']]
        self.logger.info('trained: '+str(checkpoint['trained_ta_layers']))
        self.logger.info('training: '+str(checkpoint['training_ta_layers']))
        # Prune trained TA layers
        new_layers = []
        self.logger.info('Begin pruning trained TA layers')
        for idx, layer in enumerate(trained_ta_layers):
            layer_name = checkpoint['trained_ta_layers'][idx]
            compress_rate = pruning_plan_dict[layer_name]['compress_rate']
            self.logger.info(str(layer) + " compress rate: " + str(compress_rate))
            new_layers.append(pruner.prune(layer, compress_rate))
        args = []
        for i, new_layer in enumerate(new_layers):
            for param in new_layer.parameters():
                param.requires_grad = False
            layer_name = checkpoint['trained_ta_layers'][idx]
            args.append(DistillationArgs(layer_name, new_layer, layer_name))

            optimizer_arg = checkpoint['config']['optimizer']['args']
            self.optimizer.add_param_group({'params': new_layer.parameters(),
                                            **optimizer_arg})
        # TODO: Causing problem if not call assign_blocks, will fix later
        self.model.update_pruned_layers(args)
        self.model._assign_blocks(True)
        self.model.to_teacher()

        # Prune training TA layers
        new_layers = []
        self.logger.info('Begin pruning training TA layers')
        for idx, layer in enumerate(training_ta_layers):
            layer_name = checkpoint['training_ta_layers'][idx]
            compress_rate = pruning_plan_dict[layer_name]
            self.logger.info(str(layer) + " compress rate: " + str(compress_rate))
            new_layers.append(pruner.prune(layer, compress_rate))
        args = []
        for i, new_layer in enumerate(new_layers):
            for param in new_layer.parameters():
                param.requires_grad = True
            layer_name = checkpoint['training_ta_layers'][idx]
            args.append(DistillationArgs(layer_name, new_layer, layer_name))

            optimizer_arg = checkpoint['config']['optimizer']['args']
            self.optimizer.add_param_group({'params': new_layer.parameters(),
                                            **optimizer_arg})
        # load state dict
        self.model.update_pruned_layers(args)
        self.logger.debug(self.model)
        forgiving_state_restore(self.model, checkpoint['state_dict'])
        self.logger.info("Loaded model's state dict successfully")

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def eval(self):
        self.model.to_teacher()
        self.logger.debug(self.model.model)
        result = self._valid_epoch(1)

        # save logged informations into log dict
        log = {}
        log.update(result)
        log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})

        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))