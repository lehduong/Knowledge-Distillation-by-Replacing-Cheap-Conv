"""
Knowledge distillation via Pruning with Teacher Assistant
"""
from .kdp_trainer import KDPTrainer
import numpy as np


class TAKDPTrainer(KDPTrainer):
    """
    Knowledge distillation with teacher assistant + filter pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None, len_epoch=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler, len_epoch)
        self.ta_interval = self.config['teaching_assistant']['interval']
        self.ta_tol = self.config['teaching_assistant']['tol']

    def _train_epoch(self, epoch):
        if self._teacher_student_iou_gap < self.ta_tol:
            # transfer student to teaching assistant
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

        return super()._train_epoch(epoch)

