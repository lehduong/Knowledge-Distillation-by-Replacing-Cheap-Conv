"""
Knowledge distillation via Pruning with Teacher Assistant
"""
from .kdp_trainer import KDPTrainer
from models.student import BaseStudent

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
        if ((epoch+1) % self.ta_interval == 0) or (self._teacher_student_iou_gap < self.ta_tol):
            self.model.to_teacher()
            print('Promoted Student to Teaching Assistant')
            print('Number of parameters: ' + str(BaseStudent.__get_number_param(self.model)))

        return super()._train_epoch(epoch)

