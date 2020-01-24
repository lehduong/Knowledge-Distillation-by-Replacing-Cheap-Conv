from abc import ABC

import numpy as np
import torch
from torch import nn
from base import BaseTrainer


class BaseKnowledgeDistillationTrainer(BaseTrainer, ABC):
    """
    Base class for all Knowledge Distillation trainers
    """
    def __init__(self, student, teacher, criterions, metric_ftns, optimizer, config):
        # setup GPU device if available, move models into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.teacher = teacher.to(self.device)
        self.teacher.eval()
        self.lamb = config['KD']['lambda']

        super(BaseKnowledgeDistillationTrainer, self).__init__(student, criterions, metric_ftns, optimizer, config)

        # create alias to increase the readable of
        self.student = self.model
        del self.criterion


