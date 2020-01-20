from abc import ABC

import numpy as np
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from models.loss import KLDivergenceLoss


class BaseKnowledgeDistillationTrainer(BaseTrainer, ABC):
    """
    Base class for all Knowledge Distillation trainers
    """

    def __init__(self, student, teacher, criterion, kd_criterion, metric_ftns, optimizer, config):
        # setup GPU device if available, move models into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.teacher = teacher.to(self.device)
        # TODO: a bug in the implementation that requires the DeepWV3Plus having wrapped by
        #  nn.DataParallel to give reasonable ouput
        self.teacher = nn.DataParallel(teacher)
        self.teacher.eval()
        # TODO: making it's easier to configurate the distil_criterion
        self.kd_criterion = kd_criterion
        self.lamb = config['KD']['lambda']
        super(BaseKnowledgeDistillationTrainer, self).__init__(student, criterion, metric_ftns, optimizer, config)
        # create alias to increase the readable of
        self.student = self.model


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps
            loss.backward()
            if (batch_idx+1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self._logging(batch_idx, epoch, data, output, target, loss)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of models parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _logging(self, batch_idx, epoch, data, output, target, loss):
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.train_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output, target))

        if batch_idx % self.log_step == 0:
            self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx),
                loss.item()))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))


class TrainerTeacherAssistant(BaseKnowledgeDistillationTrainer, BaseTrainer):
    """
       Trainer use TA technique 
    """

    def __init__(self, student, teacher, criterion, kd_criterion, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):

        super(TrainerTeacherAssistant, self).__init__(student, teacher, criterion, kd_criterion, metric_ftns, optimizer,
                                                config)
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        self.student.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_tc = self.teacher(data)
            # TODO: Find an elegant way to free the feature map and computation graph
            output_tc = torch.tensor(output_tc.detach().cpu().numpy()).cuda()
            
            output_st = self.student(data)
            supervised_loss = self.criterion(output_st, target)
            kd_loss = self.kd_criterion(output_st, output_tc)

            loss = (1 - self.lamb) * supervised_loss + self.lamb* kd_loss
            loss = loss / self.accumulation_steps
            loss.backward()

            if (batch_idx+1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of models parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
