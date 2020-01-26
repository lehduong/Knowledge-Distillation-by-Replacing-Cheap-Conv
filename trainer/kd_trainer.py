import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from functools import reduce
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import gc

class KnowledgeDistillationTrainer(BaseTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, None, metric_ftns, optimizer, config)
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size*self.accumulation_steps))
        self.alpha = self.config['KD']['alpha']
        self.beta = self.config['KD']['beta']
        if self.alpha + self.beta > 1:
            raise Exception('Weight between supervised loss and div loss must not greater than 1')
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        # also log teacher loss for comparision
        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'div_loss', 'kd_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'supervised_loss', 'div_loss', 'kd_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # Only used list of criterions and remove the unused property
        self.criterions = criterions
        self.criterions = nn.ModuleList(self.criterions).to(self.device)
        if isinstance(self.model, nn.DataParallel):
            self.criterions = nn.DataParallel(self.criterions)
        del self.criterion

    def _clean_cache(self):
        hidden_st, hidden_tc = None, None
        self.model._student_hidden_outputs, self.model._teacher_hidden_outputs = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc, hidden_st, hidden_tc = self.model(data)
            
            supervised_loss = self.criterions[0](output_st, target)/self.accumulation_steps
            div_loss = self.criterions[1](output_st, output_tc)/self.accumulation_steps
            kd_loss = reduce(lambda acc, elem: acc+self.criterions[2](elem[0], elem[1]),
                             zip(hidden_st, hidden_tc),
                             0)/self.accumulation_steps
            #TODO: Early stop with teacher loss
            teacher_loss = self.criterions[0](output_tc, target) # for comparision

            loss = self.alpha * supervised_loss + self.beta * div_loss + (1-self.alpha-self.beta)*kd_loss
            loss.backward()
            self._clean_cache()

            if (batch_idx+1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item()*self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item()*self.accumulation_steps)
            self.train_metrics.update('div_loss', div_loss.item()*self.accumulation_steps)
            self.train_metrics.update('kd_loss', kd_loss.item()*self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.logger.debug(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} Supervised Loss: {:.6f} Divergence loss: {:.6f} Knowledge '
                    'Distillation Loss: {:.6f} Teacher Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        len(self.train_data_loader),
                        self.train_metrics.avg('loss'),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('div_loss'),
                        self.train_metrics.avg('kd_loss'),
                        self.train_metrics.avg('teacher_loss')
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

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
                output_st, output_tc, hidden_st, hidden_tc = self.model(data)

                supervised_loss = self.criterions[0](output_st, target)
                div_loss = self.criterions[1](output_st, output_tc)
                kd_loss = reduce(lambda acc, elem: acc + self.criterions[2](elem[0], elem[1]),
                                 zip(hidden_st, hidden_tc),
                                 0)
                teacher_loss = self.criterions[0](output_tc, target)  # for comparision
                loss = self.alpha * supervised_loss + self.beta * div_loss + (1 - self.alpha - self.beta) * kd_loss
                self._clean_cache()

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('supervised_loss', supervised_loss.item())
                self.valid_metrics.update('div_loss', div_loss.item())
                self.valid_metrics.update('kd_loss', kd_loss.item())
                self.valid_metrics.update('teacher_loss', teacher_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output_st, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of models parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
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
