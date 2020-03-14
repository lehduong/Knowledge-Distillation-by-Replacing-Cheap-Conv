from .layerwise_trainer import LayerwiseTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau 
from functools import reduce
from utils import MetricTracker 
import torch 

class ClassificationTrainer(LayerwiseTrainer):
    def __init__(self, model, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None, test_data_loader=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.train_teacher_metrics = MetricTracker(*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_data_loader = test_data_loader

    def _train_epoch(self, epoch):
        self.prepare_train_epoch(epoch)

        self.model.train()
        self.model.save_hidden = True
        self.train_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc) / self.accumulation_steps

            hint_loss = reduce(lambda acc, elem: acc + self.criterions[2](elem[0], elem[1]),
                               zip(self.model.student_hidden_outputs, self.model.teacher_hidden_outputs),
                               0) / self.accumulation_steps

            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            # Only use hint loss
            loss = supervised_loss + kd_loss
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item() * self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item() * self.accumulation_steps)
            self.train_metrics.update('kd_loss', kd_loss.item() * self.accumulation_steps)
            self.train_metrics.update('hint_loss', hint_loss.item() * self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            for met in self.metric_ftns:
                self.train_teacher_metrics.update(met.__name__, met(output_tc, target))

            if batch_idx % self.log_step == 0:
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] acc: {:.6f} teacher_acc: {:.6f} Loss: {:.6f} Supervised Loss: {:.6f} '
                    'Knowledge Distillation loss: {:.6f} Hint Loss: {:.6f} Teacher Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        self.len_epoch,
                        self.train_metrics.avg('accuracy'),
                        self.train_teacher_metrics.avg('accuracy'),
                        self.train_metrics.avg('loss'),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('kd_loss'),
                        self.train_metrics.avg('hint_loss'),
                        self.train_metrics.avg('teacher_loss'),
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation and ((epoch % self.do_validation_interval) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

            if self.test_data_loader:
                self._clean_cache()
                test_log = self._test_epoch(epoch)
                log.update(**{'test_' + k: v for k, v in test_log.items()})

        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.step(self.train_metrics.avg('loss'))
            else:
                self.lr_scheduler.step()

        self.weight_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.model.save_hidden = False 
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                supervised_loss = self.criterions[0](output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('supervised_loss', supervised_loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.model.save_hidden = False
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'valid')
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))

        return self.test_metrics.result()