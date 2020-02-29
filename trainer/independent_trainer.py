from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker
from .atakdp import ATAKDPTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from utils.util import EarlyStopTracker
from utils import optim as module_optim


class IndependentTrainer(ATAKDPTrainer):
    """
    Independent teaching assistant knowledge distillation pruning
    """
    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           'aux_loss',*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_iou_tracker = EarlyStopTracker('best', 'max', 0.01, 'rel')
        self._complete = False # indicate that all layer has been trained separately

    def _train_epoch(self, epoch):
        # unfreeze
        if not self._complete:
            # saved previous layer
            self.model.reset()

            # reset optimizer
            self.optimizer = self.config.init_obj('optimizer', module_optim, self.model.parameters())
            self.lr_scheduler = self.config.init_obj('lr_scheduler', module_optim.lr_scheduler, self.optimizer)

            # find the first layer that will be pruned afterward and set its pruned epoch to current epoch
            idxes = self.get_index_of_pruned_layer(epoch)
            if len(idxes) == 0:
                self.logger.info("All layers have been trained, now unfreeze all of them and finetune again...")
                self.model.restore()
                self.logger.debug(self.model.dump_student_teacher_blocks_info())
                self._complete = True
                self.optimizer = self.config.init_obj('optimizer', module_optim, self.model.student_blocks.parameters())
                self.len_epoch = 200
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['optimizer']['args']['lr']/10
            else:
                for idx in idxes:
                    self.pruning_plan[idx]['epoch'] = epoch
                self.prune(epoch)

            # reset lr scheduler o.w. the lr of new layer would be constantly reduced
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.reset()
            self.weight_scheduler.reset()
            self.val_iou_tracker.reset()

        # trivial trainer
        self.model.train()
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc) / self.accumulation_steps
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            # hint loss
            gamma = self.weight_scheduler.gamma
            exponent_magnitude = list(range(1, 1 + len(self.model.teacher_hidden_outputs)))
            normalized_term = reduce(lambda acc, elem: acc + gamma ** elem, exponent_magnitude, 0)
            hint_loss = reduce(lambda acc, elem: acc + gamma ** elem[2] * self.criterions[2](elem[0], elem[1]),
                               zip(self.model.student_hidden_outputs, self.model.teacher_hidden_outputs,
                                   exponent_magnitude),
                               0) / self.accumulation_steps / normalized_term

            beta = self.weight_scheduler.beta
            if self._complete:
                loss = kd_loss
            else:
                loss = hint_loss
            loss.backward()

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()
                if isinstance(self.lr_scheduler, MyOneCycleLR):
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item() * self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item() * self.accumulation_steps)
            self.train_metrics.update('kd_loss', kd_loss.item() * self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            self.train_metrics.update('hint_loss', hint_loss.item()*self.accumulation_steps)
            self.train_iou_metrics.update(output_st.detach().cpu(), target.cpu())
            self.train_teacher_iou_metrics.update(output_tc.cpu(), target.cpu())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} mIoU: {:.6f} Teacher mIoU: {:.6f} Supervised Loss: {:.6f} '
                    'Knowledge Distillation loss: {:.6f} Hint Loss: {:.6f} Teacher Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        self.len_epoch,
                        self.train_metrics.avg('loss'),
                        self.train_iou_metrics.get_iou(),
                        self.train_teacher_iou_metrics.get_iou(),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('kd_loss'),
                        self.train_metrics.avg('hint_loss'),
                        self.train_metrics.avg('teacher_loss'),
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        log.update({'train_teacher_mIoU': self.train_teacher_iou_metrics.get_iou()})
        log.update({'train_student_mIoU': self.train_iou_metrics.get_iou()})

        if self._complete and self.do_validation and ((epoch % self.config["trainer"]["do_validation_interval"]) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})
            self.val_iou_tracker.update(self.valid_iou_metrics.get_iou())

        # step lr scheduler
        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.step(self.train_metrics.avg('loss'))
            else:
                self.lr_scheduler.step()

        # step weight between losses
        self.weight_scheduler.step()

        return log
