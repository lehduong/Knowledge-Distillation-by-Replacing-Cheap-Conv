from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker
from .kd_trainer import KnowledgeDistillationTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from utils.util import EarlyStopTracker
from utils import optim as optim_module
import numpy as np


class LayerwiseTrainer(KnowledgeDistillationTrainer):
    """
    Train each layer separately. Note that the later layer will be trained to reconstruct the TEACHER output, not the
        TA's one i.e. we have to use WrappedStudent instead of BaseStudent
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.val_iou_tracker = EarlyStopTracker('best', 'max', 0.01, 'rel')

    def prepare_train_epoch(self, epoch):
        """
        Prepare before training an epoch i.e. prune new layer, create new optimizer ....
        :param epoch:
        :return:
        """
        # Check if there is any layer that would be replaced in this epoch
        # list of epochs that would have an update on student networks
        epochs = list(map(lambda x: x['epoch'], self.config['pruning']['pruning_plan']))
        # if not:
        if epoch not in epochs:
            self.logger.info('EPOCH: ' + str(epoch))
            self.logger.info('There is no update ...')
            return

        # there is at least 1 layer would be replaced then:
        # freeze all previous layers
        for param in self.model.parameters():
            param.requires_grad = False

        # layers that would be replaced by depthwise separable conv
        replaced_layers = list(map(lambda x: x['name'],
                                   filter(lambda x: x['epoch'] == epoch,
                                          self.config['pruning']['pruning_plan'])
                                   )
                               )
        # layers which outputs will be used as loss
        hint_layers = list(map(lambda x: x['name'],
                               filter(lambda x: x['epoch'] == epoch,
                                      self.config['pruning']['hint'])
                               )
                           )
        # layers that would be trained in this epoch
        unfreeze_layers = list(map(lambda x: x['name'],
                                   filter(lambda x: x['epoch'] == epoch,
                                          self.config['pruning']['unfreeze'])
                                   )
                               )
        self.logger.info('EPOCH: ' + str(epoch))
        self.logger.info('Replaced layers: ' + str(replaced_layers))
        self.logger.info('Hint layers: ' + str(hint_layers))
        self.logger.info('Unfreeze layers: ' + str(unfreeze_layers))
        self.model.replace(replaced_layers)
        self.model.register_hint_layers(hint_layers)
        self.model.unfreeze(unfreeze_layers)

        # Create new optimizer
        self.optimizer = self.config.init_obj('optimizer',
                                              optim_module,
                                              list(filter(lambda x: x.requires_grad, self.model.student.parameters())))
        self.logger.info(self.model.dump_trainable_params())
        self.logger.info(self.model.dump_student_teacher_blocks_info())
        self.reset_scheduler()

    def reset_scheduler(self):
        """
        reset all schedulers, metrics, trackers, etc
        :return:
        """
        self.weight_scheduler.reset()  # weight between loss
        self.val_iou_tracker.reset()  # verify val iou would increase each time
        self.train_metrics.reset()  # metrics for loss,... in training phase
        self.valid_metrics.reset()  # metrics for loss,... in validating phase
        self.train_iou_metrics.reset()  # train iou of student
        self.valid_iou_metrics.reset()  # val iou of student
        self.train_teacher_iou_metrics.reset()  # train iou of teacher
        self.lr_scheduler.reset()

    def _train_epoch(self, epoch):
        # replace chosen layers in this epoch
        self.prepare_train_epoch(epoch)

        # reset
        self.model.student.train()
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc) / self.accumulation_steps

            hint_loss = reduce(lambda acc, elem: acc + elem[2] * self.criterions[2](elem[0], elem[1]),
                               zip(self.model.student_hidden_outputs, self.model.teacher_hidden_outputs),
                               0) / self.accumulation_steps

            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            # Only use hint loss
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
            self.train_metrics.update('hint_loss', hint_loss.item() * self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            self.train_iou_metrics.update(output_st.detach().cpu(), target.cpu())
            self.train_teacher_iou_metrics.update(output_tc.cpu(), target.cpu())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # st_masks = visualize.viz_pred_cityscapes(output_st)
                # tc_masks = visualize.viz_pred_cityscapes(output_tc)
                # self.writer.add_image('st_pred', make_grid(st_masks, nrow=8, normalize=False))
                # self.writer.add_image('tc_pred', make_grid(tc_masks, nrow=8, normalize=False))
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} mIoU: {:.6f} Teacher mIoU: {:.6f} Supervised Loss: {:.6f} '
                    'Knowledge Distillation loss: '
                    '{:.6f} Hint Loss: {:.6f} Aux Loss: {:.6f} Teacher Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        self.len_epoch,
                        self.train_metrics.avg('loss'),
                        self.train_iou_metrics.get_iou(),
                        self.train_teacher_iou_metrics.get_iou(),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('kd_loss'),
                        self.train_metrics.avg('hint_loss'),
                        self.train_metrics.avg('aux_loss'),
                        self.train_metrics.avg('teacher_loss'),
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        log.update({'train_teacher_mIoU': self.train_teacher_iou_metrics.get_iou()})
        log.update({'train_student_mIoU': self.train_iou_metrics.get_iou()})

        if self.do_validation and ((epoch % self.config["trainer"]["do_validation_interval"]) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})
            self.val_iou_tracker.update(self.valid_iou_metrics.get_iou())

        self._teacher_student_iou_gap = self.train_teacher_iou_metrics.get_iou() - self.train_iou_metrics.get_iou()

        # step lr scheduler
        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.step(self.train_metrics.avg('loss'))
            else:
                self.lr_scheduler.step()

        # anneal weight between losses
        self.weight_scheduler.step()

        return log

    def _clean_cache(self):
        self.model.student_aux_outputs, self.model.teacher_aux_outputs = None, None
        super()._clean_cache()
