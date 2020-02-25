import torch
import torch.nn as nn
from torchvision.utils import make_grid
from functools import reduce
from base import BaseTrainer
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker, save_image
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
import gc
import copy
import os
import numpy as np

class KnowledgeDistillationTrainer(BaseTrainer):
    """
    Base trainer class for knowledge distillation with unified teacher-student network
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, None, metric_ftns, optimizer, config)
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_validation_interval = self.config['trainer']['do_validation_interval']
        self.lr_scheduler = lr_scheduler
        self.weight_scheduler = weight_scheduler
        self.log_step = config['trainer']['log_step']
        if "len_epoch" in self.config['trainer']:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = self.config['trainer']['len_epoch']
        else:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)


        # also log teacher loss for comparision
        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        self.train_teacher_iou_metrics = CityscapesMetricTracker(writer=self.writer)

        self.valid_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        self.test_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        self.valid_teacher_iou_metrics = CityscapesMetricTracker(writer=self.writer)

        # Only used list of criterions and remove the unused property
        self.criterions = criterions
        self.criterions = nn.ModuleList(self.criterions).to(self.device)
        if isinstance(self.model, nn.DataParallel):
            self.criterions = nn.DataParallel(self.criterions)
        del self.criterion

        # early stop or prune
        self._teacher_student_iou_gap = 1

    def _clean_cache(self):
        self.model.student_hidden_outputs, self.model.teacher_hidden_outputs = None, None
        gc.collect()
        torch.cuda.empty_cache()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            supervised_loss = self.criterions[0](output_st, target)/self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc)/self.accumulation_steps

            # when computing the loss between output of teacher net and student net, we penalty the shallow layer
            # more than deep layer
            # the following loss will gradually increase the weight for former layer by exponential of gamma
            # i.e. (loss_layer5*gamma^3 + loss_layer8*gamma^2 + loss_layer12*gamma^1)/(gamma^3+gamma^2+gamma^1)
            gamma = self.weight_scheduler.gamma
            exponent_magnitude = list(range(1, 1+len(self.model.teacher_hidden_outputs)))
            normalized_term = reduce(lambda acc, elem: acc+gamma**elem, exponent_magnitude, 0)
            hint_loss = reduce(lambda acc, elem: acc+gamma**elem[2]*self.criterions[2](elem[0], elem[1]),
                             zip(self.model.student_hidden_outputs, self.model.teacher_hidden_outputs,
                                 exponent_magnitude),
                             0)/self.accumulation_steps/normalized_term

            #TODO: Early stop with teacher loss
            teacher_loss = self.criterions[0](output_tc, target) # for comparision

            alpha = self.weight_scheduler.alpha
            beta = self.weight_scheduler.beta
            loss = alpha * supervised_loss + beta * kd_loss + (1-alpha-beta)*hint_loss
            loss.backward()

            if (batch_idx+1) % self.accumulation_steps == 0:
                self.optimizer.step()
                if isinstance(self.lr_scheduler, MyOneCycleLR):
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if isinstance(self.lr_scheduler, MyReduceLROnPlateau) and \
                    (((batch_idx+1) % self.config['trainer']['lr_scheduler_step_interval']) == 0):
                # batch + 1 as the result of batch 0 always much smaller than other
                # don't know why ( ͡° ͜ʖ ͡°)
                self.lr_scheduler.step(self.train_metrics.avg('loss'))

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item()*self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item()*self.accumulation_steps)
            self.train_metrics.update('kd_loss', kd_loss.item()*self.accumulation_steps)
            self.train_metrics.update('hint_loss', hint_loss.item()*self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            self.train_iou_metrics.update(output_st.detach().cpu(), target.cpu())
            self.train_teacher_iou_metrics.update(output_tc.cpu(), target.cpu())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # st_masks = visualize.viz_pred_cityscapes(output_st)
                # tc_masks = visualize.viz_pred_cityscapes(output_tc)
                # self.writer.add_image('st_pred', make_grid(st_masks, nrow=8, normalize=False))
                # self.writer.add_image('tc_pred', make_grid(tc_masks, nrow=8, normalize=False))
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} mIoU: {:.6f} Teacher mIoU: {:.6f} Supervised Loss: {:.6f} Knowledge Distillation loss: '
                    '{:.6f} Hint Loss: {:.6f} Teacher Loss: {:.6f}'.format(
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

        if self.do_validation and ((epoch % self.do_validation_interval) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})

        self._teacher_student_iou_gap = self.train_teacher_iou_metrics.get_iou()-self.train_iou_metrics.get_iou()

        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                pass
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
        self.valid_metrics.reset()
        self.valid_iou_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output_aux, output = self.model.inference(data)
                supervised_loss = self.criterions[0](output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('supervised_loss', supervised_loss.item())
                self.valid_iou_metrics.update(output.detach().cpu(), target)

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of models parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _test_epoch(self, epoch):
        self.model.eval()
        self.test_metrics.reset()
        self.test_iou_metrics.reset()
        args = self.config['test']['args']
        save_4_sm = self.config['submission']['save_output']
        path_output = self.config['submission']['path_output']
        if save_4_sm and not os.path.exists(path_output):
            os.mkdir(path_output)

        with torch.no_grad():
            for batch_idx, (img_name, data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                # output = self.model.inference_test(data, args)
                output = self.model.inference(data)
                if save_4_sm:
                    self.save_for_submission(output, img_name[0])
                supervised_loss = self.criterions[0](output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'test')
                self.test_metrics.update('supervised_loss', supervised_loss.item())
                self.test_iou_metrics.update(output.detach().cpu(), target)

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of models parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.test_metrics.result()

    def save_for_submission(self, output, image_name, img_type=np.uint8):
        args = self.config['submission']
        path_output = args['path_output']
        image_save = '{}.{}'.format(image_name, args['ext'])
        path_save = os.path.join(path_output, image_save)
        result = torch.argmax(output, dim=1)
        result_mapped = self.re_map_for_submission(result)
        if output.size()[0] == 1:
            result_mapped = result_mapped[0]

        save_image(result_mapped.cpu().numpy().astype(img_type), path_save)
        print('Saved output of test data: {}'.format(image_save))

    def re_map_for_submission(self, output):
        mapping = self.valid_data_loader.dataset.id_to_trainid
        cp_output = torch.zeros(output.size())
        for k, v in mapping.items():
            cp_output[output == v] = k

        return cp_output

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

    def finetune(self):
        self.model._assign_blocks(False)
        self.teacher = copy.deepcopy(self.model)
        self.teacher.to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.model._assign_blocks(True)
        self.model.to_teacher()
        for param in self.model.parameters():
            param.requires_grad = True

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._finetune_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate models performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether models performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _finetune_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, _ = self.model(data)
            with torch.no_grad():
                output_tc = self.teacher(data)

            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc) / self.accumulation_steps
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            alpha = self.weight_scheduler.alpha
            loss = alpha * supervised_loss + (1-alpha) * kd_loss
            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
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
            self.train_iou_metrics.update(output_st.detach().cpu(), target.cpu())
            self.train_teacher_iou_metrics.update(output_tc.cpu(), target.cpu())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            if batch_idx % self.log_step == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                # st_masks = visualize.viz_pred_cityscapes(output_st)
                # tc_masks = visualize.viz_pred_cityscapes(output_tc)
                # self.writer.add_image('st_pred', make_grid(st_masks, nrow=8, normalize=False))
                # self.writer.add_image('tc_pred', make_grid(tc_masks, nrow=8, normalize=False))
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} mIoU: {:.6f} Teacher mIoU: {:.6f} Supervised Loss: {:.6f} Knowledge Distillation loss: '
                    '{:.6f} Teacher Loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        self.len_epoch,
                        self.train_metrics.avg('loss'),
                        self.train_iou_metrics.get_iou(),
                        self.train_teacher_iou_metrics.get_iou(),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('kd_loss'),
                        self.train_metrics.avg('teacher_loss'),
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()
        log.update({'train_teacher_mIoU': self.train_teacher_iou_metrics.get_iou()})
        log.update({'train_student_mIoU': self.train_iou_metrics.get_iou()})

        if self.do_validation and ((epoch % self.do_validation_interval) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})

        self._teacher_student_iou_gap = self.train_teacher_iou_metrics.get_iou() - self.train_iou_metrics.get_iou()

        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                pass
            else:
                self.lr_scheduler.step()

        self.weight_scheduler.step()

        return log
