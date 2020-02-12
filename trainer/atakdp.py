from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker
from .takdp_trainer import TAKDPTrainer
from  utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
import numpy as np


class ATAKDPTrainer(TAKDPTrainer):
    """
    Auxiliary loss teaching assistant knowledge distillation pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss','aux_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        # Teaching assistant
        if (self._teacher_student_iou_gap < self.ta_tol) or ((self._ta_count % self.ta_interval) == 0):
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
                print('Early stop as there is not any layer to be pruned...')
                return {}

            self.pruning_plan[idx]['epoch'] = epoch

            # dump the new teacher:
            self.logger.debug('Promoted Student to Teaching Assistant')
            number_of_param = sum(p.numel() for p in self.model.parameters())
            self.logger.debug('Number of parameters: ' + str(number_of_param))

            self._ta_count = 0
            self.weight_scheduler.reset()

            # remove old auxiliary hooks
            self.model.flush_aux_layers()

            # update auxiliary loss
            sorted_pruning_plan = sorted(self.pruning_plan, key=lambda x: x['epoch'])
            sorted_layer_names = list(map(lambda x: x['name'], sorted_pruning_plan))
            pruned_layer_name = self.pruning_plan[idx]['name']
            pruned_layer_name_idx = sorted_layer_names.index(pruned_layer_name)
            num = self.config['pruning']['auxiliary_num_layers']
            if pruned_layer_name_idx > (len(sorted_layer_names) - num):
                aux_layer_names = sorted_layer_names[pruned_layer_name_idx+1:]
            else:
                aux_layer_names = sorted_layer_names[pruned_layer_name_idx+1: pruned_layer_name_idx+num+1]
            self.model.update_aux_layers(aux_layer_names)

        self._ta_count += 1

        # pruning
        self.prune(epoch)

        # trivial trainer
        self.model.train()
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = self.criterions[1](output_st, output_tc) / self.accumulation_steps

            # when computing the loss between output of teacher net and student net, we penalty the shallow layer
            # more than deep layer
            # the following loss will gradually increase the weight for former layer by exponential of gamma
            # i.e. (loss_layer5*gamma^3 + loss_layer8*gamma^2 + loss_layer12*gamma^1)/(gamma^3+gamma^2+gamma^1)
            gamma = self.weight_scheduler.gamma
            exponent_magnitude = list(range(1, 1 + len(self.model.teacher_hidden_outputs)))
            normalized_term = reduce(lambda acc, elem: acc + gamma ** elem, exponent_magnitude, 0)
            hint_loss = reduce(lambda acc, elem: acc + gamma ** elem[2] * self.criterions[2](elem[0], elem[1]),
                               zip(self.model.student_hidden_outputs, self.model.teacher_hidden_outputs,
                                   exponent_magnitude),
                               0) / self.accumulation_steps / normalized_term

            # auxiliary loss:
            # when computing the loss between output of teacher net and student net, we penalty the shallow layer
            # more than deep layer
            # the following loss will gradually increase the weight for former layer by exponential of gamma
            # i.e. (loss_layer5*gamma^3 + loss_layer8*gamma^2 + loss_layer12*gamma^1)/(gamma^3+gamma^2+gamma^1)
            aux_loss = reduce(lambda acc, elem: acc + self.criterions[2](elem[0], elem[1]),
                               zip(self.model.student_aux_outputs, self.model.teacher_aux_outputs),
                               0) / self.accumulation_steps

            # TODO: Early stop with teacher loss
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            alpha = self.weight_scheduler.alpha
            beta = self.weight_scheduler.beta
            loss = hint_loss+aux_loss
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
            self.train_metrics.update('hint_loss', hint_loss.item() * self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            self.train_metrics.update('aux_loss', aux_loss.item())
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

        # TODO: Fix out of memory when runing validation
        # if self.do_validation:
        #     # clean cache to prevent out-of-memory with 1 gpu
        #     self._clean_cache()
        #     val_log = self._valid_epoch(epoch)
        #     log.update(**{'val_' + k: v for k, v in val_log.items()})
        #     log.update(**{'val_mIoU': self.valid_iou_metrics.get_iou()})

        self._teacher_student_iou_gap = self.train_teacher_iou_metrics.get_iou() - self.train_iou_metrics.get_iou()

        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.step(self.train_metrics.avg('loss'))
            else:
                self.lr_scheduler.step()

        self.weight_scheduler.step()

        return log

    def _clean_cache(self):
        self.model.student_aux_outputs, self.model.teacher_aux_outputs = None, None
        super()._clean_cache()

