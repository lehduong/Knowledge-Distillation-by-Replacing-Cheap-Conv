from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker
from .takdp_trainer import TAKDPTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from utils.util import EarlyStopTracker
from pruning import PFEC
import numpy as np


class ATAKDPTrainer(TAKDPTrainer):
    """
    Auxiliary loss teaching assistant knowledge distillation pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           'aux_loss',*[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_iou_tracker = EarlyStopTracker('best', 'max', 0.01, 'rel')

    def _train_epoch(self, epoch):
        # Teaching assistant
        if (self._teacher_student_iou_gap < self.ta_tol) or ((self._ta_count % self.ta_interval) == 0) or \
                (not self.val_iou_tracker.last_update_success):
            # transfer student to teaching assistant
            self.model.to_teacher()
            # dump the new teacher:
            self.logger.info('Promoted Student to Teaching Assistant')
            number_of_param = sum(p.numel() for p in self.model.parameters())
            self.logger.info('Number of parameters: ' + str(number_of_param))

            # find the first layer that will be pruned afterward and set its pruned epoch to current epoch
            idxes = self.get_index_of_pruned_layer(epoch)
            for idx in idxes:
                self.pruning_plan[idx]['epoch'] = epoch

            # reset lr scheduler o.w. the lr of new layer would be constantly reduced
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.reset()
            self._ta_count = 0
            self.weight_scheduler.reset()
            self.val_iou_tracker.reset()

        self._ta_count += 1

        # pruning
        self.prune(epoch)
        # update layers of auxiliary loss
        aux_layer_names = self.get_aux_layer_names(epoch)
        if len(aux_layer_names) > 0:
            self.model.update_aux_layers(aux_layer_names)
            self.logger.debug('Auxiliary layers including: ' + str(self.model.aux_layer_names))

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
            if len(self.model.student_aux_outputs) > 0:
                aux_loss = reduce(lambda acc, elem: acc + self.criterions[2](elem[0], elem[1]),
                                  zip(self.model.student_aux_outputs, self.model.teacher_aux_outputs),
                                  0) / self.accumulation_steps / len(self.model.student_aux_outputs)
            else:
                aux_loss = 0

            # TODO: Early stop with teacher loss
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            alpha = self.weight_scheduler.alpha
            beta = self.weight_scheduler.beta
            if len(self.model.student_aux_outputs) > 0:
                loss = beta * hint_loss + (1 - beta) * aux_loss
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
            self.train_metrics.update('hint_loss', hint_loss.item() * self.accumulation_steps)
            self.train_metrics.update('teacher_loss', teacher_loss.item())
            if len(self.model.student_aux_outputs) > 0:
                self.train_metrics.update('aux_loss', aux_loss.item())
            self.train_iou_metrics.update(output_st.detach().cpu(), target.cpu())
            self.train_teacher_iou_metrics.update(output_tc.cpu(), target.cpu())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            # if isinstance(self.lr_scheduler, MyReduceLROnPlateau) and \
            #         (((batch_idx+1) % self.config['trainer']['lr_scheduler_step_interval']) == 0):
            #     # batch + 1 as the result of batch 0 always much smaller than other
            #     # don't know why ( ͡° ͜ʖ ͡°)
            #     self.lr_scheduler.step(self.train_metrics.avg('loss'))

            if batch_idx % self.log_step == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
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

        # step weight between losses
        self.weight_scheduler.step()

        return log

    def _clean_cache(self):
        self.model.student_aux_outputs, self.model.teacher_aux_outputs = None, None
        super()._clean_cache()

    def get_aux_layer_names(self, epoch):
        # sort pruning plan according to epoch (smallest to largest)
        sorted_pruning_plan = sorted(self.pruning_plan, key=lambda x: x['epoch'])

        # names of layer in (sorted pruning plan)
        sorted_layer_names = list(map(lambda x: x['name'], sorted_pruning_plan))

        # names of layers that will be pruned in this epoch
        pruned_layer_names = list(map(lambda x: x['name'], filter(lambda x: x['epoch'] == epoch, self.pruning_plan)))

        # indexes of layers that will be pruned in this epoch in pruning_plan list
        pruned_layer_name_idxes = [sorted_layer_names.index(pruned_layer_name) for pruned_layer_name in pruned_layer_names]
        # largest index in previous list
        if len(pruned_layer_name_idxes) == 0:
            return []
        pruned_layer_name_idx = max(pruned_layer_name_idxes)

        num = self.config['pruning']['auxiliary_num_layers']
        if pruned_layer_name_idx > (len(sorted_layer_names) - num):
            aux_layer_names = sorted_layer_names[pruned_layer_name_idx + 1:]
        else:
            aux_layer_names = sorted_layer_names[pruned_layer_name_idx + 1: pruned_layer_name_idx + num + 1]

        return aux_layer_names

    def get_index_of_pruned_layer(self, epoch):
        # prune_epoch_to_now = np.array(list(map(lambda x: x['epoch'], self.pruning_plan))) - epoch
        # idx = -1
        # min_value = np.inf
        # for i in range(len(prune_epoch_to_now)):
        #     if min_value > prune_epoch_to_now[i] >= 0:
        #         idx = i
        #         min_value = prune_epoch_to_now[i]
        # if idx < 0:
        #     raise Exception('Early stop as there is not any layer to be pruned...')
        # return idx

        unpruned_layers = list(filter(lambda x: x['epoch'] >= epoch, self.pruning_plan))
        unpruned_layers_epoch = np.array(list(map(lambda x: x['epoch'], unpruned_layers)))
        prune_epoch_to_now = unpruned_layers_epoch-epoch
        soonest_layer_idxes = np.where(prune_epoch_to_now == prune_epoch_to_now.min())[0]
        soonest_layer_names = list()
        for i in soonest_layer_idxes:
            soonest_layer_names.append(unpruned_layers[i]['name'])

        pruning_plan_names = list(map(lambda x: x['name'], self.pruning_plan))
        idxes = []
        for soonest_layer_name in soonest_layer_names:
            idxes.append(pruning_plan_names.index(soonest_layer_name))

        return idxes

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        # prune old model
        self.pruner = PFEC(self.model, checkpoint['config'])
        for i in range(checkpoint['epoch']+1):
            self.prune(i)
        self.model.to_teacher()
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

