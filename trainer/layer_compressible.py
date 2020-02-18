from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker
from .kdp_trainer import KDPTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from utils.util import EarlyStopTracker
from models.students.base_student import DistillationArgs
import copy
import collections


class LayerCompressibleTrainer(KDPTrainer):
    """
    Auxiliary loss teaching assistant knowledge distillation pruning
    """

    def __init__(self, model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, pruner, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)

        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           'aux_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.val_iou_tracker = EarlyStopTracker('best', 'max', 0.01, 'rel')

    def train_epoch(self, epoch, layer_name, compress_rate, lr):

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

            # TODO: Early stop with teacher loss
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            alpha = self.weight_scheduler.alpha
            beta = self.weight_scheduler.beta
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
                self.writer.add_scalars("mIoU/" + layer_name, {str(lr): self.train_iou_metrics.get_iou()})
                self.writer.add_scalars("loss/" + layer_name, {str(lr): loss.item()})
                iou_gap = self.train_teacher_iou_metrics.get_iou() - self.train_iou_metrics.get_iou()
                self.writer.add_scalars("student_teacher_iou_gap/" + layer_name, { str(lr): iou_gap})
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

        # step weight between losses
        self.weight_scheduler.step()

        return log

    def train(self):
        # pruning
        for layer in self.config['layer_compressible']:
            layer_name = layer['layer_name']
            compress_rate = layer['compress_rate']
            lrs = layer['lrs']
            self.logger.info(self.model.dump_student_teacher_blocks_info())
            for lr in lrs:
                self.prune(layer_name, compress_rate, lr)
                self.logger.info(f'Pruning layer: {layer_name} learning rate: {lr:.6f} compress rate: {compress_rate:.2f}')
                for epoch in range(1, self.epochs):
                    self.train_epoch(epoch, layer_name, compress_rate, lr)
                self.model.reset()

    def prune(self, layer_name, compress_rate=0, lr=0.001):
        # get all layers (nn.Module object) in to_be_pruned_layers list by their names
        layer = self.model.get_block(layer_name)

        # prune above layers and get the new blocks
        new_layer = self.pruner.prune(layer, compress_rate)

        # create new Distillation args
        args = [DistillationArgs(layer_name, new_layer, layer_name)]

        # if lr is specified for each layer then use that lr otherwise use default lr of optimizer
        optimizer_arg = copy.deepcopy(self.config['optimizer']['args'])
        optimizer_arg['lr'] = lr
        # reset optimizer state otherwise the momentum of adam will update teacher blocks even though
        # the gradient is 0
        self.optimizer.state = collections.defaultdict(dict)

        # add new parameters to optimizer
        self.optimizer.add_param_group({'params': new_layer.parameters(),
                                        **optimizer_arg})
        # add new blocks to student model
        self.model.update_pruned_layers(args)
