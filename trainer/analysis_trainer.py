from .layerwise_trainer import LayerwiseTrainer
from models.students import AnalysisStudent
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau 
from functools import reduce 

class AnalysisTrainer(LayerwiseTrainer):
    def __init__(self, model: AnalysisStudent, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler)


    def train(self):
        for layer in self.config['layer_compressible']:
            layer_name = layer['layer_name']
            lrs = layer['lrs']
            args = layer['args']

            for lr in lrs:
                self.logger.info(f'Replacing layer: {layer_name} learning rate: {lr:.6f}')
                # Replace chosen layer
                self.model.replace([layer_name], **args)
                # Register hints layer
                self.model.register_hint_layers([layer_name])
                # Unfreeze
                # self.model.unfreeze([layer_name])
                # reset scheduler
                self.reset_scheduler()
                # create new optimizer and set its learning rate 
                self.create_new_optimizer()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                # logging 
                self.logger.info(self.model.dump_trainable_params())
                self.logger.info(self.model.dump_student_teacher_blocks_info())
                # start finetuning 
                for epoch in range(1, self.epochs):
                    self._train_epoch(epoch, lr=lr, layer_name=layer_name)
                self.model.reset()

    def _train_epoch(self, epoch, **kwargs):
        # reset
        # self.model.student.train()
        self.model.save_hidden = True  # hack: save hidden output if training is set to true
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
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
            loss = hint_loss
            loss.backward()

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()
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
                # save the result to tensorboard
                # miou, loss, miou gap between teacher and student network
                self.writer.add_scalars("mIoU/" + kwargs['layer_name'].replace('.', '_'), {str(kwargs['lr']): self.train_iou_metrics.get_iou()}, batch_idx)
                self.writer.add_scalars("loss/" + kwargs['layer_name'].replace('.', '_'), {str(kwargs['lr']): loss.item()}, batch_idx)
                iou_gap = self.train_teacher_iou_metrics.get_iou() - self.train_iou_metrics.get_iou()
                self.writer.add_scalars("student_teacher_iou_gap/" + kwargs['layer_name'].replace('.', '_'), { str(kwargs['lr']): iou_gap}, batch_idx)
                # logging result
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] Loss: {:.6f} mIoU: {:.6f} Teacher mIoU: {:.6f} Supervised Loss: {:.6f} '
                    'Knowledge Distillation loss: '
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

        if self.do_validation and ((epoch % self.config["trainer"]["do_validation_interval"]) == 0):
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
                self.logger.debug('stepped lr')
                for param_group in self.optimizer.param_groups:
                    self.logger.debug(param_group['lr'])
                    
        # anneal weight between losses
        self.weight_scheduler.step()

        return log

