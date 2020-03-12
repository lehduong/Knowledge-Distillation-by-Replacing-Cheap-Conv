from torchvision.utils import make_grid
from functools import reduce
from utils import inf_loop, MetricTracker, visualize, CityscapesMetricTracker, save_image, ImportanceFilterTracker
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from utils.util import EarlyStopTracker
from utils import optim as optim_module
from models.students import DepthwiseStudent
from models import forgiving_state_restore
from base import BaseTrainer
from utils import stat_cuda
from torch import nn
import numpy as np
import os
import gc
import torch


class TaylorPruneTrainer(BaseTrainer):
    """
    Trainer
    """

    def __init__(self, model: DepthwiseStudent, criterions, metric_ftns, optimizer, config, train_data_loader,
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
        self.importance_log_interval = self.config['trainer']['importance_log_interval']
        if "len_epoch" in self.config['trainer']:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = self.config['trainer']['len_epoch']
        else:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)

        # Metrics
        # Train
        self.train_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        self.train_teacher_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        # Valid
        self.valid_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                           *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_iou_metrics = CityscapesMetricTracker(writer=self.writer)
        # Test
        self.test_metrics = MetricTracker('loss', 'supervised_loss', 'kd_loss', 'hint_loss', 'teacher_loss',
                                          *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_iou_metrics = CityscapesMetricTracker(writer=self.writer)

        # Tracker for early stop if val miou doesn't increase
        self.val_iou_tracker = EarlyStopTracker('best', 'max', 0.01, 'rel')
        # Tracker for importance of filter in layer
        self.importance_tracker = ImportanceFilterTracker(writer=self.writer)

        # Only used list of criterions and remove the unused property
        self.criterions = criterions
        self.criterions = nn.ModuleList(self.criterions).to(self.device)
        if isinstance(self.model, nn.DataParallel):
            self.criterions = nn.DataParallel(self.criterions)
        del self.criterion

        # Resume checkpoint if path is available in config
        if 'resume_path' in self.config['trainer']:
            self.resume(self.config['trainer']['resume_path'])

    def prepare_train_epoch(self, epoch, config=None):
        """
        Prepare before training an epoch i.e. prune new layer, unfreeze some layers, create new optimizer ....
        :param epoch:  int - indicate which epoch the trainer's in
        :param config: a config object that contain pruning_plan, hint, unfreeze information
        :return:
        """
        # if the config is not set (training normaly, then set config to current trainer config)
        # if the config is set (in case you're resuming a checkpoint) then use saved config to replace
        #    layers in student so that it would have identical archecture with saved checkpoint
        if config is None:
            config = self.config

            # Check if there is any layer that would any update in current epoch
        # list of epochs that would have an update on student networks
        epochs = list(map(lambda x: x['epoch'], config['pruning']['pruning_plan']))
        # if there isn't any update then simply return
        if epoch not in epochs:
            self.logger.info('EPOCH: ' + str(epoch))
            self.logger.info('There is no update ...')
            return

        # there is at least 1 layer would be replaced/add as hint/unfreeze then:
        # freeze all previous layers
        # TODO: Verify if we should freeze previous layer or not
        # self.logger.debug('Freeze all weight of student network')
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # layers that would be replaced by depthwise separable conv
        replaced_layers = list(filter(lambda x: x['epoch'] == epoch,
                                      config['pruning']['pruning_plan'])
                               )

        self.logger.info('EPOCH: ' + str(epoch))
        self.logger.info('Replaced layers: ' + str(replaced_layers))
        # Avoid error when loading deprecate checkpoint which don't have 'args' in config.pruning
        if 'args' in config['pruning']:
            kwargs = config['pruning']['args']
        else:
            self.logger.warning('Using deprecate checkpoint...')
            kwargs = config['pruning']['pruner']

        self.model.replace(replaced_layers, **kwargs)  # replace those layers with depthwise separable conv
        # initialize importance vector for layer
        self.importance_tracker.update_importance_list(self.model.added_gates)

        # TODO: Verify if we should unfreeze the trained layer or not
        if epoch == 1:
            self.create_new_optimizer()  # create new optimizer to remove the effect of momentum
        else:
            self.update_optimizer(list(filter(lambda x: x['epoch'] == epoch, config['pruning']['unfreeze'])))

        self.logger.info(self.model.dump_trainable_params())
        self.logger.info(self.model.dump_student_teacher_blocks_info())
        self.reset_scheduler()

    def update_optimizer(self, unfreeze_config):
        """
        Update param groups for optimizer with unfreezed layers of this epoch
        :param unfreeze_config - list of arg. Each arg is the dictionary with following format:
            {'name': 'layer1', 'epoch':1, 'lr'(optional): 0.01}
        return:
        """
        self.logger.debug('Updating optimizer for new layer')
        for config in unfreeze_config:
            layer_name = config['name']  # layer that will be unfreezed
            self.logger.debug('Add parameters of layer: {} to optimizer'.format(layer_name))

            layer = self.model.get_block(layer_name, self.model.student)  # actual layer i.e. nn.Module obj
            optimizer_arg = self.config['optimizer']['args']  # default args for optimizer

            # we can also specify layerwise learning !
            if "lr" in config:
                optimizer_arg['lr'] = config['lr']
            # add unfreezed layer's parameters to optimizer
            self.optimizer.add_param_group({'params': layer.parameters(),
                                            **optimizer_arg})

    def create_new_optimizer(self):
        """
        Create new optimizer if trainer is in epoch 1 otherwise just run update optimizer
        """
        # Create new optimizer
        self.logger.debug('Creating new optimizer ...')
        self.optimizer = self.config.init_obj('optimizer',
                                              optim_module,
                                              list(filter(lambda x: x.requires_grad, self.model.student.parameters())))
        self.lr_scheduler = self.config.init_obj('lr_scheduler',
                                                 optim_module.lr_scheduler,
                                                 self.optimizer)

    def reset_scheduler(self):
        """
        reset all schedulers, metrics, trackers, etc when unfreeze new layer
        :return:
        """
        self.weight_scheduler.reset()  # weight between loss
        self.val_iou_tracker.reset()  # verify val iou would increase each time
        self.train_metrics.reset()  # metrics for loss,... in training phase
        self.valid_metrics.reset()  # metrics for loss,... in validating phase
        self.train_iou_metrics.reset()  # train iou of student
        self.valid_iou_metrics.reset()  # val iou of student
        self.train_teacher_iou_metrics.reset()  # train iou of teacher
        if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
            self.lr_scheduler.reset()

    def _train_epoch(self, epoch):
        """
        Training logic for 1 epoch
        """
        # Prepare the network i.e. unfreezed new layers, replaced new layer with depthwise separable conv, ...
        self.prepare_train_epoch(epoch)

        # reset
        #self.model.train() 
        self.model.save_hidden = True
        self.train_metrics.reset()
        self.train_iou_metrics.reset()
        self.train_teacher_iou_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)

            # do not divide accumulation_steps to keep value of gradient
            supervised_loss = self.criterions[0](output_st, target)
            teacher_loss = self.criterions[0](output_tc, target)  # for comparision

            # Only use supervised loss
            loss = supervised_loss
            loss.backward()

            # Update tracker to track importance of filter in layer
            importance_dict = self.model.get_gate_importance()
            self.importance_tracker.update(importance_dict)

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item() * self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item() * self.accumulation_steps)
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

            if batch_idx % self.importance_log_interval == 0:
                importance_hitherto = self.importance_tracker.average()
                self.logger.info('Importance of filters in layers')
                for name, vector in importance_hitherto.items():
                    self.logger.info('{}: {}'.format(name, vector))
                filename = 'importance_filter_ep{}_batch_idx{}.pth'.format(epoch, batch_idx)
                file_path = os.path.join(self.checkpoint_dir, filename)
                torch.save(importance_hitherto, file_path)

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

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self._clean_cache()
        #self.model.eval()
        self.model.save_hidden = False
        self.valid_metrics.reset()
        self.valid_iou_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.inference(data)
                supervised_loss = self.criterions[0](output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('supervised_loss', supervised_loss.item())
                self.valid_iou_metrics.update(output.detach().cpu(), target)
                self.logger.debug(str(batch_idx) + " : " + str(self.valid_iou_metrics.get_iou()))

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
        result = self.valid_metrics.result()
        result['mIoU'] = self.valid_iou_metrics.get_iou()

        return result

    def _test_epoch(self, epoch):
        # cleaning up memory
        self._clean_cache()
        self.model.training = False
        self.model.cpu()
        self.model.student.to(self.device)

        # prepare before running submission
        self.test_metrics.reset()
        self.test_iou_metrics.reset()
        args = self.config['test']['args']
        save_4_sm = self.config['submission']['save_output']
        path_output = self.config['submission']['path_output']
        if save_4_sm and not os.path.exists(path_output):
            os.mkdir(path_output)
        n_samples = len(self.valid_data_loader)

        with torch.no_grad():
            for batch_idx, (img_name, data, target) in enumerate(self.valid_data_loader):
                self.logger.info('{}/{}'.format(batch_idx, n_samples))
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.inference_test(data, args)
                if save_4_sm:
                    self.save_for_submission(output, img_name[0])
                supervised_loss = self.criterions[0](output, target)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'test')
                self.test_metrics.update('supervised_loss', supervised_loss.item())
                self.test_iou_metrics.update(output.detach().cpu(), target)

                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))

        result = self.test_metrics.result()
        result['mIoU'] = self.test_iou_metrics.get_iou()

        return result

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

    def _clean_cache(self):
        self.model.student_hidden_outputs, self.model.teacher_hidden_outputs = list(), list()
        gc.collect()
        torch.cuda.empty_cache()

    def resume(self, checkpoint_path):
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        config = checkpoint['config']  # config of checkpoint
        epoch = checkpoint['epoch']  # stopped epoch

        # load model state from checkpoint
        # first, align the network by replacing depthwise separable for student
        for i in range(1, epoch + 1):
            self.prepare_train_epoch(i, config)
        # load weight
        forgiving_state_restore(self.model, checkpoint['state_dict'])
        self.logger.info("Loaded model's state dict")

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info("Loaded optimizer state dict")