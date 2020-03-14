from .classification_trainer import ClassificationTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from functools import reduce
from utils import MetricTracker
from models import forgiving_state_restore
from torch import nn
import torch
import copy
from models.cifar_models import resnet20, resnet56,resnet110
WEIGHT = 0.1
class EnsembleTrainer(ClassificationTrainer):
    """
        Used to run evaluation only
            Currently only support train classification
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, weight_scheduler=None, test_data_loader=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config, train_data_loader,
                         valid_data_loader, lr_scheduler, weight_scheduler, test_data_loader)
        if not 'resume_paths' in self.config['trainer']:
            raise ValueError("Cannot find path to checkpoints, please specify them by adding 'resume_paths' in config.trainer")
        self.models = []
        self.resume_ensemble(self.config['trainer']['resume_paths'])

    def resume_ensemble(self, checkpoint_paths):
        for i, checkpoint_path in enumerate(checkpoint_paths):
            self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            config = checkpoint['config']  # config of checkpoint
            epoch = checkpoint['epoch']  # stopped epoch

            # load model state from checkpoint
            # first, align the network by replacing depthwise separable for student
            for i in range(1, epoch+1):
                self.prepare_train_epoch(i, config)
            # load weight
            forgiving_state_restore(self.model, checkpoint['state_dict'])
            self.model._remove_hooks()
            self.logger.info("Loaded state dict for model {}".format(i))
            # stor the student network
            self.models.append(self.model.student)
            # rewind the student network back to teacher
            self.model.student = copy.deepcopy(self.model.teacher)
            self.model.replaced_block_names = []
        self.logger.info('loaded state dict for all models')

    def prepare_models(self, epoch):
        # freeze the teacher and unfreeze student
        self.logger.debug('Freeze teacher and Unfreeze student networks')
        for param in self.model.student.parameters():
            param.requires_grad = True 
        for param in self.model.teacher.parameters():
            param.requires_grad = False 
        # set teacher and other ensemble model to eval() 
        self.logger.debug('Set ensemble model to eval mode and student model to train mode')
        for model in self.models:
            model.eval()
        self.model.teacher.eval()
        # student to train()
        self.model.student.train()
        # disable hint layer
        self.model.save_hidden = False 
        if epoch == 1:
            self.create_new_optimizer()
            self.logger.debug(self.model.student)

    def _train_epoch(self, epoch):
        self.prepare_models(epoch)
        self.train_metrics.reset()
        self._clean_cache()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output_st, output_tc = self.model(data)
            with torch.no_grad():
                outputs = []
                for model in self.models:
                    outputs += [model(data)]
            supervised_loss = self.criterions[0](output_st, target) / self.accumulation_steps
            kd_loss = reduce(lambda acc, elem: acc + WEIGHT*self.criterions[1](output_st, elem), outputs, 0) 
            kd_loss += self.criterions[1](output_st, output_tc)
            kd_loss = kd_loss/ (WEIGHT*len(outputs)+1) / (self.accumulation_steps)
            # Only use hint loss
            loss = kd_loss+supervised_loss
            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # update metrics
            self.train_metrics.update('loss', loss.item() * self.accumulation_steps)
            self.train_metrics.update('supervised_loss', supervised_loss.item() * self.accumulation_steps)
            self.train_metrics.update('kd_loss', kd_loss.item() * self.accumulation_steps)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_st, target))

            for met in self.metric_ftns:
                self.train_teacher_metrics.update(met.__name__, met(output_tc, target))

            if batch_idx % self.log_step == 0:
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                self.logger.info(
                    'Train Epoch: {} [{}]/[{}] acc: {:.6f} teacher_acc: {:.6f} Loss: {:.6f} Supervised Loss: {:.6f} '
                    'Knowledge Distillation loss: {:.6f}'.format(
                        epoch,
                        batch_idx,
                        self.len_epoch,
                        self.train_metrics.avg('accuracy'),
                        self.train_teacher_metrics.avg('accuracy'),
                        self.train_metrics.avg('loss'),
                        self.train_metrics.avg('supervised_loss'),
                        self.train_metrics.avg('kd_loss'),
                    ))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation and ((epoch % self.do_validation_interval) == 0):
            # clean cache to prevent out-of-memory with 1 gpu
            self._clean_cache()
            # single student model 
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            # ensemble
            tc_log = self._test_epoch(epoch)
            log.update(**{'ensemble_' + k: v for k, v in tc_log.items()})

        if (self.lr_scheduler is not None) and (not isinstance(self.lr_scheduler, MyOneCycleLR)):
            if isinstance(self.lr_scheduler, MyReduceLROnPlateau):
                self.lr_scheduler.step(self.train_metrics.avg('loss'))
            else:
                self.lr_scheduler.step()

        self.weight_scheduler.step()

        return log

    def ensemble_predict(self, data, weight=WEIGHT):
        """
        :param data: Tensor of shape (Bx3xHxW)
        :param weight: weight of student predictions
        :return: ACTIVATED Tensor of shape (BxC) for classification or (BxCxHxW) for segmentation
                by activated, I mean the tensor have already gone through softmax layer
        """
        # classification
        # TODO: Only compatible with classification
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            # predict of teacher network
            output = softmax(self.model.teacher(data))
            # enhance the teacher prediction with student networks
            for model in self.models:
                tmp = softmax(model(data))
                output = output + weight*tmp
            # normalize 
            output = output/(1+len(self.models)*weight)
        return output 

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
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _test_epoch(self, epoch):
        """
        Testing with ensemble models
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for model in self.models:
            model.eval()
        self.model.teacher.eval()
        self.test_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.ensemble_predict(data)
                # Update Metrics
                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))

        return self.test_metrics.result()
