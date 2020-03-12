from .classification_trainer import ClassificationTrainer
from utils.optim.lr_scheduler import MyOneCycleLR, MyReduceLROnPlateau
from functools import reduce
from utils import MetricTracker
from models import forgiving_state_restore
from torch import nn
import torch
import copy

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
        self.resume(self.config['resume_paths'])

    def resume(self, checkpoint_paths):
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
            
        self.logger.info('loaded state dict for all models')

    def _train_epoch(self, epoch):
        raise NotImplementedError("Ensemble Trainer only support test method...")

    def _valid_epoch(self, epoch):
        raise NotImplementedError("Ensemble Trainer only support test method...")

    def _test_epoch(self, epoch):
        """
        Test after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for model in self.models:
            model.eval()
        self.test_metrics.reset()
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                # Unweighted average predictions from list of models 
                output = self.models[0](data)
                for model in self.models[1:]:
                    tmp, _ = model(data)
                    tmp = softmax(tmp)
                    output = output + tmp
                output = output / len(self.models)
                # Update Metrics
                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'valid')
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))

        return self.test_metrics.result()
