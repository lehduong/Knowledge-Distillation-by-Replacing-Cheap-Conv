import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
import losses as module_loss
import models.metric as module_metric
import models.cifar_models as module_arch
import utils.optim as module_optim
from models.students import DepthwiseStudent, AnalysisStudent
from parse_config import ConfigParser
from trainer import ClassificationTrainer, EnsembleTrainer
from utils import WeightScheduler

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    train_data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = config.init_obj('test_data_loader', module_data)

    # Load pretrained teacher model
    teacher = config.restore_snapshot('teacher', module_arch)
    teacher = teacher.cpu()  # saved some memory as student network will use a (deep) copy of teacher model

    # build models architecture, then print to console
    if config['trainer']['name'] == 'ClassificationTrainer':
        student = DepthwiseStudent(teacher, config)
    elif config['trainer']['name'] == 'EnsembleTrainer':
        student = DepthwiseStudent(teacher, config)
    elif config['trainer']['name'] == 'AnalysisTrainer':
        student = AnalysisStudent(teacher, config)
    else:
        raise NotImplementedError("Supported: Layerwise Trainer")

    # get function handles of loss and metrics
    supervised_criterion = config.init_obj('supervised_loss', module_loss)
    kd_criterion = config.init_obj('kd_loss', module_loss)
    hint_criterion = config.init_obj('hint_loss', module_loss)
    criterions = [supervised_criterion, kd_criterion, hint_criterion]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer',
                                module_optim,
                                student.student.parameters()
                                )
    lr_scheduler = config.init_obj('lr_scheduler', module_optim.lr_scheduler, optimizer)
    # create weight scheduler to anneal the weights between losses
    weight_scheduler = WeightScheduler(config['weight_scheduler'])

    # Run trainer
    if config['trainer']['name'] == 'ClassificationTrainer':
        trainer = ClassificationTrainer(student, criterions, metrics, optimizer, config, train_data_loader,
                                        valid_data_loader, lr_scheduler, weight_scheduler)
    elif config['trainer']['name'] == 'EnsembleTrainer':
        trainer = EnsembleTrainer(student, criterions, metrics, optimizer, config, train_data_loader,
                                  valid_data_loader, lr_scheduler, weight_scheduler)
    else:
        raise NotImplementedError("Only support Classification, Ensemble Trainer")

    if config['trainer']['name'] == 'EnsembleTrainer':
        trainer.test()
        return 
        
    trainer.train()

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Knowledge Distillation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)