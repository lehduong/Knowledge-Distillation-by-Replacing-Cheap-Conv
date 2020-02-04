import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
import losses as module_loss
import models.metric as module_metric
import models as module_arch
from models.student import BaseStudent
from data_loader import _create_transform
from parse_config import ConfigParser
from trainer import KDPTrainer, TAKDPTrainer
from pruning import PFEC
from utils import WeightScheduler

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_joint_transform, train_input_transform, target_transform, val_input_transform = _create_transform(config)
    train_data_loader = config.init_obj('train_data_loader', module_data, transform=train_input_transform,
                                        transforms=train_joint_transform, target_transform=target_transform)
    valid_data_loader = config.init_obj('val_data_loader', module_data, transform=val_input_transform,
                                        target_transform=target_transform)

    # Load pretrained teacher model
    teacher = config.restore_snapshot('teacher', module_arch)
    teacher = teacher.cpu() # saved some memory as student network will use a (deep) copy of teacher model

    # build models architecture, then print to console
    args = []
    student = BaseStudent(teacher, args)
    logger.info(student)

    # get function handles of loss and metrics
    supervised_criterion = config.init_obj('supervised_loss', module_loss)
    kd_criterion = config.init_obj('kd_loss', module_loss)
    hint_criterion = config.init_obj('hint_loss', module_loss)
    criterions = [supervised_criterion, kd_criterion, hint_criterion]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, student.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    # create weight scheduler to anneal the weights between losses
    weight_scheduler = WeightScheduler(config['weight_scheduler'])

    # Knowledge Distillation only
    pruner = PFEC(student, config, config['pruning']['compress_rate'])
    if config['trainer']['name'] == "TAKDPTrainer":
        trainer = TAKDPTrainer(student, pruner, criterions, metrics, optimizer, config, train_data_loader,
                               valid_data_loader, lr_scheduler, weight_scheduler)
    elif config['trainer']['name'] == 'KDPTrainer':
        trainer = KDPTrainer(student, pruner, criterions, metrics, optimizer, config, train_data_loader,
                            valid_data_loader, lr_scheduler, weight_scheduler)
    else:
        raise Exception("Unsupported trainer")

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
