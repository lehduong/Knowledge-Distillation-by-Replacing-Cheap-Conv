from .unet import Unet
from .fpn import FPN
import logging
import importlib

def get_net(config, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=config['TA']['tc_arch'], num_classes=config['segmentation']['num_classes'], criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net

def get_model(network, num_classes, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)
    return net