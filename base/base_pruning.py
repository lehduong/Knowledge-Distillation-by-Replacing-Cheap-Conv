from abc import ABC, abstractmethod


class BasePruner(ABC):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.use_cuda = config['n_gpu'] > 0

    @abstractmethod
    def prune(self, layers):
        pass