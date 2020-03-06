from torchvision import datasets
from torchvision import transforms as tfs
from base import BaseDataLoader
from .cityscapes import Cityscapes, CityScapesUniform
from torch.utils.data import ConcatDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar100Dataloader(BaseDataLoader):
    """
    CIFAR100 data loading using BaseDataloder
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class TinyImageNetDataloader(BaseDataLoader):
    """
    Stanford Tiny Imagenet data loading using BaseDataloder
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = tfs.Compose([
            tfs.RandomResizedCrop(224),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImageNetDataloader(BaseDataLoader):
    """
    ImageNet data loading using BaseDataloader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = tfs.Compose([
            tfs.RandomResizedCrop(224),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CityscapesDataloader(BaseDataLoader):
    """
    CityScape data loading using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, split='train',
                 transform=None, target_transform=None, transforms=None, mode='fine', target_type='semantic',
                 num_samples=None, return_image_name=False):
        self.data_dir = data_dir
        if split == 'train_val':
            train_dataset = self.dataset = Cityscapes(root=self.data_dir, transform=transform, transforms=transforms,
                                                           target_transform=target_transform, split='train', mode=mode,
                                                           target_type=target_type, num_samples=num_samples,
                                                           return_image_name=return_image_name)
            val_dataset = self.dataset = Cityscapes(root=self.data_dir, transform=transform, transforms=transforms,
                                                         target_transform=target_transform, split='val', mode=mode,
                                                         target_type=target_type, num_samples=num_samples,
                                                         return_image_name=return_image_name)
            self.dataset = ConcatDataset([train_dataset, val_dataset])
        else:
            self.dataset = Cityscapes(root=self.data_dir, transform=transform, transforms=transforms,
                                      target_transform=target_transform, split=split, mode=mode,
                                      target_type=target_type, num_samples=num_samples,
                                      return_image_name=return_image_name)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CityscapesUniformDataloader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, split='train',
                 transform=None, target_transform=None, transforms=None, mode='fine', target_type='semantic',
                 class_uniform_pct=0.5, class_uniform_tile = 1024, num_samples=None, return_image_name=False):
        self.data_dir = data_dir
        if split == 'train_val':
            pass
        else:
            self.dataset = CityScapesUniform(root=self.data_dir, quality=mode, mode=split,
                                             joint_transform_list=transforms, transform=transform,
                                             target_transform=target_transform, class_uniform_pct=class_uniform_pct,
                                             class_uniform_tile=class_uniform_tile, num_samples=num_samples,
                                             return_image_name=return_image_name)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

