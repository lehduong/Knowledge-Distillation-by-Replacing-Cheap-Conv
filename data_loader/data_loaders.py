from torchvision import datasets
from torchvision import transforms as tfs
from base import BaseDataLoader
from .cityscapes import Cityscapes, CityScapesUniform
from torch.utils.data import ConcatDataset


class Cifar100Dataloader(BaseDataLoader):
    """
    CIFAR100 data loading using BaseDataloder
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            trsfm = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            trsfm = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar10Dataloader(BaseDataLoader):
    """
    CIFAR10 data loading using BaseDataloder
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        if training:
            trsfm = tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            trsfm = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
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
            raise ValueError("Only support train split for Uniform Cityscapes")
        else:
            self.dataset = CityScapesUniform(root=self.data_dir, quality=mode, mode=split,
                                             joint_transform_list=transforms, transform=transform,
                                             target_transform=target_transform, class_uniform_pct=class_uniform_pct,
                                             class_uniform_tile=class_uniform_tile, num_samples=num_samples,
                                             return_image_name=return_image_name)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

