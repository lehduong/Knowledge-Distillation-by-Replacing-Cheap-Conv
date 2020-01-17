from .data_loaders import *
from . import joint_transforms
from . import transforms as extended_transforms
from torchvision import transforms as standard_transforms


def _create_transform(config):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    joint_transforms_params = config['transforms']['joint_transforms']
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(joint_transforms_params['crop_size'],
                                           False,
                                           pre_size=joint_transforms_params['pre_size'],
                                           scale_min=joint_transforms_params['crop_size'],
                                           scale_max=joint_transforms_params['scale_max'],
                                           ignore_index=joint_transforms_params['ignore_label']),
        joint_transforms.Resize(joint_transforms_params['crop_size']),
        joint_transforms.RandomHorizontallyFlip()
    ]
    train_joint_transforms = standard_transforms.Compose(train_joint_transform_list)

    # Image appearance transformations
    extended_transforms_params = config['transforms']['extended_transforms']
    train_input_transform = []
    if extended_transforms_params['color_aug'] > 0:
        train_input_transform += [extended_transforms.ColorJitter(
            brightness=extended_transforms_params['color_aug'],
            contrast=extended_transforms_params['color_aug'],
            saturation=extended_transforms_params['color_aug'],
            hue=extended_transforms_params['color_aug'])]

    if extended_transforms_params['blur'] == 'bilateral':
        train_input_transform += [extended_transforms.RandomBilateralBlur()]
    elif extended_transforms_params['blur'] == 'gaussian':
        train_input_transform += [extended_transforms.RandomGaussianBlur()]
    else:
        pass

    train_input_transform += [standard_transforms.ToTensor(),
                              standard_transforms.Normalize(*mean_std)]
    train_input_transform = standard_transforms.Compose(train_input_transform)

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    target_transform = extended_transforms.MaskToTensor()

    return train_joint_transforms, train_input_transform, target_transform, val_input_transform





