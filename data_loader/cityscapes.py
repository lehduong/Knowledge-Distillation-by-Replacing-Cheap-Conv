import json
import os
from collections import namedtuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from .datasets import VisionDataset
from torch.utils import data
from PIL import Image
from . import uniform
import numpy as np

CV_SPLITS = 3
CLASS_UNIFORM_PCT = 0.0

class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``gtFine`` or ``gtCoarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    ignore_label = 255
    
    def __init__(self, root, split='train', mode='fine', target_type='semantic',
                 transform=None, target_transform=None, transforms=None, num_samples=None,
                 return_image_name=False):
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        self.rt_img_name = return_image_name
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

        # Limit numbers of Dataset
        if num_samples is not None and num_samples < len(self.images):
            idx = np.random.choice(len(self.images),num_samples)
            self.images = [self.images[i] for i in idx]
            self.targets = [self.targets[i] for i in idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        # relabel target
        target = np.array(target)
        target_copy = target.copy()
        for k, v in self.id_to_trainid.items():
            target_copy[target == k] = v
        target = Image.fromarray(target_copy.astype(np.uint8))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.rt_img_name:
            img_file = os.path.basename(self.images[index])
            img_name = os.path.splitext(img_file)[0]
            return img_name, image, target

        return image, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


def make_dataset(root, mode, maxSkip=0, cv_split=0):
    """
    Assemble list of images + mask files
    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na
    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []
    aug_items = []

    assert mode in ['train', 'val', 'test', 'trainval']
    img_path = os.path.join(root, 'leftImg8bit')
    mask_path = os.path.join(root, 'gtFine')
    mask_postfix = '_gtFine_labelIds.png'
    cv_splits = make_cv_splits(root)
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        if mode == 'test':
            cv_splits = make_test_split(root)
            add_items(items, cv_splits, img_path, mask_path,
                      mask_postfix, mode, maxSkip)
        else:
            add_items(items, cv_splits[cv_split][mode], img_path, mask_path,
                      mask_postfix, mode, maxSkip)

    return items, aug_items


def make_test_split(root):
    test_path = os.path.join(root, 'leftImg8bit', 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities


def make_cv_splits(root):
    """
    Create splits of train/val data.
    A split is a lists of cities.
    split0 is aligned with the default Cityscapes train/val.
    """
    trn_path = os.path.join(root, 'leftImg8bit', 'train')
    val_path = os.path.join(root, 'leftImg8bit', 'val')

    trn_cities = ['train/' + c for c in os.listdir(trn_path)]
    val_cities = ['val/' + c for c in os.listdir(val_path)]

    # want reproducible randomly shuffled
    trn_cities = sorted(trn_cities)

    all_cities = val_cities + trn_cities
    num_val_cities = len(val_cities)
    num_cities = len(all_cities)

    cv_splits = []
    for split_idx in range(CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        offset = split_idx * num_cities // CV_SPLITS
        for j in range(num_cities):
            if j >= offset and j < (offset + num_val_cities):
                split['val'].append(all_cities[j])
            else:
                split['train'].append(all_cities[j])
        cv_splits.append(split)

    return cv_splits


def add_items(items, cities, img_path, mask_path, mask_postfix):
    """
    Add More items ot the list from the augmented dataset
    """

    for c in cities:
        c_items = [name.split('_leftImg8bit.png')[0] for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                    os.path.join(mask_path, c, it + mask_postfix))
            items.append(item)


class CityScapesUniform(data.Dataset):
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    ignore_label = 255
    num_classes = 19
    id_to_trainid = {elem.id: elem.train_id for elem in classes}

    def __init__(self, root, quality, mode, maxSkip=0, joint_transform_list=None, sliding_crop=None,
                 transform=None, target_transform=None, class_uniform_pct=0.5,
                 class_uniform_tile=1024, coarse_boost_classes=None):
        self.root = root
        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_tile = class_uniform_tile
        self.coarse_boost_classes = coarse_boost_classes
        self.cv_split = 0

        self.imgs, self.aug_imgs = make_dataset(self.root, mode, self.maxSkip, cv_split=self.cv_split)
        assert len(self.imgs), 'Found 0 images, please check the data set'

        # Centroids for fine data
        json_fn = 'cityscapes_{}_cv{}_tile{}.json'.format(
            self.mode, self.cv_split, self.class_uniform_tile)
        if os.path.isfile(json_fn):
            with open(json_fn, 'r') as json_data:
                centroids = json.load(json_data)
            self.centroids = {int(idx): centroids[idx] for idx in centroids}
        else:
            self.centroids = uniform.class_centroids_all(
                self.imgs,
                self.num_classes,
                id2trainid=self.id_to_trainid,
                tile_size=class_uniform_tile)
            with open(json_fn, 'w') as outfile:
                json.dump(self.centroids, outfile, indent=4)

        self.fine_centroids = self.centroids.copy()
        # Centroids for augmented data
        if self.maxSkip > 0:
            json_fn = 'cityscapes_{}_cv{}_tile{}_skip{}.json'.format(
                self.mode, self.cv_split, self.class_uniform_tile, self.maxSkip)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.aug_centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                self.aug_centroids = uniform.class_centroids_all(
                    self.aug_imgs,
                    self.num_classes,
                    id2trainid=self.id_to_trainid,
                    tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.aug_centroids, outfile, indent=4)

            # add centroids for augmented data
            # TODO: later, we can also pick classes for augmented data
            for class_id in range(self.num_classes):
                self.centroids[class_id].extend(self.aug_centroids[class_id])

        # Add in coarse centroids for certain classes
        if self.coarse_boost_classes is not None:
            json_fn = 'cityscapes_coarse_{}_tile{}.json'.format(
                self.mode, self.class_uniform_tile)
            if os.path.isfile(json_fn):
                with open(json_fn, 'r') as json_data:
                    centroids = json.load(json_data)
                self.coarse_centroids = {int(idx): centroids[idx] for idx in centroids}
            else:
                self.coarse_imgs, _ = make_dataset('coarse', mode, cv_split=0)
                self.coarse_centroids = uniform.class_centroids_all(
                    self.coarse_imgs,
                    self.num_classes,
                    id2trainid=self.id_to_trainid,
                    tile_size=class_uniform_tile)
                with open(json_fn, 'w') as outfile:
                    json.dump(self.coarse_centroids, outfile, indent=4)

            # add centroids for boost classes
            for class_id in self.coarse_boost_classes:
                self.centroids[class_id].extend(self.coarse_centroids[class_id])

        self.build_epoch()

    def cities_uniform(self, imgs, name):
        """ list out cities in imgs_uniform """
        cities = {}
        for item in imgs:
            img_fn = item[0]
            img_fn = os.path.basename(img_fn)
            city = img_fn.split('_')[0]
            cities[city] = 1
        city_names = cities.keys()

    def build_epoch(self, cut=False):
        """
        Perform Uniform Sampling per epoch to create a new list for training such that it
        uniformly samples all classes
        """
        if self.class_uniform_pct > 0:
            if cut:
                # after max_cu_epoch, we only fine images to fine tune
                self.imgs_uniform = uniform.build_epoch(self.imgs,
                                                        self.fine_centroids,
                                                        self.num_classes,
                                                        CLASS_UNIFORM_PCT)
            else:
                self.imgs_uniform = uniform.build_epoch(self.imgs + self.aug_imgs,
                                                        self.centroids,
                                                        self.num_classes,
                                                        CLASS_UNIFORM_PCT)
        else:
            self.imgs_uniform = self.imgs

    def __getitem__(self, index):
        elem = self.imgs_uniform[index]
        centroid = None
        if len(elem) == 4:
            img_path, mask_path, centroid, class_id = elem
        else:
            img_path, mask_path = elem
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                if idx == 0 and centroid is not None:
                    # HACK
                    # We assume that the first transform is capable of taking
                    # in a centroid
                    img, mask = xform(img, mask, centroid)
                else:
                    img, mask = xform(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, img_name

    def __len__(self):
        return len(self.imgs_uniform)

if __name__ == '__main__':
    dataset = CityScapesUniform('../data', 'fine', 'train')