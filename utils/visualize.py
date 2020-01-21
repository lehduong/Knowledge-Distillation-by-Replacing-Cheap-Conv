import numpy as np
from ..data_loader.cityscapes import Cityscapes as Dataset


cityscapes_cls2color = {}
classes = Dataset.classes
for c in classes:
    cityscapes_cls2color[c.train_id] = c.color


def apply_mask(image, mask, cls2color=cityscapes_cls2color, alpha=0.5):
    """
    Apply the mask color to given image
    :param image: NUMPY array of shape (HxWx3). The image must be normalized to [0,1] range
    :param mask: NUMPY array of shape (HxW) denotes the class of each pixel in integer
    :param cls2color: dict type. The dictionary contains the mapping from class to its corresponding color
    :param alpha: float. Must be in [0,1] range. Alpha represents the weight between image and mask color
    :return NUMPY array of shape (HxWx3) represent the image which is applied mask colors.
    """
    masks = []
    for c in range(3):
        mask_copy = mask.copy()
        for k, v in cls2color.items():
            mask_copy[mask == k] = v[c]
        mask_copy = np.expand_dims(mask_copy, 2)
        masks.append(mask_copy)
    mask = np.concatenate(masks, axis=-1)
    ret = image*(1-alpha)+alpha*mask/255.0
    return ret

