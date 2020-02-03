import torch
import torch.nn.functional as F
import numpy as np


def iou(outputs, labels, ignore_index=255):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # code is borrowed from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    num_classes = outputs.size()[1]
    with torch.no_grad():
        labels[labels==ignore_index] = num_classes
        outputs = torch.argmax(outputs, dim=1)
        conf = confusion_for_batch(outputs.view(-1), labels.view(-1), num_classes+1)
        iou_pc = iou_per_class(conf, num_classes)

    return np.nanmean(iou_pc, 0)  # Or thresholded.mean() if you are interested in average across the batch


def mask2onehot(masks, num_classes=19, ignore_class=255):
    """
    :param masks: Torch.Tensor of shape (Batchsize x H x W)
    :param num_classes: int - number of classes that would be evaluated
    :param ignore_class: int - classes that would be ignored during eval phase
    :return: onehot represent of masks
    """
    pass


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def confusion_for_batch(output, target, num_classes):
    np_op = output.cpu().numpy()
    np_tg = target.cpu().numpy()
    x = np_op + num_classes * np_tg
    bincount_2d = np.bincount( x.astype(np.int32), minlength=num_classes**2)
    conf = np.reshape(bincount_2d, (num_classes, num_classes))
    return conf

def iou_per_class(conf_matrix, ignore_class, SMOOTH = 1e-6):
    conf_matrix[:, ignore_class] = 0
    conf_matrix[ignore_class, :] = 0
    tp = np.diag(conf_matrix)
    iou_pc = (tp + SMOOTH) / (SMOOTH + np.sum(conf_matrix, 0) + np.sum(conf_matrix, 1) - tp)
    return iou_pc