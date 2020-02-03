import torch
import torch.nn.functional as F

SMOOTH = 1e-6
num_classes = 19
ignore_label = 255


def iou(outputs, labels, ignore_index=255):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # code is borrowed from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    with torch.no_grad():
        outputs = torch.argmax(outputs, dim=1)

        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


def mask2onehot(masks, num_classes=19, ignore_class=255):
    """
    :param masks: Torch.Tensor of shape (Batchsize x H x W)
    :param num_classes: int - number of classes that would be evaluated
    :param ignore_class: int - classes that would be ignored during eval phase
    :return: onehot represent of masks
    """


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

def get_masks_from_label(labels):
    labels[labels==ignore_label] = num_classes
    lbl_size = labels.size()
    mk_size = (lbl_size[0], num_classes + 1, lbl_size[1], lbl_size[2])
    labels.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.IntTensor(*mk_size).zero_()
    one_hot.scatter_(1, labels.cuda(), 1) 
    return one_hot

def result_to_mask(outputs):
    outputs = F.softmax(outputs, dim=1)
    idxs = torch.argmax(outputs, dim=1)
    one_hot = get_masks_from_label(idxs)
    return one_hot
