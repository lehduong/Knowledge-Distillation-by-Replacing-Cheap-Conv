import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from PIL import Image


def stat_cuda(msg):
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """ wrapper function for endless data loader. """
    for loader in repeat(data_loader):
        yield from loader

def save_image(np_arr, file_path):
    img = Image.fromarray(np_arr)
    img.save(file_path)

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class CityscapesMetricTracker:
    class_names = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sight",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle"
    ]
    num_classes = len(class_names)

    def __init__(self, writer=None, ignore_index=255):
        self.writer = writer
        class_iou = list(map(lambda x: "class_iou_"+x, self.class_names))
        self._data = pd.DataFrame(index=class_iou, columns=['total', 'counts', 'average'])
        self.ignore_index = ignore_index
        self.conf = np.zeros((self.num_classes, self.num_classes))  # 19class + 1 ignore class
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.num_classes, self.num_classes))

    def update(self, outputs, labels):
        labels[labels == self.ignore_index] = self.num_classes
        outputs = torch.argmax(outputs, dim=1)
        conf = self.confusion_for_batch(outputs.view(-1), labels.view(-1))
        self.conf = self.conf + conf

    def get_iou(self):
        if not np.any(self.conf):
            return 1.
        tp = np.diag(self.conf)
        iou_pc = tp / (np.sum(self.conf, 0) + np.sum(self.conf, 1) - tp)
        return np.nanmean(iou_pc)

    def confusion_for_batch(self, output, target):
        pred = output.flatten().detach().cpu().numpy()
        target = target.flatten().detach().cpu().numpy()
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) +
            pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist


class EarlyStopTracker:
    def __init__(self, mode='last', criterion='min', threshold=0.0001, threshold_mode='rel'):
        """
        :param mode: str - either 'last' or 'best'
        """
        self.mode = mode
        if self.mode != 'last' and self.mode != 'best':
            raise ValueError('Unsupported type of mode. Expect either "last" or "best" but got: ' + str(self.mode))
        self.criterion = criterion
        if self.criterion != 'min' and self.criterion != 'max':
            raise ValueError('Unsupported type of mode. Expect either "min" or "max" but got: ' + str(self.criterion))
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.last = None
        self.best = None
        self.last_update_success = True

    def is_better(self, old_value, new_value):
        if old_value is None:
            return True
        elif self.criterion == 'min':
            if self.threshold_mode == 'rel':
                threshold_old_value = old_value*(1-self.threshold)
            else:
                threshold_old_value = old_value - self.threshold
            if new_value < threshold_old_value:
                return True
            return False
        else:  # max
            if self.threshold_mode == 'rel':
                threshold_old_value = old_value*(1+self.threshold)
            else:
                threshold_old_value = old_value + self.threshold
            if new_value > threshold_old_value:
                return True
            return False

    def update(self, new_value):
        if self.mode == 'best':
            old_value = self.best
        else:
            old_value = self.last

        if self.is_better(old_value, new_value):
            if self.mode == 'best':
                self.best = new_value
            self.last = new_value
            self.last_update_success = True
            return True
        else:
            self.last = new_value
            self.last_update_success = False
            return False

    def reset(self):
        self.last = None
        self.best = None
        self.last_update_success = True


