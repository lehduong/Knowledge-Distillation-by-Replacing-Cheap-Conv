import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


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
        self.conf = np.zeros((self.num_classes+1, self.num_classes+1)) # 19class + 1 ignore class
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.num_classes+1, self.num_classes+1))

    def update(self, outputs, labels):
        labels[labels == self.ignore_index] = self.num_classes
        outputs = torch.argmax(outputs, dim=1)
        conf = self.confusion_for_batch(outputs.view(-1), labels.view(-1))
        self.conf = self.conf + conf

    def get_iou(self, smooth=1e-6):
        tp = np.diag(self.conf)
        iou_pc = (tp + smooth) / (smooth + np.sum(self.conf, 0) + np.sum(self.conf, 1) - tp)
        return np.nanmean(iou_pc[:self.num_classes], 0)

    def confusion_for_batch(self, output, target):
        num_classes = self.num_classes + 1  # ignore label +1
        np_op = output.detach().cpu().numpy()
        np_tg = target.detach().cpu().numpy()
        x = np_op + num_classes * np_tg
        bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes ** 2)
        conf = np.reshape(bincount_2d, (num_classes, num_classes))
        return conf
