from torchvision import transforms
from PIL import Image
from math import ceil
import numpy as np
import cv2
import torch


def reverse_mapping(mapping, results, ori_size):
    idx = 0
    outputs = []
    for items in mapping:
        w, h = items[0], items[1]
        coordinates = items[2]
        n_slices = len(coordinates)
        probs_no_flip = collect_windows_result(w, h, coordinates, results[idx: idx + n_slices])
        probs_flipped = collect_windows_result(w, h, coordinates, results[idx + n_slices: idx + 2 * n_slices])
        list_slices_restore = [np.expand_dims(np.fliplr(x), 0) for x in probs_flipped]
        probs_flipped_restored = np.concatenate(list_slices_restore, axis=0)
        probs_no_flip_rs = resize_output(probs_no_flip, ori_size)
        probs_flipped_rs = resize_output(probs_flipped_restored, ori_size)
        probs_mean = (probs_no_flip_rs + probs_flipped_rs) / 2
        outputs.append(np.expand_dims(probs_mean, axis=0))
        idx += 2 * n_slices
    return np.concatenate(outputs, axis=0)


def resize_output(masks, ori_size):
    mask_rs = []
    for x in masks:
        img_rs = cv2.resize(x, ori_size, interpolation=cv2.INTER_LINEAR)
        mask_rs.append(np.expand_dims(img_rs, axis=0))
    result = np.concatenate(mask_rs, axis=0)
    return result


def collect_windows_result(w, h, coordinates, windows):
    num_classes = windows.shape[1]
    full_probs = np.zeros((num_classes, h, w))
    count_predictions = np.zeros((num_classes, h, w))
    for i, coor in enumerate(coordinates):
        x1, y1, x2, y2 = coor
        count_predictions[y1:y2, x1:x2] += 1
        average = windows[i]
        if full_probs[:, y1: y2, x1: x2].shape != average.shape:
            average = average[:, :y2 - y1, :x2 - x1]
        full_probs[:, y1:y2, x1:x2] += average
    full_probs = full_probs / count_predictions.astype(np.float)
    return full_probs


def scale_and_flip_image(image, mean_std, scales=[1.0]):
    w, h = image.size
    new_images = []
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*mean_std)])
    for scale in scales:
        tg_w, tg_h = int(w * scale), int(h * scale)
        scaled_image = image.resize((tg_w, tg_h), Image.BILINEAR)
        flipped_image = scaled_image.transpose(Image.FLIP_LEFT_RIGHT)
        scaled_image = img_transform(scaled_image)
        flipped_image = img_transform(flipped_image)
        new_images.append([scaled_image, flipped_image])

    return ((w, h), new_images)


def get_crops_image(image_data, scales=[1.0], crop_size=512, overlap=1 / 3):
    # image_data[0] is size of original image
    new_images = image_data[1]
    result = []
    # result = [
    # tensor1(row*col*2, 3, crop_size, crop_size) for scale 1, 2 in "row*col*2" for no-flip and flip
    # ]
    mapping = []
    # mapping =[
    # [w1, h1, [(x1, x2, y1, y2), (x1, x2, y1, y2), ...]], for scale 1
    # [w2, h2, [(x1, x2, y1, y2), (x1, x2, y1, y2), ...]], for scale 2
    # ...
    # ]

    for i, scale in enumerate(scales):
        scaled_image, flipped_image = new_images[i]
        h, w = scaled_image.shape[1:]
        tile_size = (int(scale * crop_size), int(scale * crop_size))
        stride = ceil(tile_size[0] * (1 - overlap))
        tile_rows = int(ceil((w - tile_size[0]) / stride) + 1)
        tile_cols = int(ceil((h - tile_size[1]) / stride) + 1)
        windows_image = []
        windows_flipped_image = []
        coordinates = [w, h, []]
        for row in range(tile_rows):
            for col in range(tile_cols):
                y1 = int(col * stride)
                x1 = int(row * stride)
                x2 = min(x1 + tile_size[1], w)
                y2 = min(y1 + tile_size[0], h)
                x1 = int(x2 - tile_size[1])
                y1 = int(y2 - tile_size[0])
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0

                coordinates[2].append((x1, y1, x2, y2))
                img_ts = scaled_image[:, y1:y2, x1:x2].unsqueeze(0)
                fl_img_ts = flipped_image[:, y1:y2, x1:x2].unsqueeze(0)
                windows_image.append(img_ts)
                windows_flipped_image.append(fl_img_ts)

        windows_image = torch.cat(windows_image, dim=0)
        windows_flipped_image = torch.cat(windows_flipped_image, dim=0)
        result.append(torch.cat([windows_image, windows_flipped_image], dim=0))
        mapping.append(coordinates)

    tensor_result = torch.cat(result, dim=0)
    return image_data[0], mapping, tensor_result
