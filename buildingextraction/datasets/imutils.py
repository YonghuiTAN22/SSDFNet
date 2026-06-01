import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision

from PIL import ImageEnhance


# def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
def normalize_img(img, mean, std):
    """
    Normalize image by subtracting mean and dividing by std.
    Singapore OPT:
    train: mean=[ 71.238, 74.439, 69.759], std=[39.958, 36.441, 37.036]
    val:   mean=[100.075, 99.008, 93.626], std=[49.446, 45.590, 45.320]
    test:  mean=[ 91.971, 95.132, 89.715], std=[45.057, 41.980, 41.703]
    """
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


def random_fliplr_be(img_opt, img_sar, label, edge):
    if random.random() > 0.5:
        edge = np.fliplr(edge)
        label = np.fliplr(label)
        img_opt = np.fliplr(img_opt)
        img_sar = np.fliplr(img_sar)

    return img_opt, img_sar, label, edge


def random_flipud_be(img_opt, img_sar, label, edge):
    if random.random() > 0.5:
        label = np.flipud(label)
        edge = np.flipud(edge)

        img_opt = np.flipud(img_opt)
        img_sar = np.flipud(img_sar)

    return img_opt, img_sar, label, edge


def random_rot_be(img_opt, img_sar, label, edge):
    k = random.randrange(3) + 1

    img_opt = np.rot90(img_opt, k).copy()
    img_sar = np.rot90(img_sar, k).copy()
    label = np.rot90(label, k).copy()
    edge = np.rot90(edge, k).copy()

    return img_opt, img_sar, label, edge


def random_crop_be(input_img_opt, loc_label, edge_label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_input_image_opt = np.zeros((H, W, 3), dtype=np.float32)
    # pad_input_image_sar = np.zeros((H, W, 3), dtype=np.float32)

    pad_loc_label = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_edge_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_input_image_opt[:, :, 0] = mean_rgb[0]
    pad_input_image_opt[:, :, 1] = mean_rgb[1]
    pad_input_image_opt[:, :, 2] = mean_rgb[2]

    # pad_input_image_sar[:, :, 0] = mean_rgb[0]
    # pad_input_image_sar[:, :, 1] = mean_rgb[1]
    # pad_input_image_sar[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_input_image_opt[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = input_img_opt
    # pad_input_image_sar[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = input_img_sar
    pad_loc_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = loc_label
    pad_edge_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = edge_label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_loc_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    img_opt = pad_input_image_opt[H_start:H_end, W_start:W_end, :]
    # img_sar = pad_input_image_sar[H_start:H_end, W_start:W_end, :]
    loc_label = pad_loc_label[H_start:H_end, W_start:W_end]
    edge_label = pad_edge_label[H_start:H_end, W_start:W_end]

    return img_opt, loc_label, edge_label
