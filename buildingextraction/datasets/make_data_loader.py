import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/SSDFNet')

import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tifffile
import rasterio

import cv2

import buildingextraction.datasets.imutils as imutils

from PIL import Image


def rasterio_loader(path):
    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)
        img = np.moveaxis(img, 0, -1)  # (H, W, C)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img

def rasterio_loader_sar(path):
    return rasterio_loader(path)  # SAR 与 OPT 读取方式相同

def img_loader(path):
    # img = np.array(imageio.imread(path), np.float32)
    try:
        img = tifffile.imread(path).astype(np.float32)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32)
    # img = np.array(Image.open(path), dtype=np.float32)
    return img

def img_loader_sar(path):
    img = tifffile.imread(path).astype(np.float32)
    # img = np.array(Image.open(path), dtype=np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot
    

class BuildingDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        # self.data_list = data_list
        self.loader = data_loader
        self.loader_sar = img_loader_sar
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

        # 读取对应split的图片名列表
        txt_path = os.path.join(dataset_path, f'{type}.txt')
        with open(txt_path, 'r') as f:
            self.data_list = [line.strip() for line in f if line.strip()]
        
        print(f"{type} set: {len(self.data_list)} samples")

    def __transforms_with_sar(self, aug, img_opt, img_sar, loc_label, edge_label, type):
        if aug:
            img_opt, img_sar, loc_label, edge_label = imutils.random_crop_be(img_opt, img_sar, loc_label, edge_label, self.crop_size)
            img_opt, img_sar, loc_label, edge_label = imutils.random_fliplr_be(img_opt, img_sar, loc_label, edge_label)
            img_opt, img_sar, loc_label, edge_label = imutils.random_flipud_be(img_opt, img_sar, loc_label, edge_label)
            img_opt, img_sar, loc_label, edge_label = imutils.random_rot_be(img_opt, img_sar, loc_label, edge_label)

        if type == 'train':
            img_opt = imutils.normalize_img(img_opt, mean=[ 71.238, 74.439, 69.759], std=[39.958, 36.441, 37.036])
        elif type == 'val':
            img_opt = imutils.normalize_img(img_opt, mean=[100.075, 99.008, 93.626], std=[49.446, 45.590, 45.320])
        else:  # test
            img_opt = imutils.normalize_img(img_opt, mean=[ 91.971, 95.132, 89.715], std=[45.057, 41.980, 41.703])
        img_opt = np.transpose(img_opt, (2, 0, 1))

        
        if type == 'train':
            img_sar = imutils.normalize_img(img_sar, mean=[ 71.238, 74.439, 69.759], std=[39.958, 36.441, 37.036])
        elif type == 'val':
            img_sar = imutils.normalize_img(img_sar, mean=[100.075, 99.008, 93.626], std=[49.446, 45.590, 45.320])
        else:  # test
            img_sar = imutils.normalize_img(img_sar, mean=[ 91.971, 95.132, 89.715], std=[45.057, 41.980, 41.703])
        # 确保是3通道再转置
        if img_sar.ndim == 3 and img_sar.shape[2] == 3:
            img_sar = np.transpose(img_sar, (2, 0, 1))
        else:
            # 如果是单通道，先增加通道维度
            img_sar = img_sar[np.newaxis, :, :]

        return img_opt, img_sar, loc_label, edge_label
    
    def __transforms(self, aug, img_opt, loc_label, edge_label, type):
        if aug:
            img_opt, loc_label, edge_label = imutils.random_crop_be(img_opt, loc_label, edge_label, self.crop_size)
            img_opt, loc_label, edge_label = imutils.random_fliplr_be(img_opt, loc_label, edge_label)
            img_opt, loc_label, edge_label = imutils.random_flipud_be(img_opt, loc_label, edge_label)
            img_opt, loc_label, edge_label = imutils.random_rot_be(img_opt, loc_label, edge_label)

        if type == 'train':
            img_opt = imutils.normalize_img(img_opt, mean=[ 71.238, 74.439, 69.759], std=[39.958, 36.441, 37.036])
        elif type == 'val':
            img_opt = imutils.normalize_img(img_opt, mean=[100.075, 99.008, 93.626], std=[49.446, 45.590, 45.320])
        else:  # test
            img_opt = imutils.normalize_img(img_opt, mean=[ 91.971, 95.132, 89.715], std=[45.057, 41.980, 41.703])
        img_opt = np.transpose(img_opt, (2, 0, 1))

        return img_opt, loc_label, edge_label


    def __getitem__(self, index):
        sar_img_path = os.path.join(self.dataset_path, self.type, 'sar', self.data_list[index] + '.tif')
        opt_img_path = os.path.join(self.dataset_path, self.type, 'rgb', self.data_list[index] + '.tif')
        loc_label_path = os.path.join(self.dataset_path, self.type, 'masks', self.data_list[index]+ '.tif')
        edge_path = os.path.join(self.dataset_path, self.type, 'boundary', self.data_list[index]+ '.tif')
        img_name = self.data_list[index]

        for path in [sar_img_path, opt_img_path, loc_label_path, edge_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")

        # SAR 用 tifffile
        input_img_sar = tifffile.imread(sar_img_path)
        if input_img_sar.ndim == 2:
            input_img_sar = np.stack((input_img_sar,) * 3, axis=-1)

        # 光学图像仍然用 PIL
        input_img_opt = np.array(Image.open(opt_img_path).convert("RGB"))

        # Labels
        loc_label = np.array(Image.open(loc_label_path))
        edge_label = np.array(Image.open(edge_path))

        if loc_label.ndim == 3 and loc_label.shape[2] == 3:
            loc_label = cv2.cvtColor(loc_label, cv2.COLOR_RGB2GRAY)
        if edge_label.ndim == 3 and edge_label.shape[2] == 3:
            edge_label = cv2.cvtColor(edge_label, cv2.COLOR_RGB2GRAY)

        # resize
        target_size = (512, 512)
        input_img_opt = cv2.resize(input_img_opt, target_size, interpolation=cv2.INTER_LINEAR)
        input_img_sar = cv2.resize(input_img_sar, target_size, interpolation=cv2.INTER_LINEAR)
        loc_label = cv2.resize(loc_label, target_size, interpolation=cv2.INTER_NEAREST)
        edge_label = cv2.resize(edge_label, target_size, interpolation=cv2.INTER_NEAREST)

        # transforms
        input_img_opt, input_img_sar, loc_label, edge_label = self.__transforms_with_sar(
            False, input_img_opt, input_img_sar, loc_label, edge_label, type=self.data_pro_type
        )

        data_idx = self.data_list[index]
        return input_img_opt, input_img_sar, loc_label, edge_label, data_idx

    def __len__(self):
        return len(self.data_list)



def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SN6' in args.dataset or 'DFC' in args.dataset:
        dataset = BuildingDatset(args.train_dataset_path, args.train_data_name_list, args.crop_size, None, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=6,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError
