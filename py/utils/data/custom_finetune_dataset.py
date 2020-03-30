# -*- coding: utf-8 -*-

"""
@date: 2020/3/29 下午9:05
@file: custom_finetune_dataset.py
@author: zj
@description: 
"""

import cv2
import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import models.alexnet_spp as alexnet_spp
import utils.util as util


class CustomFinetuneDataset(Dataset):

    def __init__(self, root_dir, transform, model, s):
        """
        加载所有的图像以及正负样本边界框
        """
        samples = util.parse_car_csv(root_dir)

        jpeg_images = list()
        positive_list = list()
        negative_list = list()
        for idx in range(len(samples)):
            sample_name = samples[idx]
            jpeg_images.append(cv2.imread(os.path.join(root_dir, 'JPEGImages', sample_name + ".jpg")))

            positive_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_1.csv')
            positive_annotations = np.loadtxt(positive_annotation_path, dtype=np.int, delimiter=' ')
            # 考虑csv文件为空或者仅包含单个标注框
            if len(positive_annotations.shape) == 1:
                # 单个标注框坐标
                if positive_annotations.shape[0] == 4:
                    positive_dict = dict()

                    positive_dict['rect'] = positive_annotations
                    positive_dict['image_id'] = idx
                    # positive_dict['image_name'] = sample_name

                    positive_list.append(positive_dict)
            else:
                for positive_annotation in positive_annotations:
                    positive_dict = dict()

                    positive_dict['rect'] = positive_annotation
                    positive_dict['image_id'] = idx
                    # positive_dict['image_name'] = sample_name

                    positive_list.append(positive_dict)

            negative_annotation_path = os.path.join(root_dir, 'Annotations', sample_name + '_0.csv')
            negative_annotations = np.loadtxt(negative_annotation_path, dtype=np.int, delimiter=' ')
            # 考虑csv文件为空或者仅包含单个标注框
            if len(negative_annotations.shape) == 1:
                # 单个标注框坐标
                if negative_annotations.shape[0] == 4:
                    negative_dict = dict()

                    negative_dict['rect'] = negative_annotations
                    negative_dict['image_id'] = idx
                    # negative_dict['image_name'] = sample_name

                    negative_list.append(negative_dict)
            else:
                for negative_annotation in negative_annotations:
                    negative_dict = dict()

                    negative_dict['rect'] = negative_annotation
                    negative_dict['image_id'] = idx
                    # negative_dict['image_name'] = sample_name

                    negative_list.append(negative_dict)

        self.jpeg_images = jpeg_images
        self.positive_list = positive_list
        self.negative_list = negative_list

        # 保存所有图片的特征图有可能会导致内存不足
        # 提取所有图片的特征图
        feature_map_list = list()
        for img in jpeg_images:
            # 图像处理
            img = transform(img)
            # 添加一维
            img = img.unsqueeze(0)
            # 获取特征图
            feature_map = model.feature_map(img)
            # 降维
            # feature_map = feature_map.squeeze(0)
            feature_map_list.append(feature_map)

        self.feature_map_list = feature_map_list
        self.model = model
        self.transform = transform
        self.s = s

    def __getitem__(self, index: int):
        # 定位下标所属图像
        if index < len(self.positive_list):
            # 正样本
            target = 1
            positive_dict = self.positive_list[index]

            xmin, ymin, xmax, ymax = positive_dict['rect']
            image_id = positive_dict['image_id']

            cache_dict = positive_dict
        else:
            # 负样本
            target = 0
            idx = index - len(self.positive_list)
            negative_dict = self.negative_list[idx]

            xmin, ymin, xmax, ymax = negative_dict['rect']
            image_id = negative_dict['image_id']

            cache_dict = negative_dict

        src_image = self.jpeg_images[image_id]
        h, w = src_image.shape[:2]
        ratio = self.scale(h, w, self.s)
        S = self.model.get_S()

        xmin = int(np.floor(xmin * ratio / S))
        ymin = int(np.floor(ymin * ratio / S))
        xmax = int(np.ceil(xmax * ratio / S))
        ymax = int(np.ceil(ymax * ratio / S))

        feature_map = self.feature_map_list[image_id]
        h, w = feature_map.shape[2:4]

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > w:
            xmax = w
        if ymax > h:
            ymax = h

        image = feature_map[:, :, ymin:ymax, xmin:xmax]
        image = self.model.spp(image).squeeze(0)

        return image, target, cache_dict

    def __len__(self) -> int:
        return len(self.positive_list) + len(self.negative_list)

    def get_positive_num(self):
        return len(self.positive_list)

    def get_negative_num(self):
        return len(self.negative_list)

    def scale(self, h, w, s):
        """
        计算缩放比例
        :param h: 原图长
        :param w: 原图宽
        :param s: 缩放后最短边
        """
        if h > w:
            return 1.0 * s / h
        else:
            return 1.0 * s / w


if __name__ == '__main__':
    root_dir = '../../data/finetune_car/train'
    s = 688
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(s),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    model = alexnet_spp.AlexNet_SPP(num_classes=20)

    data_set = CustomFinetuneDataset(root_dir, transform, model, s)

    image, target, cache_dict = data_set.__getitem__(0)
    print(image.shape)
    print(target)
    print(cache_dict)
