# -*- coding: utf-8 -*-

"""
@date: 2020/3/28 下午4:27
@file: alexnet_spp.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet

from models.spatial_pyramid_pooling import SpatialPyramidPooling


class AlexNet_SPP(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_SPP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.spp = SpatialPyramidPooling(num_pools=(1, 4, 9, 36), mode='max')
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 只训练分类器
        for p in self.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 50, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.spp(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def feature_map(self, x):
        return self.features(x)

    def classify(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_S(self):
        return 16


def alexnet_spp(num_classes=20):
    model = AlexNet_SPP(num_classes=num_classes)
    model_alexnet = alexnet(pretrained=True, progress=True)

    pretrained_dict = model_alexnet.state_dict()
    model_dict = model.state_dict()

    res_dict = dict()
    for item in zip(pretrained_dict.items(), model_dict.items()):
        pretrained_dict_item, model_dict_item = item

        k1, v1 = pretrained_dict_item
        k2, v2 = model_dict_item
        # print(k1, k2)
        # print(v1.shape, v2.shape)

        if k1 == k2 and v1.shape == v2.shape:
            res_dict[k2] = v1
        else:
            res_dict[k2] = v2

    model_dict.update(res_dict)
    model.load_state_dict(model_dict)

    return model


def test():
    model = AlexNet_SPP(num_classes=20)
    print(model)

    input = torch.randn((128, 3, 227, 227))
    output = model.forward(input)
    print(output.size())


def test2():
    model = alexnet_spp(num_classes=20)
    print(model)


def test3():
    model = AlexNet_SPP(num_classes=20)
    print(model)

    input = torch.randn((128, 3, 180, 180))
    output = model.forward(input)
    print(output.size())


def test4():
    model = AlexNet_SPP(num_classes=21)

    input = torch.randn((1, 3, 668, 668))
    features = model.feature_map(input)
    print(features.shape)

    res = model.classify(features)
    print(res.shape)


if __name__ == '__main__':
    model = alexnet_spp(num_classes=2)

    model_dict = model.state_dict()
    for item in model_dict.items():
        k, v = item
        print(k, v.requires_grad)
        if 'classifier' not in k:
            v.requires_grad = False

    for k, v in model.named_parameters():
        if 'classifier' not in k:
            v.requires_grad = False  # 固定参数

    for param in model.parameters():
        print(param.requires_grad)

    print('# Model parameters:', sum(param.numel() for param in model.parameters()))
