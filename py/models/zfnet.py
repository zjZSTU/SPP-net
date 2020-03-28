# -*- coding: utf-8 -*-

"""
@date: 2020/3/26 下午4:33
@file: zfnet.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from torchvision.models import alexnet


class ZFNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def zfnet(num_classes=20):
    model = ZFNet(num_classes=num_classes)
    model_alexnet = alexnet(pretrained=True, progress=True)

    pretrained_dict = model_alexnet.state_dict()
    model_dict = model.state_dict()

    res_dict = dict()
    for item in zip(pretrained_dict.items(), model_dict.items()):
        pretrained_dict_item, model_dict_item = item

        k1, v1 = pretrained_dict_item
        k2, v2 = model_dict_item
        # print(v1.shape, v2.shape)

        if k1 == k2 and v1.shape == v2.shape:
            res_dict[k2] = v1
        else:
            res_dict[k2] = v2

    model_dict.update(res_dict)
    model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    # # model = ZFNet(num_classes=20)
    # model = zfnet(num_classes=20)
    # print(model)
    #
    # input = torch.randn((128, 3, 227, 227))
    # output = model.forward(input)
    # print(output.size())

    model = alexnet()
    print(model)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 20)
    print(model)

    # model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print(k, v.shape)
