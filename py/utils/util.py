# -*- coding: utf-8 -*-

"""
@date: 2020/3/25 下午3:06
@file: util.py
@author: zj
@description: 
"""

import os
import numpy as np
import xmltodict
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_car_csv(csv_dir):
    csv_path = os.path.join(csv_dir, 'car.csv')
    samples = np.loadtxt(csv_path, dtype=np.str)
    return samples


def parse_xml(xml_path):
    """
    解析xml文件，返回标注边界框坐标
    """
    # print(xml_path)
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)
        # print(xml_dict)

        bndboxs = list()
        objects = xml_dict['annotation']['object']
        if isinstance(objects, list):
            for obj in objects:
                obj_name = obj['name']
                difficult = int(obj['difficult'])
                if 'car'.__eq__(obj_name) and difficult != 1:
                    bndbox = obj['bndbox']
                    bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        elif isinstance(objects, dict):
            obj_name = objects['name']
            difficult = int(objects['difficult'])
            if 'car'.__eq__(obj_name) and difficult != 1:
                bndbox = objects['bndbox']
                bndboxs.append((int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])))
        else:
            pass

        return np.array(bndboxs)


def iou(pred_box, target_box):
    """
    计算候选建议和标注边界框的IoU
    :param pred_box: 大小为[4]
    :param target_box: 大小为[N, 4]
    :return: [N]
    """
    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def compute_ious(rects, bndboxs):
    iou_list = list()
    for rect in rects:
        scores = iou(rect, bndboxs)
        iou_list.append(max(scores))
    return iou_list


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def save_png(title, res_dict):
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    fig = plt.figure()

    plt.title(title)
    for name, res in res_dict.items():
        for k, v in res.items():
            x = list(range(len(v)))
            plt.plot(v, label='%s-%s' % (name, k))

    plt.legend()
    plt.savefig('%s.png' % title)


def show():
    x = list(range(10))
    y = random.sample(list(range(100)), 10)

    plt.figure(1, figsize=(9, 3))

    plt.title('test')
    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='unset')
    plt.legend()

    plt.subplot(122)

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x, y, label='set')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # res_dict = {'alexnet': {'train': [1, 2, 3], 'val': [2, 3, 5]}, 'zfnet': {'train': [5, 5, 6], 'val': [2, 6, 7]}}
    # save_png('loss', res_dict)

    show()
