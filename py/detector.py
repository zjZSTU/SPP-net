# -*- coding: utf-8 -*-

"""
@date: 2020/4/4 下午9:54
@file: detector.py
@author: zj
@description: 
"""

import copy
import cv2
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import selectivesearch

import utils.util as util
import models.alexnet_spp as alexnet_spp


def get_transform(s=688):
    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(s),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_model(device=None):
    # 加载CNN模型
    model = alexnet_spp.AlexNet_SPP(num_classes=2)
    # 加载预训练模型
    # model.load_state_dict(torch.load('../data/models/alexnet_spp_car.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model


def compute_project_window(src_rect, S=16):
    xmin, ymin, xmax, ymax = src_rect

    xmin = int(np.floor(xmin / S) + 1)
    ymin = int(np.floor(ymin / S) + 1)
    xmax = int(np.ceil(xmax / S) - 1)
    ymax = int(np.ceil(ymax / S) - 1)

    if xmin >= xmax:
        xmax = xmin + 1
    if ymin >= ymax:
        ymax = ymin + 1

    return (xmin, ymin, xmax, ymax)


def nms(rect_list, score_list):
    """
    非最大抑制
    :param rect_list: list，大小为[N, 4]
    :param score_list： list，大小为[N]
    """
    nms_rects = list()
    nms_scores = list()

    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    # 一次排序后即可
    # 按分类概率从大到小排序
    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]

    thresh = 0.3
    while len(score_array) > 0:
        # 添加分类概率最大的边界框
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]

        length = len(score_array)
        if length <= 0:
            break

        # 计算IoU
        iou_scores = util.iou(np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # print(iou_scores)
        # 去除重叠率大于等于thresh的边界框
        idxs = np.where(iou_scores < thresh)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]

    return nms_rects, nms_scores


def draw_box_with_text(img, rect_list, score_list):
    """
    绘制边框及其分类概率
    :param img:
    :param rect_list:
    :param score_list:
    :return:
    """
    for i in range(len(rect_list)):
        xmin, ymin, xmax, ymax = rect_list[i]
        score = score_list[i]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == '__main__':
    start = time.time()

    device = util.get_device()
    transform = get_transform()
    model = get_model(device=device)

    # 创建selectivesearch对象
    gs = selectivesearch.get_selective_search()

    # test_img_path = '../imgs/000007.jpg'
    # test_xml_path = '../imgs/000007.xml'
    test_img_path = '../imgs/000012.jpg'
    test_xml_path = '../imgs/000012.xml'

    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    # 计算CNN特征图
    image = transform(img).unsqueeze(0).to(device)
    feature_map = model.feature_map(image)
    feature_map_w, feature_map_h = feature_map.shape[2:4]
    # 标注边界框
    bndboxs = util.parse_xml(test_xml_path)
    # 候选区域建议
    selectivesearch.config(gs, img, strategy='f')
    rects = selectivesearch.get_rects(gs)
    print('候选区域建议数目： %d' % len(rects))

    svm_thresh = 0.60

    # 保存正样本边界框以及
    score_list = list()
    positive_list = list()

    for rect in rects:
        project_rect = compute_project_window(rect, S=16)
        xmin, ymin, xmax, ymax = project_rect
        if xmin >= feature_map_w:
            xmin = feature_map_w - 1
            ymax = feature_map_w
        if ymin >= feature_map_h:
            ymin = feature_map_h - 1
            ymax = feature_map_h
        sub_feature_map = feature_map[:, :, ymin:ymax, xmin:xmax]

        spp_vector = model.spp(sub_feature_map)
        output = model.classify(spp_vector)[0]

        if torch.argmax(output).item() == 1:
            """
            预测为汽车
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)
                # cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
                print(rect, output, probs)

    nms_rects, nms_scores = nms(positive_list, score_list)
    print(nms_rects)
    print(nms_scores)
    draw_box_with_text(dst, nms_rects, nms_scores)

    end = time.time()
    print('detect time: %d s' % (end - start))
    cv2.imshow('img', dst)
    cv2.waitKey(0)
