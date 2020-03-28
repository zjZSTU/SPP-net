# -*- coding: utf-8 -*-

"""
@date: 2020/3/28 下午4:24
@file: spatial_pyramid_pooling.py
@author: zj
@description: 
"""

from math import sqrt
import torch
import torch.nn as nn


class SpatialPyramidPooling(nn.Module):
    """Generate fixed length representation regardless of image dimensions
    Based on the paper "Spatial Pyramid Pooling in Deep Convolutional Networks
    for Visual Recognition" (https://arxiv.org/pdf/1406.4729.pdf)
    :param [int] num_pools: Number of pools to split each input feature map into.
        Each element must be a perfect square in order to equally divide the
        pools across the feature map. Default corresponds to the original
        paper's implementation
    :param str mode: Specifies the type of pooling, either max or avg
    """

    # 默认使用3级金字塔，其向量长度为21
    def __init__(self, num_pools=(1, 4, 16), mode='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.name = 'SpatialPyramidPooling'
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool2d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool2d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools = []
        for p in num_pools:
            side_length = sqrt(p)
            if not side_length.is_integer():
                raise ValueError(f'Bin size {p} is not a perfect square')
            self.pools.append(pool_func(int(side_length)))

    def forward(self, feature_maps):
        """Pool feature maps at different bin levels and concatenate
        :param torch.tensor feature_maps: Arbitrarily shaped spatial and
            channel dimensions extracted from any generic convolutional
            architecture. Shape ``(N, C, H, W)``
        :return torch.tensor pooled: Concatenation of all pools with shape
            ``(N, C, sum(num_pools))``
        """
        assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        batch_size = feature_maps.size(0)
        channels = feature_maps.size(1)
        pooled = []
        for p in self.pools:
            pooled.append(p(feature_maps).view(batch_size, channels, -1))
        return torch.cat(pooled, dim=2)
