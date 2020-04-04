
# SPP-net

参考：[[空间金字塔池化]SPP-net](https://blog.zhujian.life/posts/caf02cb8.html)

>文章[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)提出空间金字塔池化（`spatial pyramid pooling`）的概念，避免了固定大小的图像输入，能够有效提高子窗口的识别精度；同时通过共用特征图的方式，极大的提高了检测速度

本仓库利用`SPP-net`实现图像分类和目标检测任务

1. 对于图像分类，在`AlexNet`模型基础上添加了空间金字塔池化层
2. 对于目标检测，测试了检测速度