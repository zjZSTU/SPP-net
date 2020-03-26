# SPP-net

[![Documentation Status](https://readthedocs.org/projects/spp-net/badge/?version=latest)](https://spp-net.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

> `SPP-net`算法实现

学习论文[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)，实现`SPP-net`算法

## 内容列表

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

论文[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)创新性的提出空间金字塔池化（`Spatial Pyramid Pooling`）的概念，解决了输入层固定大小的限制。同时，该解决方案能够适用于所有的`CNN`架构，进一步提高检测和分类的能力

## 安装

### 本地编译文档

需要预先安装以下工具：

```
$ pip install mkdocs
```

## 用法

### 文档浏览

有两种使用方式

1. 在线浏览文档：[SPP-net](https://spp-net.readthedocs.io/zh_CN/latest/)

2. 本地浏览文档，实现如下：

    ```
    $ git clone https://github.com/zjZSTU/SPP-net.git
    $ cd SPP-net
    $ mkdocs serve
    ```
    启动本地服务器后即可登录浏览器`localhost:8000`

## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 致谢

### 引用

```
@article{He_2014,
   title={Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition},
   ISBN={9783319105789},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-319-10578-9_23},
   DOI={10.1007/978-3-319-10578-9_23},
   journal={Lecture Notes in Computer Science},
   publisher={Springer International Publishing},
   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
   year={2014},
   pages={346–361}
}
```

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/SPP-net/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjZSTU
