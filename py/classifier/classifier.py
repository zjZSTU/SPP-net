# -*- coding: utf-8 -*-

"""
@date: 2020/3/26 下午2:33
@file: classifier.py
@author: zj
@description: 
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision
from torch.hub import load_state_dict_from_url

import models.alexnet as alexnet
import utils.util as util

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def load_data(root_dir):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    dataset_sizes = {}
    for phase in ['train', 'val']:
        phase_dir = os.path.join(root_dir, phase)

        data_set = ImageFolder(phase_dir, transform=transform)
        data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)

        data_loaders[phase] = data_loader
        dataset_sizes[phase] = len(data_set)

    return data_loaders, dataset_sizes


def train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, num_epochs=25, device=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    data_loaders, data_sizes = load_data('../data/train_val/')

    model = alexnet.AlexNet(num_classes=20)
    # 切换到GPU
    device = util.get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
c
    best_model = train_model(model, criterion, optimizer, lr_scheduler, data_sizes, data_loaders, num_epochs=25,
                             device=device)
    # 保存最好的模型参数
    util.check_dir('./data/models')
    torch.save(best_model.state_dict(), './data/models/alexnet_car.pth')
