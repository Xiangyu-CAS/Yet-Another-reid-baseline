import torch

import torch
from torch import nn
from .resnet50_ibn_a import resnet50_ibn_a
from lib.utils import load_checkpoint


class Encoder(nn.Module):
    def __init__(self, model_path, pretrain_choice):
        super(Encoder, self).__init__()
        self.base = resnet50_ibn_a()
        self.in_planes = 2048

        if pretrain_choice == 'imagenet':
            load_checkpoint(self.base, model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)

    def forward(self, x):
        featmap = self.base(x)  # (b, 2048, 1, 1)
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)
        feat = self.bottleneck(global_feat)
        return feat


class Head(nn.Module):
    def __init__(self, encoder, num_class):
        super(Head, self).__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(self.encoder.in_planes, num_class, bias=False)

    def forward(self, x):
        feat = self.encoder(x)
        score = self.fc(feat)
        return score, feat