import torch

import torch
from torch import nn
from .resnet_ibn_a import resnet50_ibn_a
from .regnet import regnety_800mf
from lib.utils import load_checkpoint

# backbone func and feature channels
backbone_factory = {
    'resnet50_ibn_a': [resnet50_ibn_a, 2048],
    'regnety_800mf': [regnety_800mf, 768],
}


def build_backbone(name, *args, **kwargs):
    if name not in backbone_factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return backbone_factory[name][0](*args, **kwargs), backbone_factory[name][1]


class Encoder(nn.Module):
    def __init__(self, backbone_name, model_path, pretrain_choice):
        super(Encoder, self).__init__()
        self.base, self.in_planes = build_backbone(backbone_name)

        if pretrain_choice == 'imagenet':
            load_checkpoint(self.base, model_path)
            print('Loading pretrained ImageNet model......')

        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = GeM()
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


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else torch.nn.parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)
