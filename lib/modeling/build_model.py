import torch

import torch
from torch import nn
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .resnest import resnest50, resnest101
from .regnet import regnety_800mf
from lib.utils import load_checkpoint
from lib.losses.id_loss import Cosface, Circle, Arcface, CrossEntropyLabelSmooth, CosineSoftmax

# backbone func and feature channels
backbone_factory = {
    'resnet50_ibn_a': [resnet50_ibn_a, 2048],
    'resnet101_ibn_a': [resnet101_ibn_a, 2048],
    'resnest50': [resnest50, 2048],
    'resnest101': [resnest101, 2048],
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
        self.bottleneck.apply(weights_init_kaiming)

    def reset_bn(self):
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        featmap = self.base(x)  # (b, 2048, 1, 1)
        global_feat = self.gap(featmap)
        global_feat = global_feat.flatten(1)
        feat = self.bottleneck(global_feat)
        return feat

class Head(nn.Module):
    def __init__(self, encoder, num_class, cfg):
        super(Head, self).__init__()
        self.encoder = encoder
        self.id_loss_type = cfg.MODEL.ID_LOSS_TYPE
        if self.id_loss_type == 'cosface':
            self.classifier = Cosface(self.encoder.in_planes, num_class,
                              cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN)
        elif self.id_loss_type == 'circle':
            self.classifier = Circle(self.encoder.in_planes, num_class,
                              cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN)
        elif self.id_loss_type == 'cosine':
            self.classifier = CosineSoftmax(self.encoder.in_planes, num_class)
        else:
            self.classifier = torch.nn.Linear(self.encoder.in_planes, num_class, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        feat = self.encoder(x)

        if self.id_loss_type in ('cosface', 'circle'):
            score = self.classifier(feat, label)
        else:
            score = self.classifier(feat)

        return score, feat


class GeM(nn.Module):

    def __init__(self, size=(1, 1), p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else torch.nn.parameter(torch.ones(1) * p)
        self.eps = eps
        self.size = size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            self.size).pow(1. / self.p)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            #nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)