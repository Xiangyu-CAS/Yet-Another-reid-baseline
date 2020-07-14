import torch
import math
import torch.nn as nn
import numpy as np

#---------------------------------------- configs-------------------------------------------------------------
import os
from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
regnet_cfg = _C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type
_C.MODEL.TYPE = ""

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = "cross_entropy"


# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.RESNET = CN()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = True


# ---------------------------------------------------------------------------- #
# AnyNet options
# ---------------------------------------------------------------------------- #
_C.ANYNET = CN()

# Stem type
_C.ANYNET.STEM_TYPE = "plain_block"

# Stem width
_C.ANYNET.STEM_W = 32

# Block type
_C.ANYNET.BLOCK_TYPE = "plain_block"

# Depth for each stage (number of blocks in the stage)
_C.ANYNET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.ANYNET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.ANYNET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.ANYNET.BOT_MULS = []

# Group widths for each stage (applies to bottleneck block)
_C.ANYNET.GROUP_WS = []

# Whether SE is enabled for res_bottleneck_block
_C.ANYNET.SE_ON = False

# SE ratio
_C.ANYNET.SE_R = 0.25

# ---------------------------------------------------------------------------- #
# RegNet options
# ---------------------------------------------------------------------------- #
_C.REGNET = CN()

# Stem type
_C.REGNET.STEM_TYPE = "simple_stem_in"
# Stem width
_C.REGNET.STEM_W = 32
# Block type
_C.REGNET.BLOCK_TYPE = "res_bottleneck_block"
# Stride of each stage
_C.REGNET.STRIDE = 2
# Squeeze-and-Excitation (RegNetY)
_C.REGNET.SE_ON = False
_C.REGNET.SE_R = 0.25

# Depth
_C.REGNET.DEPTH = 10
# Initial width
_C.REGNET.W0 = 32
# Slope
_C.REGNET.WA = 5.0
# Quantization
_C.REGNET.WM = 2.5
# Group width
_C.REGNET.GROUP_W = 16
# Bottleneck multiplier (bm = 1 / b from the paper)
_C.REGNET.BOT_MUL = 1.0


# ---------------------------------------------------------------------------- #
# EfficientNet options
# ---------------------------------------------------------------------------- #
_C.EN = CN()

# Stem width
_C.EN.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EN.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EN.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EN.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) ratio
_C.EN.SE_R = 0.25

# Strides for each stage (applies to the first block of each stage)
_C.EN.STRIDES = []

# Kernel sizes for each stage
_C.EN.KERNELS = []

# Head width
_C.EN.HEAD_W = 1280

# Drop connect ratio
_C.EN.DC_RATIO = 0.0

# Dropout ratio
_C.EN.DROPOUT_RATIO = 0.0


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Image size
_C.TRAIN.IM_SIZE = 224

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Image size
_C.TEST.IM_SIZE = 256

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = True


# ---------------------------------------------------------------------------- #
# Precise timing options
# ---------------------------------------------------------------------------- #
_C.PREC_TIME = CN()

# Perform precise timing at the start of training
_C.PREC_TIME.ENABLED = False

# Total mini-batch size
_C.PREC_TIME.BATCH_SIZE = 128

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port for initializing multi-process groups
_C.HOST = "localhost"
_C.PORT = 10001

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


#------------------------------------model structure----------------------------------
def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and regnet_cfg.BN.ZERO_INIT_FINAL_GAMMA
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(
        stem_type
    )
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(
        block_type
    )
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        assert (
            bm is None and gw is None and se_r is None
        ), "Vanilla block does not support bm, gw, and se_r options"
        super(VanillaBlock, self).__init__()
        self.construct(w_in, w_out, stride)

    def construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.construct(w_in, w_out, stride)

    def construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        assert (
            bm is None and gw is None and se_r is None
        ), "Basic transform does not support bm, gw, and se_r options"
        super(ResBasicBlock, self).__init__()
        self.construct(w_in, w_out, stride)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)

    def construct(self, w_in, w_out, stride):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.construct(w_in, w_se)

    def construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Activation, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self.construct(w_in, w_out, stride, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bm, gw, se_r):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = nn.BatchNorm2d(w_b, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # Squeeze-and-Excitation (SE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.construct(w_in, w_out, stride, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)

    def construct(self, w_in, w_out, stride, bm, gw, se_r):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.construct(w_in, w_out)

    def construct(self, w_in, w_out):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.construct(w_in, w_out)

    def construct(self, w_in, w_out):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self.construct(in_w, out_w)

    def construct(self, in_w, out_w):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_w, eps=regnet_cfg.BN.EPS, momentum=regnet_cfg.BN.MOM)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self.construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self.construct(
                stem_type=kwargs["stem_type"],
                stem_w=kwargs["stem_w"],
                block_type=kwargs["block_type"],
                ds=kwargs["ds"],
                ws=kwargs["ws"],
                ss=kwargs["ss"],
                bms=kwargs["bms"],
                gws=kwargs["gws"],
                se_r=kwargs["se_r"],
                nc=kwargs["nc"],
            )
        else:
            self.construct(
                stem_type=regnet_cfg.ANYNET.STEM_TYPE,
                stem_w=regnet_cfg.ANYNET.STEM_W,
                block_type=regnet_cfg.ANYNET.BLOCK_TYPE,
                ds=regnet_cfg.ANYNET.DEPTHS,
                ws=regnet_cfg.ANYNET.WIDTHS,
                ss=regnet_cfg.ANYNET.STRIDES,
                bms=regnet_cfg.ANYNET.BOT_MULS,
                gws=regnet_cfg.ANYNET.GROUP_WS,
                se_r=regnet_cfg.ANYNET.SE_R if regnet_cfg.ANYNET.SE_ON else None,
                nc=regnet_cfg.MODEL.NUM_CLASSES,
            )
        self.apply(init_weights)

    def construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gws = gws if gws else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        # Construct the stages
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r)
            )
            prev_w = w
        # Construct the head
        self.in_planes = prev_w
        #self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, last_stride):
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            regnet_cfg.REGNET.WA, regnet_cfg.REGNET.W0, regnet_cfg.REGNET.WM, regnet_cfg.REGNET.DEPTH
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [regnet_cfg.REGNET.GROUP_W for _ in range(num_s)]
        bms = [regnet_cfg.REGNET.BOT_MUL for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [regnet_cfg.REGNET.STRIDE for _ in range(num_s)]
        ss[-1] = last_stride
        # Use SE for RegNetY
        se_r = regnet_cfg.REGNET.SE_R if regnet_cfg.REGNET.SE_ON else None
        # Construct the model
        kwargs = {
            "stem_type": regnet_cfg.REGNET.STEM_TYPE,
            "stem_w": regnet_cfg.REGNET.STEM_W,
            "block_type": regnet_cfg.REGNET.BLOCK_TYPE,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": regnet_cfg.MODEL.NUM_CLASSES,
        }
        super(RegNet, self).__init__(**kwargs)


def regnety_800mf(last_stride=1, **kwargs):
    regnet_cfg.REGNET.SE_ON = True
    regnet_cfg.REGNET.DEPTH = 14
    regnet_cfg.REGNET.W0 = 56
    regnet_cfg.REGNET.WA = 38.84
    regnet_cfg.REGNET.WM = 2.4
    regnet_cfg.REGNET.GROUP_W = 16
    return RegNet(last_stride)


