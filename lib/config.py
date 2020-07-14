from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.BACKBONE = 'resnet50_ibn_a'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.AUGMIX_PROB = 0.0 # augmix
_C.INPUT.AUTOAUG_PROB = 0.0 # auto augmentation
_C.INPUT.COLORJIT_PROB = 0.0 # Color jitter
# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.DEVICE_ID = '0'
_C.SOLVER.FP16 = False
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.MAX_EPOCHS = 20
_C.SOLVER.BASE_LR = 3.5e-4

_C.SOLVER.FREEZE_EPOCHS = 0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = './data'


_C.OUTPUT_DIR = ''