from yacs.config import CfgNode as CN

_C = CN()

#---------------------------- model-----------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'


#----------------------------SOLVER-----------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 20
_C.SOLVER.BASE_LR = 3.5e-4

_C.SOLVER.FREEZE_BASE_EPOCHS = 0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = './data'


_C.OUTPUT_DIR = ''