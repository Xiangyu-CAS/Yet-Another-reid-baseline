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
_C.MODEL.ID_LOSS_TYPE = 'softmax' #{softmax, Cosface, Circle}
_C.MODEL.METRIC_LOSS_TYPE = 'none' #{}
_C.MODEL.MEMORY_BANK = False
# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.AUGMIX_PROB = 0.0 # augmix
_C.INPUT.AUTOAUG_PROB = 0.0 # auto augmentation
_C.INPUT.COLORJIT_PROB = 0.0 # Color jitter
_C.INPUT.RE_PROB = 0.0#random erase
_C.INPUT.FIGURE_PROB = 0.0 # random figures
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

# parameters for large margin softmax loss
_C.SOLVER.COSINE_MARGIN = 0.35
_C.SOLVER.COSINE_SCALE = 30

_C.SOLVER.EVAL_PERIOD = 2
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------

_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'random' #{random, randomIdentity}
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = './data'

_C.DATASETS.TRAIN = ()
_C.DATASETS.TEST = ()

#-------------------------------------------------------------------------------
# Test
#-------------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.WEIGHT = ''
_C.TEST.DO_DBA = False
_C.TEST.DO_RERANK = False
_C.TEST.RERANK_PARAM = [20, 6, 0.3]
_C.TEST.CAM_DISTMAT = ''

_C.OUTPUT_DIR = ''