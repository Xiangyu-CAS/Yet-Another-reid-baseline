from yacs.config import CfgNode as CN

_C = CN()

#---------------------------- model-----------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'


#----------------------------SOLVER-----------------------------------
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 3e-4

_C.SOLVER.FREEZE_BASE_EPOCHS = 0