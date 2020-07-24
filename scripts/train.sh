#--------------------train-------------------------------------------------------
python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/0724-ensemble/r101'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
MODEL.METRIC_LOSS_TYPE 'circle' \
OUTPUT_DIR './output/visda20/0724-ensemble/r101-circle'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "('resnest50')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnest50-528c19ca.pth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/0724-ensemble/nest50'

#-------------------- finetune---------------------------------------------------
python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0724-ensemble/r50/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS  10 \
SOLVER.FREEZE_EPOCHS 0 \
OUTPUT_DIR './output/visda20/0723-search/finetune-50'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0724-ensemble/r101/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS  30 \
SOLVER.FREEZE_EPOCHS 0 \
OUTPUT_DIR './output/visda20/0724-ensemble/r101/finetune'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnest50" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0724-ensemble/nest50/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS  10 \
SOLVER.FREEZE_EPOCHS 0 \
OUTPUT_DIR './output/visda20/0724-ensemble/nest50/finetune'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnest101" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0723-ensemble/nest101/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS  10 \
SOLVER.FREEZE_EPOCHS 0 \
OUTPUT_DIR './output/visda20/0723-search/finetune-nest101'

#-------------------------- BN adaptive----------------------

python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0725-search/ori-personx/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS  30 \
SOLVER.FREEZE_EPOCHS 0 \
OUTPUT_DIR './output/visda20/0725-search/ori-personx/finetune'