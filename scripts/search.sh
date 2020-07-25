#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "('resnest50')" \
#MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnest50-528c19ca.pth')" \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0722-ensemble/nest50'
#
#
#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "resnet101_ibn_a" \
#MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0723-ensemble/r-101'
#
#
#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "resnet50_ibn_a" \
#MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0722-ensemble/r50'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
SOLVER.MAX_EPOCHS 40 \
MODEL.ID_LOSS_TYPE 'none' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.BATCH_SIZE 192 \
OUTPUT_DIR './output/visda20/0726-search/smoothAP/only-smoothAP-N4-0.001'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH './output/visda20/0726-search/smoothAP/baseline/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
SOLVER.MAX_EPOCHS 30 \
SOLVER.FREEZE_EPOCHS 0 \
MODEL.ID_LOSS_TYPE 'cosface' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.BATCH_SIZE 192 \
OUTPUT_DIR './output/visda20/0726-search/smoothAP/only-smoothAP-finetune'




python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
OUTPUT_DIR './output/visda20/0726-search/ema'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo')" \
MODEL.EMA_MOMENTUM 0.998 \
OUTPUT_DIR './output/visda20/0726-search/ema-all'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "('densenet169_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/densenet169_ibn_a.pth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/0726-search/densenet169_ibn_a'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/0726-search/fc-bn'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
SOLVER.MAX_EPOCHS 30 \
OUTPUT_DIR './output/visda20/0726-search/fc-bn-personx'