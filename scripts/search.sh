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
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/0725-search/r50-ibn-in'

python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
MODEL.METRIC_LOSS_TYPE 'circle' \
OUTPUT_DIR './output/visda20/0725-search/circle-circle'