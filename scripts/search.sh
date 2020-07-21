#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "resnet101_ibn_a" \
#MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo')" \
#MODEL.ID_LOSS_TYPE 'cosface' \
#SOLVER.COSINE_MARGIN 0.35 \
#SOLVER.COSINE_SCALE 30 \
#MODEL.MEMORY_BANK False \
#OUTPUT_DIR './output/visda20/0718-search/cosface-101'
#
#
#
#
#


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
MODEL.MEMORY_BANK True \
SOLVER.MAX_EPOCHS 100 \
OUTPUT_DIR './output/visda20/0721-search/memory-bank-epoch100'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
MODEL.MEMORY_BANK True \
SOLVER.MAX_EPOCHS 100 \
DATALOADER.NUM_INSTANCE 4 \
OUTPUT_DIR './output/visda20/0721-search/memory-bank-epoch100-N4'