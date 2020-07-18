python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo')" \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_MARGIN 0.2 \
SOLVER.COSINE_SCALE 64 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/visda20/0717-search/circle/r101'




