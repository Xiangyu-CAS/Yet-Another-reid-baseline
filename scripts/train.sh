python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 60 \
DATALOADER.SAMPLER 'balanced_identity' \
OUTPUT_DIR './output/visda20/0716-search/r101-balanced'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.BACKBONE 'resnet50_ibn_a' \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 60 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_SCALE 64 \
DATALOADER.SAMPLER 'balanced_identity' \
OUTPUT_DIR './output/visda20/0716-search/circle'