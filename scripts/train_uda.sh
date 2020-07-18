python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATALOADER.SAMPLER 'balanced_identity' \
SOLVER.MAX_EPOCHS 60 \
OUTPUT_DIR './output/visda20/0716-search/iterative-dbscan'


python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH './output/visda20/.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 25 \
OUTPUT_DIR './output/visda20/ibn_baseline-trainfromsyn'



python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "('regnety_800mf')" \
MODEL.PRETRAIN_PATH "('/home/xiangyuzhu/.cache/torch/checkpoints/RegNetY-800MF_dds_8gpu.pyth')" \
DATASETS.ROOT_DIR '/home/xiangyuzhu/data/ReID/visda' \
OUTPUT_DIR './output/visda20/regnet_baseline'