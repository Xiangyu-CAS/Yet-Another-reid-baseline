python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_PATH '/home/xiangyuzhu/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/xiangyuzhu/data/ReID' \
OUTPUT_DIR './output/visda20/source_baseline'
