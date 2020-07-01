python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
OUTPUT_DIR './output/visda20/source_baseline'



python ./tools/train_uda.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "('regnety_800mf')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/RegNetY-800MF_dds_8gpu.pyth')" \
SOLVER.BATCH_SIZE 256 \
SOLVER.MAX_EPOCHS 5 \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
OUTPUT_DIR './output/visda20/iter5-E10-0.4-from-ImageNet'

