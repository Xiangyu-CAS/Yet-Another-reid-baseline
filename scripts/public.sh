python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "('regnety_800mf')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/RegNetY-800MF_dds_8gpu.pyth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('msmt17',)" \
OUTPUT_DIR './output/public/msmt17-baseline'