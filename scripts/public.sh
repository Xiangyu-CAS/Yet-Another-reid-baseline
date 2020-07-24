python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('5')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.MEMORY_BANK True \
MODEL.MEMORY_SCALE 64 \
SOLVER.COSINE_SCALE 64 \
OUTPUT_DIR './output/public/msmt17-baseline-memory-bank-circle-S64'