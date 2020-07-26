python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('4')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
SOLVER.COSINE_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'none' \
OUTPUT_DIR './output/public/msmt17-only-circle'