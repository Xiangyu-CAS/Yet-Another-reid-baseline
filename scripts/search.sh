# TODO: search

python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'none' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 4 \
OUTPUT_DIR './output/public/msmt17-all/0801-only-circle-N4'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'none' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 8 \
OUTPUT_DIR './output/public/msmt17-all/0801-only-circle-N8'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'none' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 16 \
OUTPUT_DIR './output/public/msmt17-all/0801-only-circle-N16'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'none' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.BATCH_SIZE 128 \
OUTPUT_DIR './output/public/msmt17-all/0801-only-circle-N16B128'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'none' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 32 \
SOLVER.BATCH_SIZE 128 \
OUTPUT_DIR './output/public/msmt17-all/0801-only-circle-N32B128'
