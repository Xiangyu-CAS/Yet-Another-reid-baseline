# market1501
python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('5')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('dukemtmc',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/public/dukemtmc/wo-memory'



python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK True \
MODEL.MEMORY_SIZE 32768 \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/SIZE-32768'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK True \
MODEL.MEMORY_SIZE  8192 \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/SIZE-8192'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/m0.25-s64-m0.25-s96'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/m0.35-s128-m0.25-s96'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/m0.35-s64-m0.25-s128'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/m0.25-s128-m0.25-s128'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/circle-circle-wo-bank'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('4')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'triplet' \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/public/msmt17-memory-bank-search/circle-triplet-w-bank'
