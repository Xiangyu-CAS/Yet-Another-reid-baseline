# TODO: search paramters
#python ./tools/train.py --config_file='configs/public.yml' \
#MODEL.DEVICE_ID "('1')" \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
#DATASETS.TRAIN "('msmt17',)" \
#DATASETS.TEST "('dukemtmc',)" \
#DATASETS.COMBINEALL True \
#MODEL.PRETRAIN_CHOICE 'finetune' \
#MODEL.PRETRAIN_PATH './output/synthetic/pretrain/best.pth' \
#MODEL.ID_LOSS_TYPE "circle" \
#MODEL.METRIC_LOSS_TYPE 'circle' \
#OUTPUT_DIR './output/public/msmt17-all/train-from-syn-pretrain'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('4')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TRAIN "('msmt17',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
SOLVER.MAX_EPOCHS 30 \
OUTPUT_DIR './output/public/msmt17-train/circle-circle-better-sampler'


python ./tools/train.py --config_file='configs/public.yml' \
MODEL.DEVICE_ID "('2')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/' \
DATASETS.TRAIN "('market1501','dukemtmc',)" \
DATASETS.TEST "('dukemtmc',)" \
DATASETS.COMBINEALL False \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
SOLVER.MAX_EPOCHS 30 \
OUTPUT_DIR './output/public/multi-dataset/section-sampler'