


# TODO:
# 1. train from synthetic pre-train
python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('4')" \
DATASETS.ROOT_DIR '/datassd/zxy/ReID/joint/' \
DATASETS.TRAIN "('market1501', 'synthetic',)" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/synthetic/pretrain'


python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('4')" \
DATASETS.ROOT_DIR '/datassd/zxy/ReID/joint/' \
DATASETS.TRAIN "('msmt17','poly', 'market1501', 'rydata','lpw')" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/synthetic/pretrain/best.pth' \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/joint/0730-circle-finetune-from-syn'

# 2. num_instance=16
python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('3')" \
DATASETS.ROOT_DIR '/datassd/zxy/ReID/joint/' \
DATASETS.TRAIN "('msmt17','poly', 'market1501', 'rydata','lpw')" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.35 \
MODEL.ID_LOSS_SCALE 64 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 96 \
MODEL.MEMORY_BANK False \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.MAX_EPOCHS 12 \
OUTPUT_DIR './output/joint/0729-circle-N16'

# 3. circle m, s
python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('2')" \
DATASETS.ROOT_DIR '/datassd/zxy/ReID/joint/' \
DATASETS.TRAIN "('msmt17','poly', 'market1501', 'rydata','lpw')" \
DATASETS.TEST "('dukemtmc',)" \
MODEL.ID_LOSS_TYPE "circle" \
MODEL.ID_LOSS_MARGIN 0.25 \
MODEL.ID_LOSS_SCALE 128 \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.METRIC_LOSS_MARGIN 0.25 \
MODEL.METRIC_LOSS_SCALE 128 \
MODEL.MEMORY_BANK False \
SOLVER.BATCH_SIZE 192 \
OUTPUT_DIR './output/joint/0730-circle-naive-v3-sampler'

