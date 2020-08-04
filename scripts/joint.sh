## TODO more rydata
python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('3')" \
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
SOLVER.MAX_EPOCHS 40 \
OUTPUT_DIR './output/joint/0803-circle-epoch40'


python ./tools/train.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('5')" \
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
OUTPUT_DIR './output/joint/0803-circle-with-NAIC'