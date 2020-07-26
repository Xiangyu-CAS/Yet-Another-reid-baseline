python ./tools/train_cam.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 30 \
INPUT.AUTOAUG_PROB 0.0 \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'none' \
OUTPUT_DIR './output/visda20/workflow/cam-model'

python ./tools/train_cam.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 30 \
INPUT.AUTOAUG_PROB 0.0 \
DATALOADER.SAMPLER 'none' \
MODEL.METRIC_LOSS_TYPE 'none' \
OUTPUT_DIR './output/visda20/workflow/cam-model-N16'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('personx',)" \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.DO_DBA True \
TEST.WEIGHT './output/visda20/workflow/cam-model/best.pth' \

