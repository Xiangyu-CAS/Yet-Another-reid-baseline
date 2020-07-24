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


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('personx',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.3])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/workflow/cam-model/best.pth' \

