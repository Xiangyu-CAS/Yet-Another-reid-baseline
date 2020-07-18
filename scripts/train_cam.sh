python ./tools/train_cam.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 40 \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'none' \
OUTPUT_DIR './output/visda20/workflow/cam-model'

