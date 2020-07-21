python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "('regnety_800mf')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/RegNetY-800MF_dds_8gpu.pyth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
MODEL.MEMORY_BANK False \
MODEL.ID_LOSS_TYPE 'none' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.MAX_EPOCHS 40 \
SOLVER.BATCH_SIZE 256 \
OUTPUT_DIR './output/visda20/0720-search/memory-bank/epoch40'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
SOLVER.MAX_EPOCHS 40 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/visda20/0721-search/local-feat-score-mean'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
SOLVER.MAX_EPOCHS  30 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.COSINE_SCALE 64 \
MODEL.MEMORY_BANK True \
OUTPUT_DIR './output/visda20/0721-search/w-memory-bank-circle'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
SOLVER.MAX_EPOCHS  25 \
MODEL.MEMORY_BANK False \
OUTPUT_DIR './output/visda20/0721-search/baseline'

