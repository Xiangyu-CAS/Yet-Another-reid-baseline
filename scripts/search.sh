#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "('resnest50')" \
#MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/resnest50-528c19ca.pth')" \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0722-ensemble/nest50'
#
#
#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "resnet101_ibn_a" \
#MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r101_ibn_a.pth' \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0723-ensemble/r-101'
#
#
#python ./tools/train.py --config_file='configs/visda20.yml' \
#MODEL.DEVICE_ID "('3')" \
#MODEL.BACKBONE "resnet50_ibn_a" \
#MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
#DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
#DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
#SOLVER.MAX_EPOCHS  50 \
#OUTPUT_DIR './output/visda20/0722-ensemble/r50'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
SOLVER.MAX_EPOCHS 40 \
MODEL.ID_LOSS_TYPE 'none' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.BATCH_SIZE 192 \
OUTPUT_DIR './output/visda20/0726-search/smoothAP/only-smoothAP-N4-0.001'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH './output/visda20/0726-search/smoothAP/baseline/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
SOLVER.MAX_EPOCHS 30 \
SOLVER.FREEZE_EPOCHS 0 \
MODEL.ID_LOSS_TYPE 'cosface' \
DATALOADER.NUM_INSTANCE 16 \
SOLVER.BATCH_SIZE 192 \
OUTPUT_DIR './output/visda20/0726-search/smoothAP/only-smoothAP-finetune'




python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo','personx')" \
SOLVER.MAX_EPOCHS 40 \
OUTPUT_DIR './output/visda20/0727-search/epoch40'

python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo','personx')" \
SOLVER.MAX_EPOCHS 40 \
OUTPUT_DIR './output/visda20/0727-search/triplet-0.3'

python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo','personx')" \
SOLVER.MAX_EPOCHS 40 \
MODEL.MEMORY_BANK True \
MODEL.MEMORY_SIZE 32768 \
OUTPUT_DIR './output/visda20/0727-search/memory-bank-32768'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
SOLVER.MAX_EPOCHS 40 \
MODEL.METRIC_LOSS_TYPE 'smoothAP' \
OUTPUT_DIR './output/visda20/0727-search-personx/'



python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('visda20_pseudo',)" \
SOLVER.MAX_EPOCHS 40 \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'none' \
SOLVER.BATCH_SIZE 128 \
OUTPUT_DIR './output/visda20/0727-search-personx/only-cosface-random-sampler-B128'