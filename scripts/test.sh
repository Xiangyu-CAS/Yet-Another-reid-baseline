#--------------------------------------validation-------------------------------------
python ./tools/test.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/public' \
DATASETS.TEST "('rytest',)" \
TEST.WEIGHT 'output/joint/0730-circle-m0.35-128-0.25-128/best.pth'
