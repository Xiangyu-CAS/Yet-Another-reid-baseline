python ./tools/pseudo_label.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
TEST.WEIGHT 'output/visda20/0724-ensemble/r50/finetune/best.pth' \
OUTPUT_DIR './output/visda20/cluster/DBSCAN-fourth'

#-----------------------------------------------------

