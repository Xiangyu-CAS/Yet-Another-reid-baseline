python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
TEST.DO_RERANK True \
TEST.WEIGHT './output/visda20/0717-search/circle/m0.2-s64/best.pth' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/validation/feat_distmat.npy'