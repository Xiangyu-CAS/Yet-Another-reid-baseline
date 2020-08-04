#--------------------------------------validation-------------------------------------
python ./tools/test.py --config_file='configs/joint.yml' \
MODEL.DEVICE_ID "('6')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/half' \
DATASETS.TEST "('dukemtmc',)" \
TEST.WEIGHT 'output/joint/0801-circle-triplet-bank/best.pth'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('personx',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WEIGHT 'output/visda20/test-ensemble/finetune-50/best.pth' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/validation/feat_distmat.npy'


