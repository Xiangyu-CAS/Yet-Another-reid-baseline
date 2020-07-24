#--------------------------------------validation-------------------------------------
python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('market1501',)" \
TEST.DO_RERANK False \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.FLIP_TEST False \
TEST.WEIGHT './output/visda20/0725-search/cosface-circle/finetune/best.pth'
 \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/validation/feat_distmat.npy'

TEST.CAM_DISTMAT './output/visda20/workflow/cam-model/feat_distmat.npy'

TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/validation/feat_distmat.npy'


python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'finetune' \
MODEL.PRETRAIN_PATH './output/visda20/0725-search/cosface-circle/best.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
MODEL.METRIC_LOSS_TYPE 'circle' \
MODEL.MEMORY_SCALE 30 \
SOLVER.MAX_EPOCHS  10 \
SOLVER.FREEZE_EPOCHS 0 \
SOLVER.BASE_LR 1e-4 \
OUTPUT_DIR './output/visda20/0725-search/cosface-circle/finetune'