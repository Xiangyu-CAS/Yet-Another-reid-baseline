python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/test-ensemble/finetune-50/best.pth' \
OUTPUT_DIR './output/visda20/test-ensemble/finetune-50/' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/test/feat_distmat.npy'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.BACKBONE "resnet101_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/test-ensemble/finetune-101/best.pth' \
OUTPUT_DIR './output/visda20/test-ensemble/finetune-101/' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/test/feat_distmat.npy'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnest50" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/test-ensemble/finetune-nest50/best.pth' \
OUTPUT_DIR './output/visda20/test-ensemble/finetune-nest50/' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/test/feat_distmat.npy'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "se_resnet101_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/test-ensemble/finetune-senet/best.pth' \
OUTPUT_DIR './output/visda20/test-ensemble/finetune-senet/' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/test/feat_distmat.npy'


python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('5')" \
MODEL.BACKBONE "densenet169_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.DO_RERANK True \
TEST.RERANK_PARAM '([30, 6, 0.15])' \
TEST.WRITE_FEAT True \
TEST.FLIP_TEST True \
TEST.WEIGHT './output/visda20/test-ensemble/finetune-d169/best.pth' \
OUTPUT_DIR './output/visda20/test-ensemble/finetune-d169/' \
TEST.CAM_DISTMAT '../ReID-simple-baseline/output/visda/ReCam_distmat/test/feat_distmat.npy'



#python ./tools/model_ensemble.py