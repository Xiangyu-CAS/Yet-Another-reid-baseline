## Reproduce Results
- prepare data: we assuming your data are structured as below
```
- visda
--personX
---image_train #(personx)
---image_train_spgan #(personx_spganx)
---image_query # (validation)
---image_gallery #(validation)

--visda20
---image_train #(target_train)
---image_query #(test)
---image_gallery #(test)

--pseudo
---DBSCAN-first # TODO: cluster from target_train
---DBSCAN-second # TODO: cluster from target_train
---DBSCAN-third # TODO: cluster from target_train

```

- Download ImageNet pretrian model from [IBN-Net](https://github.com/XingangPan/IBN-Net)


- train base model on source domain dataset
````
python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx',)" \
OUTPUT_DIR './output/visda20/workflow/persxon-model'
````
- train camera ReID model on target domain dataset and generate camera distmat for test
```
# train camera ReID
python ./tools/train_cam.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 40 \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'none' \
OUTPUT_DIR './output/visda20/workflow/cam-model'

# generate feats and cam_distmat   ./output/visda20/workflow/cam-model/feats.npy
python ./tools/test.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TEST "('visda20',)" \
TEST.WEIGHT './output/visda20/workflow/cam-model/best.pth' \
TEST.WRITE_FEAT True

# ./output/visda20/workflow/cam-model/feat_distmat.npy
python ./tools/comput_distmat.py 
```

- generate pseudo labels
```
python ./tools/pseudo_label.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
TEST.WEIGHT './output/visda20/workflow/personx-model/best.pth' \
OUTPUT_DIR './output/visda20/cluster/DBSCAN-first'


mv ./output/visda20/cluster/DBSCAN-first visda20/pseudo/
revise ./lib/dataset/visda20_pseudo.py line19 to DBSCAN-first

```

- joint train personx_spgan and pseudo labels
```
python ./tools/train.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('3')" \
MODEL.BACKBONE "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth')" \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
DATASETS.TRAIN "('personx', 'visda20_pseudo',)" \
OUTPUT_DIR './output/visda20/workflow/DBSCAN-first'
```

- generate better pseudo label based on DBSCAN-first model, you can iter it for the third time for minor increment
```
python ./tools/pseudo_label.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
TEST.WEIGHT './output/visda20/workflow/DBSCAN-first/best.pth' \
OUTPUT_DIR './output/visda20/cluster/DBSCAN-second'


mv ./output/visda20/cluster/DBSCAN-second visda20/pseudo/
revise ./lib/dataset/visda20_pseudo.py line19 to DBSCAN-second
```

- train 5 ensemble model and finetune BN
````
./scripts/train.sh
````

- generate 5 distmat and average them
```
./scripts/submit.sh
```

