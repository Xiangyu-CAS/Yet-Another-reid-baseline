## TODO
- [ ] memory bank
- [ ] Domain Specific BN


## Requirement
1. pytorch>=1.2.0
2. yacs
3. sklearn
4. [apex](https://github.com/NVIDIA/apex) (optional)
````
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
````


## Reproduce Results
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
- train camera ReID model on target domain dataset
```
python ./tools/train_cam.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_PATH '/home/zxy/.cache/torch/checkpoints/r50_ibn_a.pth' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
SOLVER.MAX_EPOCHS 40 \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'none' \
OUTPUT_DIR './output/visda20/workflow/cam-model'
```

- generate pseudo labels
```
python ./tools/pseudo_label.py --config_file='configs/visda20.yml' \
MODEL.DEVICE_ID "('2')" \
MODEL.BACKBONE "resnet50_ibn_a" \
MODEL.PRETRAIN_CHOICE 'self' \
DATASETS.ROOT_DIR '/home/zxy/data/ReID/visda' \
TEST.WEIGHT './output/visda20/workflow/personx-model/best.pth' \
OUTPUT_DIR './output/visda20/cluster/cluster-first'

```