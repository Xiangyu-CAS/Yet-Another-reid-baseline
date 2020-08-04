## What's new
- [x] circle loss (both classification and pair-wise)
- [x] memory bank for metric loss
- [x] Data augmentation: augmix, auto-augmentation
- [ ] Distributed Training (on going)

## Requirement
1. pytorch>=1.2.0
2. yacs
3. sklearn
4. [apex](https://github.com/NVIDIA/apex) (optional but strong recommended, if you don't have apex
installed, set option SOLVER.FP16=False in training)
````
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
````

## Reproduce results on VisDA 2020 Challenge
Refer to [VISDA20.md](VISDA20.md)

leader board

|team|mAP|rank1|
|----|---|-----|
|vimar|76.56%|84.25%|
|**xiangyu(ours)**|72.39%|83.95%|
|log|79.05%|83.26%|
|yxge|74.78%|82.86%|