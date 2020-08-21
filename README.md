The repo contains our code for VisDA 2020 challenge

## What's new
- [x] practical post-process: remove camera bias
- [x] circle loss (class-level and pair-wise)
- [x] memory bank
- [x] Mix precision (FP16)
- [x] Advanced Data augmentation: augmix, auto-augmentation
- [x] pseudo label based method for unsupervised learning
- [ ] efficient search and re-rank by [faiss](https://github.com/facebookresearch/faiss)
- [ ] Multi-GPU (single node DDP)
- [ ] SOTA benchmark
- [ ] [Distillation](https://github.com/JDAI-CV/fast-reid/tree/master/projects/DistillReID)

## Requirement
1. pytorch>=1.2.0
2. yacs
3. sklearn
4. [apex](https://github.com/NVIDIA/apex) 
5. faiss (pip install faiss-gpu)

## Reproduce results on VisDA 2020 Challenge
Refer to [VISDA20.md](VISDA20.md) and [tech_report](tech_report.pdf), 
trained models can be download from [here](https://drive.google.com/drive/folders/1JuyChtNFlPc_9ZYwSuJyQGaBfszaQ9zM?usp=sharing)

- leaderboard (ranged by rank1)

|team|mAP|rank1|
|----|---|-----|
|vimar|76.56%|84.25%|
|**xiangyu(ours)**|72.39%|83.85%|
|yxge|74.78%|82.86%|

- Ablation on validation set

|method|mAP|rank1|
|------|---|-----|
|personx-spgan|37.7%|63.7%|
|+pseudo label|51.8%|77.7%|
|+BN finetune|55.5%|81.4%|
|+re-rank|73.4%|80.9%|
|**+remove camera bias**|**79.5%**|**89.1%**|
|ensemble|82.7%|90.7%|

## Benchmark
Setting: ResNet50-ibn-a, single RTX 2080 Ti, FP16

- market1501

|method|mAP|rank1|
|------|---|-----|
|bag-of-tricks|88.2%|95.0%|
|fast reid|89.3%|95.3%|
|ours|88.4%|95.1%|


- dukemtmc-reid

|method|mAP|rank1|
|------|---|-----|
|bag-of-tricks|79.1%|90.1%|
|fast-reid|81.2%|90.8%|
|ours|80.1%|90.3%|

- msmt17(v2)

|method|mAP|rank1|
|------|---|-----|
|Bag of Tricks|54.4%|77.0%|
|fast reid|60.6%|83.9%|
|ours|60.6%|83.1%|