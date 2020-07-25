import numpy as np
import sys
import os
sys.path.append('.')
from lib.dataset.build_dataset import init_dataset
from lib.evaluation import eval_func
from lib.utils import write_result


def main():
    dataset = init_dataset('visda20', root='/home/zxy/data/ReID/visda') # {personx, visda20}

    distmat_path = ['./output/visda20/0723-search/finetune-50/distmat.npy',
                    './output/visda20/0723-search/finetune-101/distmat.npy',
                    './output/visda20/0723-search/finetune-nest50/distmat.npy',
                    './output/visda20/0723-search/finetune-senet/distmat.npy',
                    ]
    distmat = []
    for path in distmat_path:
        distmat.append(np.load(path))
    distmat = sum(distmat) / len(distmat)

    indices = np.argsort(distmat, axis=1)
    write_result(indices, os.path.dirname('./output/visda20/submit/'))

    q_pids, g_pids, q_camids, g_camids = [], [], [], []
    for img_path, pid, camid in dataset.query:
        q_pids.append(pid)
        q_camids.append(camid)
    for img_path, pid, camid in dataset.gallery:
        g_pids.append(pid)
        g_camids.append(camid)
    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)
    q_camids = np.array(q_camids)
    g_camids = np.array(g_camids)
    cmc, mAP = eval_func(indices, q_pids, g_pids, q_camids, g_camids)
    print('Validation Results')
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


if __name__ == '__main__':
    main()