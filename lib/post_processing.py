import torch.nn.functional as F
import numpy as np
import torch
import os
from .evaluation import eval_func


def re_ranking(probFea, galFea, k1, k2, lambda_value, cam_dist=None):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    gallery_num = galFea.size(0)
    all_num = query_num + gallery_num

    feat = torch.cat([probFea, galFea])
    print('using GPU to compute original distance')

    distmat = 2 - 2 * torch.mm(feat, feat.t())
    original_dist = distmat.cpu().numpy()
    del feat
    del distmat
    torch.cuda.empty_cache()
    if cam_dist is not None:
        original_dist = original_dist - 0.1 * cam_dist

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    #original_dist = original_dist / 2
    V = np.zeros_like(original_dist).astype(np.float16)
    print('argmax')
    # numpy naive search
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)


    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value

    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]

    if cam_dist is not None:
        final_dist -= 0.1*cam_dist[:query_num, query_num:]

    return final_dist


def DBA(feat, top_k=5):
    distmat = 2 - 2 * torch.mm(feat, feat.t())
    indices = distmat.argsort(dim=1)
    expanded_feat = (feat[indices[:, :top_k]]).mean(dim=1)
    return expanded_feat


def post_processor(cfg, batch, num_query):
    feats, pids, camids, img_paths = batch

    # data base augmentatio
    if cfg.TEST.DO_DBA:
        feats = DBA(feats)

    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    g_camids = np.asarray(camids[num_query:])

    if cfg.TEST.CAM_DISTMAT != '':
        cam_dist = np.load(cfg.TEST.CAM_DISTMAT)
    else:
        cam_dist = None

    if cfg.TEST.DO_RERANK:
        distmat = re_ranking(qf, gf, cfg.TEST.RERANK_PARAM[0],
                             cfg.TEST.RERANK_PARAM[1], cfg.TEST.RERANK_PARAM[2],
                             cam_dist)
    else:
        distmat = 2 - 2 * torch.mm(qf, gf.t())
        distmat = distmat.cpu().numpy()

    indices = np.argsort(distmat, axis=1)

    cmc, mAP = eval_func(indices, q_pids, g_pids, q_camids, g_camids)

    return cmc, mAP, indices
