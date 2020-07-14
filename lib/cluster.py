import torch
import time
import numpy as np
from sklearn.cluster import DBSCAN
from torch.nn import functional as F

def DBSCAN_cluster(feats, dataset, logger, eps=0.6, min_samples=4):
    logger.info('start generating pseduo label')
    # distmat = 1 - torch.mm(feats, feats.t()) + 1e-5
    # distmat = distmat.cpu().numpy()

    distmat = compute_jaccard_distance(feats)

    logger.info('eps in cluster: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
    logger.info('Clustering and labeling...')
    labels = cluster.fit_predict(distmat)
    num_ids = len(set(labels)) - 1
    logger.info('Got {} training ids, {} images'.format(num_ids, sum(labels!=-1) - sum(labels==0)))

    results = []
    for (img_path, _, camid), label in zip(dataset, labels):
        if label == -1 or label == 0:
            continue
        results.append([img_path, label, camid])
    return results


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, use_float16=True):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    N = target_features.size(0)
    mat_type = np.float16
    original_dist = 2 - 2 * torch.matmul(target_features, target_features.t())
    initial_rank = torch.argsort(original_dist, dim=1)
    initial_rank = initial_rank.cpu().numpy()
    # original_dist = original_dist.cpu().numpy()
    # original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    #
    # initial_rank = np.argsort(original_dist).astype(mat_type)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1,N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist