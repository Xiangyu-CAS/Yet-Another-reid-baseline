import sys
import argparse
import os
import torch
import numpy as np
import time
import shutil
sys.path.append('.')

from torch.backends import cudnn
from sklearn.cluster import DBSCAN
from lib.evaluation import eval_cluster
from lib.dataset.build_dataset import init_dataset
from lib.config import _C as cfg
from lib.modeling.build_model import Encoder
from lib.dataset.data_loader import get_test_loader
from lib.utils import setup_logger, load_checkpoint


def DBSCAN_cluster(distmat, dataset, out_dir, eps=0.6, min_samples=4):
    print('start generating pseduo label')
    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=16)
    print('Clustering and labeling...')

    labels = cluster.fit_predict(distmat)
    gts = [gt for (img_path, gt, _) in dataset]

    num_ids = len(set(labels)) - 1
    results = []
    for (img_path, gt, _), label in zip(dataset, labels):
        if label == -1 : continue
        img_name = '_'.join([str(label).zfill(4), 'c' + str(0).zfill(3),
                             't' + '0000', os.path.basename(img_path)])
        shutil.copyfile(img_path, os.path.join(out_dir, img_name))
        results.append(img_name)
    print('Got {} training ids, {} images'.format(num_ids, len(results)))

    recall, precision, fscore = eval_cluster(np.array(gts), np.array(labels))
    print("cluster recall={:.2%}, precision={:.2%}\n"
          "fscore={:.2%}\n".format(recall, precision, fscore))


def inference(cfg, logger):
    testset = 'visda20'
    #testset = 'personx'
    dataset = init_dataset(testset, root=cfg.DATASETS.ROOT_DIR)
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)

    dataset = dataset.train
    #dataset = dataset.query + dataset.gallery
    test_loader = get_test_loader(dataset, cfg)

    model = Encoder(cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_PATH,
                    cfg.MODEL.PRETRAIN_CHOICE).cuda()

    logger.info("loading model from {}".format(cfg.TEST.WEIGHT))
    load_checkpoint(model, cfg.TEST.WEIGHT)

    feats, pids, camids, img_paths = [], [], [], []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)
            img_paths.extend(img_path)
    del model
    torch.cuda.empty_cache()

    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats, dataset


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

    #cam_distmat = np.load('../ReID-simple-baseline/output/visda/ReCam_distmat/validation/feat_distmat.npy')
    #cam_distmat = np.load('../ReID-simple-baseline/output/visda/ReCam_distmat/target_training/feat_distmat.npy')
    #original_dist -= 0.1 * torch.tensor(cam_distmat, device='cuda')

    #initial_rank = torch.argsort(original_dist, dim=1)
    #initial_rank = initial_rank.cpu().numpy()
    initial_rank = np.argsort(original_dist.cpu().numpy())

    # original_dist = original_dist.cpu().numpy()
    original_dist /= 2
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
        # dist = 2 - 2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        # if use_float16:
        #     V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        # else:
        #     V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index].cpu().numpy())
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

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

    #lambda_value = 0.3
    #jaccard_dist = jaccard_dist * (1 - lambda_value) + original_dist.cpu().numpy() * lambda_value

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print ("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    feats, dataset = inference(cfg, logger)
    distmat = compute_jaccard_distance(feats)
    DBSCAN_cluster(distmat, dataset, output_dir)


if __name__ == '__main__':
    main()
