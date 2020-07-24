import torch
from torch import nn
import torch.nn.functional as F
from lib.utils import euclidean_dist

def hard_example_mining(dist_mat, labels, mining_method='batch_hard'):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    if mining_method == 'batch_hard':
        # Batch Hard
        dist_ap, relative_p_inds = torch.max(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

        dist_an, relative_n_inds = torch.min(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        # shape [N]
    elif mining_method == 'batch_sample':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        relative_p_inds = torch.multinomial(
            F.softmax(dist_mat_ap, dim=1), num_samples=1)
        dist_ap = torch.gather(dist_mat_ap, 1, relative_p_inds)

        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        relative_n_inds = torch.multinomial(
            F.softmin(dist_mat_an, dim=1), num_samples=1)
        dist_an = torch.gather(dist_mat_an, 1, relative_n_inds)
    elif mining_method == 'batch_soft':
        dist_mat_ap = dist_mat[is_pos].contiguous().view(N, -1)
        dist_mat_an = dist_mat[is_neg].contiguous().view(N, -1)
        weight_ap = torch.exp(dist_mat_ap) / torch.exp(dist_mat_ap).sum(dim=1, keepdim=True)
        weight_an = torch.exp(-dist_mat_an) / torch.exp(-dist_mat_an).sum(dim=1, keepdim=True)

        dist_ap = (weight_ap * dist_mat_ap).sum(dim=1, keepdim=True)
        dist_an = (weight_an * dist_mat_an).sum(dim=1, keepdim=True)
    else:
        print("error, unsupported mining method {}".format(mining_method))

    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, mining_method='batch_hard'):
        self.margin = margin
        self.mining_method = mining_method
        if margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    # def __call__(self, global_feat, labels, normalize_feature=True):
    #     if normalize_feature:
    #         global_feat = torch.nn.functional.normalize(global_feat, dim=1, p=2)
    #     dist_mat = 2 - 2 * global_feat.mm(global_feat.t())
    #     dist_ap, dist_an = hard_example_mining(
    #         dist_mat, labels, self.mining_method)
    #
    #     y = dist_an.new().resize_as_(dist_an).fill_(1)
    #     if self.margin > 0:
    #         loss = self.ranking_loss(dist_an, dist_ap, y)
    #     else:
    #         loss = self.ranking_loss(dist_an - dist_ap, y)
    #     return loss

    def __call__(self, batch_feat, batch_labels, memory_feat, memory_labels):
        distmat = 2 - 2 * torch.mm(memory_feat, batch_feat.t())
        #distmat = euclidean_dist(memory_feat, batch_feat)
        distmat = distmat.t()

        N, M = distmat.shape

        is_pos = batch_labels.unsqueeze(dim=-1).expand(N, M).eq(memory_labels.unsqueeze(dim=-1).expand(M, N).t())
        is_neg = batch_labels.unsqueeze(dim=-1).expand(N, M).ne(memory_labels.unsqueeze(dim=-1).expand(M, N).t())
        dist_ap, relative_p_inds = torch.max(
            distmat[is_pos].contiguous().view(N, -1), 1)

        dist_an, relative_n_inds = torch.min(
            distmat[is_neg].contiguous().view(N, -1), 1)

        # is_pos = batch_labels.unsqueeze(dim=-1).expand(N, M).eq(memory_labels.unsqueeze(dim=-1).expand(M, N).t()).float()
        # sorted_mat_distance, positive_indices = torch.sort(distmat + (-9999999.) * (1 - is_pos), dim=1,
        #                                                    descending=True)
        # dist_ap = sorted_mat_distance[:, 0]
        # sorted_mat_distance, negative_indices = torch.sort(distmat + 9999999. * is_pos, dim=1,
        #                                                    descending=False)
        # dist_an = sorted_mat_distance[:, 0]

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin > 0:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class CircleLoss(nn.Module):
    def __init__(self, m=0.25, s=64):
        super(CircleLoss, self).__init__()
        self.m = m
        self.s = s
        self.soft_plus = nn.Softplus()

    def forward(self, batch_feat, batch_labels, memory_feat, memory_labels):

        sim_mat = torch.mm(batch_feat, memory_feat.t())

        N, M = sim_mat.size()

        is_pos = batch_labels.view(N, 1).expand(N, M).eq(memory_labels.expand(N, M)).float()
        same_indx = torch.eye(N, N, device='cuda')
        remain_indx = torch.zeros(N, M-N, device='cuda')
        same_indx = torch.cat((same_indx, remain_indx), dim=1)
        is_pos = is_pos - same_indx

        is_neg = batch_labels.view(N, 1).expand(N, M).ne(memory_labels.expand(N, M)).float()

        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss