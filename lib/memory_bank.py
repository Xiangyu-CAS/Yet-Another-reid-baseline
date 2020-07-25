import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.utils import euclidean_dist
from lib.losses.metric_loss import SmoothAP

class VanillaMemoryBank(object):
    def __init__(self, dim=2048, k=10000):
        self.K = k
        self.feats = torch.zeros(self.K, dim).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

class MoCo(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=2048, K=16384, m=0.25, s=96):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K

        self.m = m
        self.s = s

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K, device='cuda'))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_label", torch.zeros((1, K), dtype=torch.long, device='cuda'))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):
        # gather keys/targets before updating queue

        keys = keys.detach()
        targets = targets.detach()

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[:, ptr:ptr + batch_size] = targets
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, feat_q, feat_k, targets):
        r"""
        Memory bank enqueue and compute metric loss
        Args:
            feat_q: model features
            feat_k: model_ema features
            targets: gt labels

        Returns:
        """
        # dequeue and enqueue
        feat_k = nn.functional.normalize(feat_k, p=2, dim=1)
        self._dequeue_and_enqueue(feat_k, targets)
        loss = self._circle_loss(feat_q, targets)
        # loss = self._triplet_loss(feat_q, targets)
        return loss

    def _triplet_loss(self, feat_q, targets):
        dist_mat = euclidean_dist(feat_q, self.queue.t())

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()

        sorted_mat_distance, positive_indices = torch.sort(dist_mat + (-9999999.) * (1 - is_pos), dim=1,
                                                           descending=True)
        dist_ap = sorted_mat_distance[:, 0]
        sorted_mat_distance, negative_indices = torch.sort(dist_mat + 9999999. * is_pos, dim=1,
                                                           descending=False)
        dist_an = sorted_mat_distance[:, 0]

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

        return loss

    def _circle_loss(self, feat_q, targets):
        feat_q = nn.functional.normalize(feat_q, p=2, dim=1)

        sim_mat = torch.mm(feat_q, self.queue)

        N, M = sim_mat.size()

        is_pos = targets.view(N, 1).expand(N, M).eq(self.queue_label.expand(N, M)).float()
        same_indx = torch.eye(N, N, device='cuda')
        remain_indx = torch.zeros(N, M-N, device='cuda')
        same_indx = torch.cat((same_indx, remain_indx), dim=1)
        is_pos = is_pos - same_indx

        is_neg = targets.view(N, 1).expand(N, M).ne(self.queue_label.expand(N, M)).float()

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
