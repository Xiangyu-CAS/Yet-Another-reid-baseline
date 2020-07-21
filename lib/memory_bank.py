import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import euclidean_dist

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


class MoCo(nn.Module):
    """
    Build a MoCo memory with a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, dim=2048, K=12800):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K

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
            feat_q:
            feat_k:
            targets:

        Returns:
        """
        self._dequeue_and_enqueue(feat_k, targets)

        #dist_mat = 2 - 2 * torch.mm(feat_q, self.queue)
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