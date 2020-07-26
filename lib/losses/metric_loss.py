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


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds, targets=None, placeholder1=None, placeholder2=None):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """


        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size, device='cuda')
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id), device='cuda')
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)


if __name__ == '__main__':
    loss = SmoothAP(0.01, 60, 6, 256)
    input = torch.randn(60, 256, requires_grad=True).cuda()
    output = loss(input)
