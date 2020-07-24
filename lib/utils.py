import torch
import os
import sys
import logging


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_checkpoint(model, path):
    param_dict = torch.load(path)
    if 'model_state' in param_dict.keys():
        param_dict = param_dict['model_state']
    elif 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']

    for i in param_dict:
        if i not in model.state_dict().keys():
            print('skip key {}'.format(i))
            continue
        if model.state_dict()[i].shape != param_dict[i].shape:
            print('skip {}, shape dismatch {} vs {}'.format(i, model.state_dict()[i].shape, param_dict[i].shape))
            continue
        model.state_dict()[i].copy_(param_dict[i])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def freeze_module(module):
    if isinstance(module, torch.nn.Conv2d):
        module.weight.requires_grad_(False)
        #module.bias.requires_grad_(False)
    else:
        for name, child in module.named_children():
            freeze_module(child)


def write_result(indices, dst_dir, topk=100):
    indices = indices[:, :topk]
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open(os.path.join(dst_dir, 'result.txt'), 'w') as f:
        for i in range(m):
            write_line = indices[i]
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f.write(write_line)