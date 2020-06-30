import torch


def load_checkpoint(model, path):
    param_dict = torch.load(path)
    model_keys = model.state_dict().keys()
    for i in param_dict:
        if i not in model_keys:
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