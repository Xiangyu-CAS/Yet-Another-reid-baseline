from .id_loss import CrossEntropyLabelSmooth
from .metric_loss import TripletLoss, CircleLoss

def build_loss_fn(cfg, num_classes):
    if cfg.MODEL.ID_LOSS_TYPE == 'none':
        def id_loss_fn(score, target):
            return 0
    else:
        id_loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes)

    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_fn = TripletLoss(margin=0.0)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'circle':
        metric_loss_fn = CircleLoss()
    else:
        def metric_loss_fn(feat, target, memory_feat, memory_target):
            return 0

    # def loss_func(score, feat, target):
    #     loss = id_loss_fn(score, target) + metric_loss_fn(feat, target)
    #    return loss
    def loss_func(score, feat, target, memory_feat, memory_target):
        return id_loss_fn(score, target), metric_loss_fn(feat, target, memory_feat, memory_target)

    return loss_func