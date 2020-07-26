import torch
from .id_loss import CrossEntropyLabelSmooth
from .metric_loss import TripletLoss, CircleLoss, SmoothAP

def build_loss_fn(cfg, num_classes):
    if cfg.MODEL.ID_LOSS_TYPE == 'none':
        def id_loss_fn(score, target):
            return 0
    else:
        id_loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes)

    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_fn = TripletLoss(margin=0.0)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'circle':
        metric_loss_fn = CircleLoss(m=cfg.MODEL.MEMORY_MARGIN, s=cfg.MODEL.MEMORY_SCALE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'smoothAP':
        assert(cfg.SOLVER.BATCH_SIZE % cfg.DATALOADER.NUM_INSTANCE==0)
        metric_loss_fn = SmoothAP(anneal=0.01, batch_size=cfg.SOLVER.BATCH_SIZE,
                 num_id=cfg.SOLVER.BATCH_SIZE // cfg.DATALOADER.NUM_INSTANCE, feat_dims=2048)
    else:
        def metric_loss_fn(feat, target, memory_feat, memory_target):
            return 0

    triplet_loss_fn = TripletLoss(margin=0.0)
    circle_loss_fn = CircleLoss(m=cfg.MODEL.MEMORY_MARGIN, s=cfg.MODEL.MEMORY_SCALE)
    smooth_loss_fn = SmoothAP(anneal=0.01, batch_size=cfg.SOLVER.BATCH_SIZE,
                              num_id=cfg.SOLVER.BATCH_SIZE // cfg.DATALOADER.NUM_INSTANCE, feat_dims=2048)

    # def loss_func(score, feat, target):
    #     loss = id_loss_fn(score, target) + metric_loss_fn(feat, target)
    #    return loss
    def loss_func(score, feat, target, memory_feat, memory_target):
        #return id_loss_fn(score, target), smooth_loss_fn(feat) #metric_loss_fn(feat, target, memory_feat, memory_target) +
        #three loss#return id_loss_fn(score, target), circle_loss_fn(feat, target, memory_feat, memory_target)+triplet_loss_fn(feat, target, memory_feat, memory_target)
        return id_loss_fn(score, target), metric_loss_fn(feat, target, memory_feat, memory_target)

        #return id_loss_fn(score, target), smooth_loss_fn(feat)

    return loss_func