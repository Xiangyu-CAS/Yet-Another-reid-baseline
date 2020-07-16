from .id_loss import CrossEntropyLabelSmooth
from .metric_loss import TripletLoss

def build_loss_fn(cfg, num_classes):
    id_loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes)
    metric_loss_fn = TripletLoss(margin=0.0)

    def loss_func(score, feat, target):
        loss = id_loss_fn(score, target) + metric_loss_fn(feat, target)
        return loss
    return loss_func