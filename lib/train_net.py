import torch
import torch.nn.functional as F


from lib.modeling.build_model import Encoder, Head
from lib.dataset.data_loader import get_train_loader, get_test_loader
from lib.utils import AverageMeter


class Trainer(object):
    def __init__(self, encoder, cfg):
        self.encoder = encoder
        self.cfg = cfg
        self.loss_fn = F.cross_entropy

    def do_train(self, dataset):
        num_class = 1
        model = Head(self.encoder, num_class)

        # optimizer and scheduler
        params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.cfg.SOLVER.BASE_LR, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.cfg.SOLVER.MAX_EPOCHS))

        # data loader
        train_loader = get_train_loader(dataset.train)
        test_loader = get_test_loader(dataset.query + dataset.gallery)

        # train
        for epoch in range(self.cfg.SOLVER.MAX_EPOCHS):
            losses = AverageMeter()

            for iteration, batch in enumerate(train_loader):
                input, target, _, _ = batch
                input = input.cuda()
                target = target.cuda()

                optimizer.zero_grad()
                score, feat = model(input, target)
                loss = self.loss_fn(score, target)

                loss.backward()
                optimizer.step()
                losses.update(loss.item()), input.size(0)

                if iteration % 100 == 0:
                    print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}".
                          format(epoch, iteration, len(train_loader), losses.val))
            scheduler.step()
