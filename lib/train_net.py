import torch
import torch.nn.functional as F
import time
import numpy as np
import logging

from lib.modeling.build_model import Encoder, Head
from lib.dataset.data_loader import get_train_loader, get_test_loader
from lib.utils import AverageMeter
from lib.evaluation import eval_func


class Trainer(object):
    def __init__(self, cfg):
        self.encoder = Encoder(cfg.MODEL.PRETRAIN_PATH,
                               cfg.MODEL.PRETRAIN_CHOICE).cuda()
        self.cfg = cfg
        self.loss_fn = F.cross_entropy

        self.logger = logging.getLogger("reid_baseline.train")

    def do_train(self, dataset):
        num_class = dataset.num_train_pids
        model = Head(self.encoder, num_class).cuda()

        # optimizer and scheduler
        params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.cfg.SOLVER.BASE_LR, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.cfg.SOLVER.MAX_EPOCHS))

        # data loader
        train_loader = get_train_loader(dataset.train)
        test_loader = get_test_loader(dataset.query + dataset.gallery)

        best_mAP = 0
        # train
        for epoch in range(self.cfg.SOLVER.MAX_EPOCHS):
            self.logger.info("Epoch[{}] lr={:.2e}"
                        .format(epoch, scheduler.get_lr()[0]))

            #self.train_epoch(model, optimizer, train_loader, epoch)
            #scheduler.step()
            if epoch % 10 == 0:
                cur_mAP = self.validate(model, test_loader, len(dataset.query))

    def train_epoch(self, model, optimizer, train_loader, epoch):
        losses = AverageMeter()
        model.train()
        for iteration, batch in enumerate(train_loader):
            input, target, _, _ = batch
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            score, feat = model(input)
            loss = self.loss_fn(score, target)

            loss.backward()
            optimizer.step()
            losses.update(loss.item()), input.size(0)

            if iteration % 100 == 0:
                self.logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}".
                      format(epoch, iteration, len(train_loader), losses.val))

    def validate(self, model, test_loader, num_query):
        model.eval()
        feats = []
        pids = []
        camids = []
        with torch.no_grad():
            for batch in test_loader:
                data, pid, camid, img_path = batch
                data = data.cuda()
                feat = model.encoder(data)

                feats.append(feat)
                pids.append(pid)
                camids.append(camids)

        feats = torch.cat(feats, dim=0)
        feats = F.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])

        distmat = 2 - 2 * torch.mm(qf, gf) # Euclidean Distance
        indices = torch.argsort(distmat, dim=1)

        indices = indices.cpu().numpy()
        cmc, mAP = eval_func(indices,q_pids, g_pids, q_camids, g_camids)

        self.logger.info("Validation Results")
        self.logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return mAP
