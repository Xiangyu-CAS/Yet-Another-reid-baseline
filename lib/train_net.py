import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import logging

from lib.modeling.build_model import Encoder, Head
from lib.dataset.data_loader import get_train_loader, get_test_loader
from lib.utils import AverageMeter, load_checkpoint
from lib.evaluation import eval_func
from lib.losses.build_loss import build_loss_fn
from lib.memory_bank import VanillaMemoryBank

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you set "
                      "SOLVER.FP16=True")


class Trainer(object):
    def __init__(self, cfg):
        self.encoder = Encoder(cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_PATH,
                               cfg.MODEL.PRETRAIN_CHOICE).cuda()
        self.cfg = cfg

        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            load_checkpoint(self.encoder, cfg.MODEL.PRETRAIN_PATH)

        self.logger = logging.getLogger("reid_baseline.train")
        self.best_mAP = 0
        self.use_memory_bank = False
        if cfg.MODEL.MEMORY_BANK:
            self.encoder_ema = Encoder(cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_PATH,
                                       cfg.MODEL.PRETRAIN_CHOICE).cuda()
            for param_k in self.encoder_ema.parameters():
                param_k.requires_grad = False  # not update by gradient

            self.memory_bank = VanillaMemoryBank()
            self.use_memory_bank = True

    def do_train(self, dataset):
        num_class, _, _ = dataset.get_imagedata_info(dataset.train)

        model = Head(self.encoder, num_class, self.cfg)
        model.cuda()

        # optimizer and scheduler
        params = [{"params": [value]} for key, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.cfg.SOLVER.BASE_LR, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.cfg.SOLVER.MAX_EPOCHS))
        loss_fn = build_loss_fn(self.cfg, num_class)


        if self.cfg.SOLVER.FP16:
            logging.getLogger("Using Mix Precision training")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        # data loader
        train_loader = get_train_loader(dataset.train, self.cfg)
        test_loader = get_test_loader(dataset.query + dataset.gallery, self.cfg)

        # train
        for epoch in range(self.cfg.SOLVER.MAX_EPOCHS):
            self.logger.info("Epoch[{}] lr={:.2e}"
                        .format(epoch, scheduler.get_lr()[0]))
            self.train_epoch(model, optimizer, loss_fn, train_loader, epoch)
            scheduler.step(epoch)

            # validation
            if epoch % 2 == 0 or epoch == self.cfg.SOLVER.MAX_EPOCHS - 1:
                cur_mAP = self.validate(model, test_loader, len(dataset.query))
                if cur_mAP > self.best_mAP:
                    self.best_mAP = cur_mAP
                    torch.save(self.encoder.state_dict(), os.path.join(self.cfg.OUTPUT_DIR, 'best.pth'))
        self.logger.info("best mAP: {:.1%}".format(self.best_mAP))

    def train_epoch(self, model, optimizer, loss_fn, train_loader, epoch):
        losses = AverageMeter()
        data_time = AverageMeter()
        model_time = AverageMeter()
        model.train()

        if epoch < self.cfg.SOLVER.FREEZE_EPOCHS:
            self.logger.info("freeze encoder training")
            self.freeze_encoder()
        elif epoch == self.cfg.SOLVER.FREEZE_EPOCHS:
            self.logger.info("open encoder training")
            self.open_encoder()

        start = time.time()
        data_start = time.time()
        for iteration, batch in enumerate(train_loader):
            input, target, _, _ = batch
            input = input.cuda()
            target = target.cuda()
            data_time.update(time.time() - data_start)

            model_start = time.time()
            optimizer.zero_grad()
            score, feat = model(input, target)
            feat = torch.nn.functional.normalize(feat, dim=1, p=2)

            if self.use_memory_bank:
                with torch.no_grad():
                    self._momentum_update_ema_encoder()
                    feat_ema = self.encoder_ema(input)
                    feat_ema = torch.nn.functional.normalize(feat_ema, dim=1, p=2)
                    self.memory_bank.enqueue_dequeue(feat_ema, target.detach())

                memory_feat, memory_target = self.memory_bank.get()
                loss = loss_fn(score, feat, target, memory_feat, memory_target)
            else:
                loss = loss_fn(score, feat, target, feat, target)
            # loss = loss_fn(score, feat, target)

            if self.cfg.SOLVER.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            losses.update(loss.item()), input.size(0)
            model_time.update(time.time() - model_start)
            data_start = time.time()

            if iteration % 100 == 0:
                self.logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, data time: {:.3f}s, model time: {:.3f}s"
                        .format(epoch, iteration, len(train_loader),
                                losses.val, data_time.val, model_time.val))
        end = time.time()
        self.logger.info("epoch takes {:.3f}s".format((end - start)))

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
                pids.extend(pid)
                camids.extend(camid)

        feats = torch.cat(feats, dim=0)
        feats = F.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:num_query, ]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats[num_query:, ]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])

        # distmat = 2 - 2 * torch.mm(qf, gf.t()) # Euclidean Distance
        distmat = - torch.mm(qf, gf.t())# Cosine Distance
        indices = distmat.argsort(dim=1)
        indices = indices.cpu().numpy()

        cmc, mAP = eval_func(indices, q_pids, g_pids, q_camids, g_camids)
        self.logger.info("Validation Results")
        self.logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        return mAP

    def extract_feature(self, loader):
        self.encoder.eval()
        feats = []
        with torch.no_grad():
            for batch in loader:
                data, pid, camid, img_path = batch
                data = data.cuda()
                feat = self.encoder(data)
                feats.append(feat)
        feats = torch.cat(feats, dim=0)
        feats = F.normalize(feats, dim=1, p=2)
        return feats

    def freeze_encoder(self):
        for name, module in self.encoder.named_children():
            if 'base' in name:
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def open_encoder(self):
        for name, module in self.encoder.named_children():
            if 'base' in name:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True

    @torch.no_grad()
    def _momentum_update_ema_encoder(self):
        """
        Momentum update of the key encoder
        """
        m = 0.999
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_ema.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
