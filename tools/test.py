import argparse
import os
import sys
import torch
import numpy as np

from torch.backends import cudnn

sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger, load_checkpoint
from lib.dataset.build_dataset import init_dataset
from lib.dataset.base import merge_datasets, BaseImageDataset
from lib.modeling.build_model import Encoder
from lib.dataset.data_loader import get_test_loader
from lib.post_processing import post_processor


def inference(cfg, logger):
    testset = cfg.DATASETS.TEST[0]
    dataset = init_dataset(testset, root=cfg.DATASETS.ROOT_DIR)
    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)
    test_loader = get_test_loader(dataset.query + dataset.gallery, cfg)

    model = Encoder(cfg.MODEL.BACKBONE, cfg.MODEL.PRETRAIN_PATH,
                    cfg.MODEL.PRETRAIN_CHOICE).cuda()

    logger.info("loading model from {}".format(cfg.TEST.WEIGHT))
    load_checkpoint(model, cfg.TEST.WEIGHT)

    feats, pids, camids, img_paths = [], [], [], []
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            if cfg.TEST.FLIP_TEST:
                data_flip = data.flip(dims=[3])  # NCHW
                feat_flip = model(data_flip)
                feat = (feat + feat_flip) / 2
            feats.append(feat)
            pids.extend(pid)
            camids.extend(camid)
            img_paths.extend(img_path)
    del model
    torch.cuda.empty_cache()

    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)

    return [feats, pids, camids, img_paths], len(dataset.query)


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info(args)
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    batch, num_query = inference(cfg, logger)
    cmc, mAP, indices_np = post_processor(cfg, batch, num_query)

    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


if __name__ == '__main__':
    main()