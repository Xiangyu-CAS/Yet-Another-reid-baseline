import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger, load_checkpoint
from lib.train_net import Trainer
from lib.dataset.build_dataset import init_dataset
from lib.dataset.base import merge_datasets, BaseImageDataset
from lib.dataset.data_loader import get_test_loader
from lib.cluster import DBSCAN_cluster


def iterative_dbscan(cfg, logger):
    trainer = Trainer(cfg)
    src_dataset = init_dataset('personx', root=cfg.DATASETS.ROOT_DIR)
    target_dataset = init_dataset('visda20', root=cfg.DATASETS.ROOT_DIR)

    dataset = BaseImageDataset()
    dataset.query = src_dataset.query
    dataset.gallery = src_dataset.gallery

    iteration = 4
    pseudo_label_dataset = []
    for i in range(iteration):
        dataset.train = merge_datasets([src_dataset.train, pseudo_label_dataset])
        dataset.relabel_train()
        dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)

        # from scratch
        load_checkpoint(trainer.encoder.base, cfg.MODEL.PRETRAIN_PATH)
        #if os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'best.pth')):
        #    load_checkpoint(trainer.encoder, os.path.join(cfg.OUTPUT_DIR, 'best.pth'))
        trainer.do_train(dataset)
        torch.cuda.empty_cache()
        test_loader = get_test_loader(target_dataset.train, cfg)
        feats = trainer.extract_feature(test_loader)
        pseudo_label_dataset = DBSCAN_cluster(feats, target_dataset.train, logger)


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
    iterative_dbscan(cfg, logger)


if __name__ == '__main__':
    main()