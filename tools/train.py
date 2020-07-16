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


def naive_train(cfg, logger):
    trainer = Trainer(cfg)
    dataset = BaseImageDataset()

    trainset = ['personx', 'visda20_pseudo']
    valset = ['personx']
    for element in trainset:
        cur_dataset = init_dataset(element, root=cfg.DATASETS.ROOT_DIR)
        dataset.train = merge_datasets([dataset.train, cur_dataset.train])
        dataset.relabel_train()
    for element in valset:
        cur_dataset = init_dataset(element, root=cfg.DATASETS.ROOT_DIR)
        dataset.query = merge_datasets([dataset.query, cur_dataset.query])
        dataset.gallery = merge_datasets([dataset.gallery, cur_dataset.gallery])

    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)

    load_checkpoint(trainer.encoder.base, cfg.MODEL.PRETRAIN_PATH)
    trainer.do_train(dataset)
    torch.cuda.empty_cache()




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
    naive_train(cfg, logger)


if __name__ == '__main__':
    main()