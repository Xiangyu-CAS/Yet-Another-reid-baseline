import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger
from lib.train_net import Trainer
from lib.dataset.build_dataset import init_dataset

def iterative_dbscan(cfg):
    trainer = Trainer(cfg)
    src_dataset = init_dataset('personx', root=cfg.DATASETS.ROOT_DIR)
    trainer.do_train(src_dataset)


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
    iterative_dbscan(cfg)


if __name__ == '__main__':
    main()