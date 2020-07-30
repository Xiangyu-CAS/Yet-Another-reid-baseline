import argparse
import os
import sys
import torch

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example if you set "
                      "SOLVER.FP16=True")

from torch.backends import cudnn

sys.path.append('.')
from lib.config import _C as cfg
from lib.utils import setup_logger, load_checkpoint
from lib.train_net import Trainer
from lib.dataset.build_dataset import prepare_multiple_dataset


def naive_train(cfg, logger, distributed, local_rank):
    trainer = Trainer(cfg, distributed, local_rank)
    dataset = prepare_multiple_dataset(cfg, logger)
    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        load_checkpoint(trainer.encoder, cfg.MODEL.PRETRAIN_PATH)
    trainer.do_train(dataset)
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
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
    cudnn.benchmark = True

    distributed = int(os.environ['WORLD_SIZE']) > 1 if 'WORLD_SIZE' in os.environ else False
    if distributed:
        gpu = args.local_rank
        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    naive_train(cfg, logger, distributed, args.local_rank)


if __name__ == '__main__':
    main()