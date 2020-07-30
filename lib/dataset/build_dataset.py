from .base import BaseImageDataset, merge_datasets
from .personx import PersonX
from .visda20 import VisDA20
from .visda20_pseudo import VisDA20Pseudo
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmc import DukeMTMCreID
from .poly import Poly
from .synthetic import Synthetic
from .rydata import Rydata
from .rytest import RYTest
from .lpw import LPW

factory = {
    'personx': PersonX,
    'visda20': VisDA20,
    'visda20_pseudo': VisDA20Pseudo,
    'market1501': Market1501,
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMCreID,
    'poly': Poly,
    'rydata': Rydata,
    'synthetic': Synthetic,
    'lpw': LPW,
    'rytest': RYTest,
    }


def get_names():
    return factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)


def prepare_multiple_dataset(cfg, logger):
    dataset = BaseImageDataset()
    trainset = cfg.DATASETS.TRAIN
    valset = cfg.DATASETS.TEST
    for element in trainset:
        cur_dataset = init_dataset(element, root=cfg.DATASETS.ROOT_DIR)
        if cfg.DATASETS.COMBINEALL:
            dataset.train = merge_datasets([dataset.train, cur_dataset.train,
                                            cur_dataset.query + cur_dataset.gallery])
        else:
            dataset.train = merge_datasets([dataset.train, cur_dataset.train])
        dataset.relabel_train()

    # TODO: support multi-joint testset
    cur_dataset = init_dataset(valset[0], root=cfg.DATASETS.ROOT_DIR)
    dataset.query = cur_dataset.query
    dataset.gallery = cur_dataset.gallery

    dataset.print_dataset_statistics(dataset.train, dataset.query, dataset.gallery, logger)
    return dataset

