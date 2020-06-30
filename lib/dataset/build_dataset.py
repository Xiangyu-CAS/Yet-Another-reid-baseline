from .personx import PersonX
from .visda20 import VisDA20

factory = {
    'personx': PersonX,
    'visda20': VisDA20,
    }


def get_names():
    return factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)
