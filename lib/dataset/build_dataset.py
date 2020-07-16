from .personx import PersonX
from .visda20 import VisDA20
from .visda20_pseudo import VisDA20Pseudo

factory = {
    'personx': PersonX,
    'visda20': VisDA20,
    'visda20_pseudo': VisDA20Pseudo
    }


def get_names():
    return factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)
