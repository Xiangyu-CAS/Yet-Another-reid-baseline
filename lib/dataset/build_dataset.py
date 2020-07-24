from .personx import PersonX
from .visda20 import VisDA20
from .visda20_pseudo import VisDA20Pseudo
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmc import DukeMTMCreID
from .visda20_testhalf import Visda20_testhalf

factory = {
    'personx': PersonX,
    'visda20': VisDA20,
    'visda20_pseudo': VisDA20Pseudo,
    'market1501': Market1501,
    'msmt17': MSMT17,
    'dukemtmc': DukeMTMCreID,
    'visda20_testhalf': Visda20_testhalf,
    }


def get_names():
    return factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)
