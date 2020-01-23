import random
from typing import Dict, Iterable, Mapping, TypeVar

from tqdm.auto import tqdm

K = TypeVar('K')
V = TypeVar('V')


def require(condition, message, exc=ValueError):
    if not condition:
        raise exc(message)


def coalesce(*args):
    """ Returns the first argument that is not None (like the SQL function) """
    return next((arg for arg in args if arg is not None), None)


def progbar(iterable=None, enable=True, total=None, desc=None, **kwargs):
    kwargs.setdefault('dynamic_ncols', True)
    return tqdm(iterable=iterable, disable=not enable, total=total, desc=desc, **kwargs)


def maybe_progbar(iterable, yes, total=None, desc=None, **kwargs):
    if not yes:
        return iterable
    return progbar(iterable, total=total, desc=desc, **kwargs)


def noop(*args, **kwargs):
    return None


def sample_dict_subset(dictionary, size):
    pairs_sample = random.sample(list(dictionary.items()), k=size)
    return dict(pairs_sample)


def shuffled(iterable):
    out = list(iterable)
    random.shuffle(out)
    return out


def invert_one_to_many(mapping: Mapping[K, Iterable[V]]) -> Dict[V, K]:
    return {
        value: key
        for key, iterable in mapping.items()
        for value in iterable
    }


class MappingComposition:  # this is a partial implementation
    def __init__(self, a2b, b2c):
        self.a2b = a2b
        self.b2c = b2c

    def __getitem__(self, key):
        return self.b2c[self.a2b[key]]

    def __contains__(self, key) -> bool:
        return key in self.a2b
