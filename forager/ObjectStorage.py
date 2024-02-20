import forager.grid as grid
import forager._utils.numba as nbu

from typing import Dict
from forager.config import ForagerObject
from forager.interface import Size
from forager.logger import logger
from forager.state import ForagerState

def add_object(s: ForagerState, obj: ForagerObject):
    coords = obj.location
    if coords is None:
        coords = grid.sample_unpopulated(s.rng, s.size, s.objects.idx_to_name)

    idx = nbu.ravel(coords, s.size)

    if idx in s.objects.idx_to_name:
        prior = s.objects.idx_to_name[idx]
        logger.warning(f'Object already found at {coords}: {prior}. Replacing with {obj.name}')

    s.objects.idx_to_name[idx] = obj.name
    s.objects.idx_to_object[idx] = obj

    color = obj.color
    if color is None:
        color = s.colors.generate(obj.name)

    s.objects.name_to_color[obj.name] = color

def add_deferred_object(s: ForagerState, name: str):
    if name not in s.objects.name_to_color:
        obj = s.config.object_types[name]()
        return add_object(s, obj)

    coords = grid.sample_unpopulated(s.rng, s.size, s.objects.idx_to_name)
    idx = nbu.ravel(coords, s.size)

    if idx in s.objects.idx_to_name:
        prior = s.objects.idx_to_name[idx]
        logger.warning(f'Object already found at {coords}: {prior}. Replacing with {name}')

    s.objects.idx_to_name[idx] = name

def add_n_deferred_objects(s: ForagerState, name: str, n: int):
    # first add a single object "manually" so that caches can be built correctly
    add_deferred_object(s, name)

    # then add n-1 objects quickly without worrying about cache states
    collisions = _add_many(s.rng, s.size, name, n - 1, s.objects.idx_to_name)

    if collisions > 0:
        logger.warning(f'Encountered {collisions} collisions while generating objects of type: {name}')

def get_object(s: ForagerState, idx: int):
    # if object is already initialized
    obj = s.objects.idx_to_object.get(idx, None)
    if obj is not None:
        return obj

    name = s.objects.idx_to_name[idx]
    obj = s.config.object_types[name]()
    s.objects.idx_to_object[idx] = obj

    return obj

def has_object(s: ForagerState, idx: int):
    return idx in s.objects.idx_to_name

def remove_object(s: ForagerState, idx: int):
    obj = get_object(s, idx)

    del s.objects.idx_to_name[idx]
    del s.objects.idx_to_object[idx]

    return obj


@nbu.njit(nogil=False)
def _add_many(rng, size: Size, name: str, n: int, store: Dict[int, str]):
    count = 0
    for _ in range(n):
        coords = grid.sample_unpopulated(rng, size, store)
        idx = nbu.ravel(coords, size)

        if idx in store:
            count += 1

        store[idx] = name

    return count
