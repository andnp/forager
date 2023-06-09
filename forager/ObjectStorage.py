import numpy as np
import numba as nb

import forager.grid as grid
import forager._utils.numba as nbu

from typing import Dict, Set
from forager.config import ForagerObject, ObjectFactory
from forager.interface import Coords, Size
from forager.logger import logger

class ObjectStorage:
    def __init__(self, size: Size, obj_factories: Dict[str, ObjectFactory],  rng: np.random.Generator):
        self.rng = rng
        self.size = size
        self.factories = obj_factories

        self.idx_to_name = nbu.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.typeof(''),
        )

        self._idx_to_config: Dict[int, ForagerObject] = {}

    def add_object(self, obj: ForagerObject):
        coords = obj.location
        if coords is None:
            coords = grid.sample_unpopulated(self.rng, self.size, self.idx_to_name)

        idx = nbu.ravel(coords, self.size)

        if idx in self.idx_to_name:
            prior = self.idx_to_name[idx]
            logger.warning(f'Object already found at {coords}: {prior}. Replacing with {obj.name}')

        self.idx_to_name[idx] = obj.name
        self._idx_to_config[idx] = obj

    def add_deferred_object(self, name: str):
        coords = grid.sample_unpopulated(self.rng, self.size, self.idx_to_name)
        idx = nbu.ravel(coords, self.size)

        if idx in self.idx_to_name:
            prior = self.idx_to_name[idx]
            logger.warning(f'Object already found at {coords}: {prior}. Replacing with {name}')

        self.idx_to_name[idx] = name

    def add_n_deferred_objects(self, name: str, n: int):
        collisions = _add_many(self.rng, self.size, name, n, self.idx_to_name)

        if collisions > 0:
            logger.warning(f'Encountered {collisions} collisions while generating objects of type: {name}')

    def get_object(self, idx: int):
        if idx in self._idx_to_config:
            return self._idx_to_config[idx]

        name = self.idx_to_name[idx]
        obj = self.factories[name]()
        self._idx_to_config[idx] = obj

        return obj

    def has_object(self, idx: int):
        return idx in self.idx_to_name

    def remove_object(self, idx: int):
        obj = self.get_object(idx)

        del self.idx_to_name[idx]
        del self._idx_to_config[idx]

        return obj

    def __len__(self):
        return len(self.idx_to_name)


@nbu.njit
def _add_many(rng, size: Size, name: str, n: int, store: Dict[int, str]):
    count = 0
    for _ in range(n):
        coords = grid.sample_unpopulated(rng, size, store)
        idx = nbu.ravel(coords, size)

        if idx in store:
            count += 1

        store[idx] = name

    return count
