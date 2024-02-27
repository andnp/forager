import numpy as np
import numba as nb

import forager.grid as grid
import forager._utils.numba as nbu

from typing import Dict
from forager.colors import Palette
from forager.config import ForagerObject, ObjectFactory
from forager.interface import Size
from forager.logger import logger

class ObjectStorage:
    def __init__(self, size: Size, obj_factories: Dict[str, ObjectFactory], palette: Palette, rng: np.random.Generator):
        self.rng = rng
        self.size = size
        self.factories = obj_factories

        self.idx_to_name = nbu.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.typeof(''),
        )
        self.name_to_color = nbu.Dict.empty(
            key_type=nb.typeof(''),
            value_type=nb.types.uint8[:],
        )

        self._idx_to_config: Dict[int, ForagerObject] = {}
        self._colors = palette

    def add_object(self, obj: ForagerObject):
        coords = obj.target_location
        if coords is None:
            coords = grid.sample_unpopulated(self.rng, self.size, self.idx_to_name)

        obj.current_location = coords
        idx = nbu.ravel(coords, self.size)

        if idx in self.idx_to_name:
            prior = self.idx_to_name[idx]
            logger.warning(f'Object already found at {coords}: {prior}. Replacing with {obj.name}')

        self.idx_to_name[idx] = obj.name
        self._idx_to_config[idx] = obj

        color = obj.color
        if color is None:
            color = self._colors.generate(obj.name)

        self.name_to_color[obj.name] = color

    def add_deferred_object(self, name: str):
        if name not in self.name_to_color:
            obj = self.factories[name]()
            return self.add_object(obj)

        coords = grid.sample_unpopulated(self.rng, self.size, self.idx_to_name)
        idx = nbu.ravel(coords, self.size)

        if idx in self.idx_to_name:
            prior = self.idx_to_name[idx]
            logger.warning(f'Object already found at {coords}: {prior}. Replacing with {name}')

        self.idx_to_name[idx] = name

    def add_n_deferred_objects(self, name: str, n: int):
        # first add a single object "manually" so that caches can be built correctly
        self.add_deferred_object(name)

        # then add n-1 objects quickly without worrying about cache states
        collisions = _add_many(self.rng, self.size, name, n - 1, self.idx_to_name)

        if collisions > 0:
            logger.warning(f'Encountered {collisions} collisions while generating objects of type: {name}')

    def get_object(self, idx: int):
        if idx in self._idx_to_config:
            return self._idx_to_config[idx]

        name = self.idx_to_name[idx]
        obj = self.factories[name]()
        self._idx_to_config[idx] = obj

        coords = nbu.unravel(idx, self.size)
        obj.current_location = coords

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
