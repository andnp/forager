import numpy as np
import numba as nb
import forager._utils.numba as nbu
import forager._utils.config as cu
import forager.grid as grid

from collections import defaultdict
from typing import Any, Dict, List

from forager.config import ForagerConfig, ForagerObject, load_config, sanity_check
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Action, Coords, Size
from forager.logging import logger
from forager.observations import get_vision_by_name


class ForagerEnv:
    def __init__(self, config: ForagerConfig | None = None, *, config_path: str | None = None, seed: int | None = None):
        if config is None and config_path is None:
            raise ForagerInvalidConfigException('Expected either {config} or {config_path} to be specified')

        # load configuration if necessary
        if config is None:
            assert config_path is not None, 'Should be impossible.'
            config = load_config(config_path)

        # parse configuration
        self._c = sanity_check(config)
        self._size: Size = cu.to_tuple(config.size)
        self._ap_size: Size = cu.to_tuple(config.aperture)

        _names = set([
            obj.name for obj in config.objects
        ])
        _names |= {'border'}
        self._names: List[str] = nbu.List(sorted(_names))

        # build object storage
        self._init_object_store()
        self._object_configs: Dict[Coords, ForagerObject] = {}
        self._to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # build state information
        self.rng = np.random.default_rng(seed)
        self._clock = 0
        self._state = (
            int(self._size[0] // 2),
            int(self._size[1] // 2),
        )
        self._populate()

    def step(self, action: Action):
        n = grid.step(self._state, self._size, action)

        r = 0
        if n in self._object_configs:
            obj = self._object_configs[n]
            r = obj.reward(self.rng, self._clock)

            if obj.blocking:
                n = self._state

            if obj.collectable:
                del self._object_configs[n]
                del self._object_names[n]

                delta = obj.regen_delay(self.rng, self._clock)
                self._to_respawn[self._clock + delta].append(obj)

        obs = get_vision_by_name(n, self._size, self._ap_size, self._object_names, self._names)
        self._state = n
        self._clock += 1
        self._respawn()

        return (obs, r)

    def add_object(self, coords: Coords, obj: ForagerObject):
        if coords in self._object_names:
            prior = self._object_names[coords]
            logger.warn(f'Object already found at {coords}: {prior}. Replacing with {obj.name}')

        self._object_names[coords] = obj.name
        self._object_configs[coords] = obj

    def _respawn(self):
        if self._clock not in self._to_respawn:
            return

        for obj in self._to_respawn[self._clock]:
            if obj.location is not None:
                coords = obj.location
            else:
                coords = get_unpopulated(self._size, self._object_names, self.rng)

            self.add_object(coords, obj)

    def _populate(self):
        # first check for hard-coded locations
        for conf in self._c.objects:
            if conf.location is not None:
                self.add_object(conf.location, conf)

        # then add sampled items
        size = self._size[0] * self._size[1]

        for conf in self._c.objects:
            if conf.freq is None: continue

            for _ in range(int(size * conf.freq)):
                coords = get_unpopulated(self._size, self._object_names, self.rng)
                self.add_object(coords, conf)

    def _init_object_store(self):
        self._object_names = nbu.Dict.empty(
            key_type=nb.types.UniTuple(nb.types.int64, 2),
            value_type=nb.typeof(''),
        )

    def __getstate__(self):
        # TODO: this should return a minimum necessary state to restart the env
        # should avoid cached and precomputed values that are only for optimization
        return {}

    def __setstate__(self, state):
        self._init_object_store()

@nbu.njit
def get_unpopulated(size: Size, objs: Dict[Coords, Any], rng: np.random.Generator):
    for _ in range(50):
        x = rng.integers(0, size[0])
        y = rng.integers(0, size[1])

        c = (x, y)
        if c not in objs:
            return c

    raise Exception('Uh-oh! We could not find an open coordinate after 50 tries!')
