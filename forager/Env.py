import numpy as np
import numba as nb
import forager._utils.numba as nbu
import forager._utils.config as cu
import forager.grid as grid

from collections import defaultdict
from typing import Dict, List

from forager.config import ForagerConfig, load_config, sanity_check
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Action, Coords, Size
from forager.objects import ForagerObject
from forager.ObjectStorage import ObjectStorage
from forager.observations import get_object_vision, get_color_vision


class ForagerEnv:
    def __init__(self, config: ForagerConfig | None = None, *, config_path: str | None = None):
        if config is None and config_path is None:
            raise ForagerInvalidConfigException('Expected either {config} or {config_path} to be specified')

        # load configuration if necessary
        if config is None:
            assert config_path is not None, 'Should be impossible.'
            config = load_config(config_path)


        # parse configuration
        self._c = sanity_check(config)
        self._size: Size = cu.to_tuple(self._c.size)
        self._ap_size: Size = cu.to_tuple(self._c.aperture)
        self._colors = cu.not_none(self._c.colors)
        self.rng = np.random.default_rng(self._c.seed)

        # build object storage
        self._obj_store = ObjectStorage(self._size, self._c.object_types, self._colors, self.rng)
        self._to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # ensure object types have a consistent object dimension
        _names = set(self._c.object_types.keys())

        self._names = nbu.List(sorted(_names))
        self._names_to_dims = nbu.Dict.empty(
            key_type=nb.typeof(''),
            value_type=nb.typeof(int(1)),
            n_keys=len(_names),
        )

        for i, n in enumerate(self._names):
            self._names_to_dims[n] = i

        # build state information
        self._clock = 0
        self._state = (
            int(self._size[0] // 2),
            int(self._size[1] // 2),
        )

    def start(self):
        return self._get_observation(self._state)

    def step(self, action: Action):
        n = grid.step(self._state, self._size, action)
        idx = nbu.ravel(n, self._size)

        r = 0.
        if self._obj_store.has_object(idx):
            obj = self._obj_store.get_object(idx)
            r = obj.collision(self.rng, self._clock)

            if obj.blocking:
                n = self._state

            if obj.collectable:
                self.remove_object(n)

        obs = self._get_observation(n)

        self._state = n
        self._clock += 1
        self._respawn()

        return (obs, float(r))

    def add_object(self, obj: ForagerObject):
        self._obj_store.add_object(obj)

    def generate_objects(self, freq: float, name: str):
        size = self._size[0] * self._size[1]
        self._obj_store.add_n_deferred_objects(name, int(size * freq))

    def remove_object(self, coords: Coords):
        idx = nbu.ravel(coords, self._size)
        obj = self._obj_store.remove_object(idx)

        delta = obj.regen_delay(self.rng, self._clock)

        if delta is not None:
            self._to_respawn[self._clock + delta].append(obj)

    def _respawn(self):
        if self._clock not in self._to_respawn:
            return

        for obj in self._to_respawn[self._clock]:
            self._obj_store.add_object(obj)

        del self._to_respawn[self._clock]

    def _get_observation(self, s: Coords):
        if self._c.observation_mode == 'objects':
            return get_object_vision(s, self._size, self._ap_size, self._obj_store.idx_to_name, self._names_to_dims)
        elif self._c.observation_mode == 'colors':
            return get_color_vision(s, self._size, self._ap_size, self._obj_store.idx_to_name, self._obj_store.name_to_color)
        else:
            raise Exception()

    def __getstate__(self):
        # TODO: this should return a minimum necessary state to restart the env
        # should avoid cached and precomputed values that are only for optimization
        return {}

    def __setstate__(self, state):
        ...
