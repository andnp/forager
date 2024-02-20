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
from forager.observations import get_object_vision, get_color_vision, get_world_vision


class ForagerEnv:
    def __init__(self, config: ForagerConfig | None = None, *, config_path: str | None = None):
        if config is None and config_path is None:
            raise ForagerInvalidConfigException('Expected either {config} or {config_path} to be specified')

        # load configuration if necessary
        if config is None:
            assert config_path is not None, 'Should be impossible.'
            config = load_config(config_path)

        # parse configuration
        config = sanity_check(config)
        self._c = config
        self._s = _ForagerState(config)

    def start(self):
        return self._get_observation(self._s.agent_state)

    def step(self, action: Action):
        n = grid.step(self._s.agent_state, self._s.size, action)
        idx = nbu.ravel(n, self._s.size)

        r = 0.
        if self._s.objects.has_object(idx):
            obj = self._s.objects.get_object(idx)
            r = obj.collision(self._s.rng, self._s.clock)

            if obj.blocking:
                n = self._s.agent_state

            if obj.collectable:
                self.remove_object(n)

        obs = self._get_observation(n)

        self._s.agent_state = n
        self._s.clock += 1
        self._respawn()

        return (obs, float(r))

    def add_object(self, obj: ForagerObject):
        self._s.objects.add_object(obj)

    def generate_objects(self, freq: float, name: str):
        size = self._s.size[0] * self._s.size[1]
        self._s.objects.add_n_deferred_objects(name, int(size * freq))

    def remove_object(self, coords: Coords):
        idx = nbu.ravel(coords, self._s.size)
        obj = self._s.objects.remove_object(idx)

        delta = obj.regen_delay(self._s.rng, self._s.clock)

        if delta is not None:
            self._s.to_respawn[self._s.clock + delta].append(obj)

    def _respawn(self):
        if self._s.clock not in self._s.to_respawn:
            return

        for obj in self._s.to_respawn[self._s.clock]:
            self._s.objects.add_object(obj)

        del self._s.to_respawn[self._s.clock]

    def _get_observation(self, s: Coords):
        if self._c.observation_mode == 'objects':
            assert self._s.ap_size is not None, "Expected non-none aperture size when observation mode is 'objects'"
            return get_object_vision(s, self._s.size, self._s.ap_size, self._s.objects.idx_to_name, self._s.names_to_dims)
        elif self._c.observation_mode == 'colors':
            assert self._s.ap_size is not None, "Expected non-none aperture size when observation mode is 'colors'"
            return get_color_vision(s, self._s.size, self._s.ap_size, self._s.objects.idx_to_name, self._s.objects.name_to_color)
        elif self._c.observation_mode == 'world':
            return get_world_vision(s, self._s.size, self._s.objects.idx_to_name, self._s.names_to_dims)
        else:
            raise Exception()

    def __getstate__(self):
        # TODO: this should return a minimum necessary state to restart the env
        # should avoid cached and precomputed values that are only for optimization
        return {}

    def __setstate__(self, state):
        ...


class _ForagerState:
    def __init__(
        self,
        config: ForagerConfig,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # parse config
        self.size: Size = cu.to_tuple(config.size)
        self.ap_size = cu.to_tuple(config.aperture) if config.aperture is not None else None

        # build object storage
        self.colors = cu.not_none(config.colors)
        self.objects = ObjectStorage(self.size, config.object_types, self.colors, self.rng)

        # build respawn timers
        self.to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # ensure object types have a consistent object dimension
        _names = set(config.object_types.keys())

        if config.observation_mode == 'world':
            _names |= {'agent'}

        self.names = nbu.List(sorted(_names))
        self.names_to_dims = nbu.Dict.empty(
            key_type=nb.typeof(''),
            value_type=nb.typeof(int(1)),
            n_keys=len(_names),
        )

        for i, n in enumerate(self.names):
            self.names_to_dims[n] = i

        # build state information
        self.clock = 0
        self.agent_state = (
            int(self.size[0] // 2),
            int(self.size[1] // 2),
        )
