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
        self._c = sanity_check(config)
        self._size: Size = cu.to_tuple(self._c.size)
        self._ap_size = cu.to_tuple(config.aperture) if config.aperture is not None else None

        self._colors = cu.not_none(self._c.colors)
        self.rng = np.random.default_rng(self._c.seed)

        # build object storage
        self._obj_store = ObjectStorage(self._size, self._c.object_types, self._colors, self.rng)
        self._to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # ensure object types have a consistent object dimension
        _names = set(self._c.object_types.keys())

        if config.observation_mode == 'world':
            _names |= {'agent'}

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

    def generate_objects_locations(self, freq: float, name: str, locations: List[Coords]):
        size = len(locations)
        self._obj_store.add_n_deferred_objects_locations(name, int(size * freq), locations)

    def render(self):
        out = np.ones((self._size[1], self._size[0], 3), dtype=np.uint8) * 255
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                c = (x, y)
                idx = nbu.ravel(c, self._size)
                if idx in  self._obj_store.idx_to_name:
                    name =  self._obj_store.idx_to_name[idx]
                    color = self._obj_store.name_to_color[name]
                    out[y, x] = color

        ax = int(self._ap_size[0] // 2)
        ay = int(self._ap_size[1] // 2)

        xs = range(self._state[0] - ax, self._state[0] + ax + 1)
        ys = range(self._state[1] - ay, self._state[1] + ay + 1)


        agent_color = np.array((0, 0, 255), dtype=np.uint8)
        out[self._state[1], self._state[0]] = agent_color
        alpha = 0.2

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                x = x % self._size[0]
                y = y % self._size[1]
                c = (x, y)
                idx = nbu.ravel(c, self._size)
                out[y, x] = (1 - alpha) * out[y, x] + alpha * agent_color

        return out

    def remove_object(self, coords: Coords):
        idx = nbu.ravel(coords, self._size)
        obj = self._obj_store.remove_object(idx)

        delta = obj.regen_delay(self.rng, self._clock)

        if delta is not None:
            self._to_respawn[self._clock + delta].append(obj)

    def get_observation_shape(self):
        if self._c.observation_mode == 'objects':
            return (self._ap_size[0], self._ap_size[1], len(self._names))
        elif self._c.observation_mode == 'colors':
            return (self._ap_size[0], self._ap_size[1], 3)
        elif self._c.observation_mode == 'world':
            return (self._size[0], self._size[1], len(self._names))

    def _respawn(self):
        if self._clock not in self._to_respawn:
            return

        for obj in self._to_respawn[self._clock]:
            self._obj_store.add_object(obj)

        del self._to_respawn[self._clock]

    def _get_observation(self, s: Coords):
        if self._c.observation_mode == 'objects':
            assert self._ap_size is not None, "Expected non-none aperture size when observation mode is 'objects'"
            return get_object_vision(s, self._size, self._ap_size, self._obj_store.idx_to_name, self._names_to_dims)
        elif self._c.observation_mode == 'colors':
            assert self._ap_size is not None, "Expected non-none aperture size when observation mode is 'colors'"
            return get_color_vision(s, self._size, self._ap_size, self._obj_store.idx_to_name, self._obj_store.name_to_color)
        elif self._c.observation_mode == 'world':
            return get_world_vision(s, self._size, self._obj_store.idx_to_name, self._names_to_dims)
        else:
            raise Exception()

    def __getstate__(self):
        return {
            '_c': self._c,
            '_size': self._size,
            '_ap_size': self._ap_size,
            '_colors': self._colors,
            '_to_respawn': self._to_respawn,
            '_names': list(self._names),
            '_names_to_dims': dict(self._names_to_dims),
            '_clock': self._clock,
            '_state': self._state,
            '_obj_store.idx_to_name': dict(self._obj_store.idx_to_name),
            '_obj_store.name_to_color': dict(self._obj_store.name_to_color),
            '_obj_store._idx_to_config': self._obj_store._idx_to_config,
            'rng': self.rng,
        }

    def __setstate__(self, state):
        self._c = state['_c']
        self._size = state['_size']
        self._ap_size = state['_ap_size']
        self._colors = state['_colors']
        self._to_respawn = state['_to_respawn']
        self._names = nbu.List(state['_names'])
        self._clock = state['_clock']
        self._state = state['_state']
        self.rng = state['rng']

        self._names_to_dims = nbu.Dict.empty(
            key_type=nb.typeof(''),
            value_type=nb.typeof(int(1)),
            n_keys=len(self._names),
        )
        for i, n in enumerate(state['_names_to_dims']):
            self._names_to_dims[n] = i

        # build object storage
        # as the object storage is numba dictionary, we need to rebuild it from the saved state
        self._obj_store = ObjectStorage(self._size, self._c.object_types, self._colors, self.rng)
        self._obj_store._idx_to_config.update(state['_obj_store._idx_to_config'])

        for key, val in state['_obj_store.idx_to_name'].items():
            self._obj_store.idx_to_name[key] = val

        for key, val in state['_obj_store.name_to_color'].items():
            self._obj_store.name_to_color[key] = val
