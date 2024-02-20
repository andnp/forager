import forager._utils.numba as nbu
import forager.grid as grid
import forager.ObjectStorage as obj_store

from forager.config import ForagerConfig, load_config, sanity_check
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Action, Coords
from forager.objects import ForagerObject
from forager.observations import get_object_vision, get_color_vision, get_world_vision
from forager.state import ForagerState


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
        self._s = ForagerState(config)

    def start(self):
        return self._get_observation()

    def step(self, action: Action):
        n = grid.step(self._s.agent_state, self._s.size, action)
        idx = nbu.ravel(n, self._s.size)

        r = 0.
        if obj_store.has_object(self._s, idx):
            obj = obj_store.get_object(self._s, idx)
            r = obj.collision(self._s)

            if obj.blocking:
                n = self._s.agent_state

            if obj.collectable:
                self.remove_object(n)

        self._s.agent_state = n
        self._s.clock += 1

        obs = self._get_observation()
        self._respawn()
        return (obs, float(r))

    def add_object(self, obj: ForagerObject):
        obj_store.add_object(self._s, obj)

    def generate_objects(self, freq: float, name: str):
        size = self._s.size[0] * self._s.size[1]
        obj_store.add_n_deferred_objects(self._s, name, int(size * freq))

    def remove_object(self, coords: Coords):
        idx = nbu.ravel(coords, self._s.size)
        obj = obj_store.remove_object(self._s, idx)

        delta = obj.regen_delay(self._s)

        if delta is not None:
            self._s.to_respawn[self._s.clock + delta].append(obj)

    def _respawn(self):
        if self._s.clock not in self._s.to_respawn:
            return

        for obj in self._s.to_respawn[self._s.clock]:
            obj_store.add_object(self._s, obj)

        del self._s.to_respawn[self._s.clock]

    def _get_observation(self):
        if self._c.observation_mode == 'objects':
            return get_object_vision(self._s)
        elif self._c.observation_mode == 'colors':
            return get_color_vision(self._s)
        elif self._c.observation_mode == 'world':
            return get_world_vision(self._s)
        else:
            raise Exception()

    def __getstate__(self):
        # TODO: this should return a minimum necessary state to restart the env
        # should avoid cached and precomputed values that are only for optimization
        return {}

    def __setstate__(self, state):
        ...
