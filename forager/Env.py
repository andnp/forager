import json
import numba as nb
import numpy as np

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, NamedTuple

Sampler = Callable[[np.random.Generator], float]
Coords = Tuple[int, int]

njit = partial(nb.njit, cache=False, fastmath=True, nogil=True)

class ForagerObject:
    def __init__(self, name: str):
        self.name = name

        self.freq: float | None = None
        self.collectable: bool = True
        self.blocking: bool = False
        self.location: Coords | None = None
        self.color: Tuple[float, float, float] | None = None

    def regen_delay(self, rng: np.random.Generator) -> int:
        return rng.integers(10, 100)

    @abstractmethod
    def reward(self, rng: np.random.Generator) -> float:
        raise NotImplementedError()


@dataclass
class ForagerConfig:
    size: int | Tuple[int, int]
    objects: List[ForagerObject]

    colors: bool = False
    aperture: int | Tuple[int, int] = 3

# making this a namedtuple for numba interop
class InternalForagerObject(NamedTuple):
    name: str
    color: np.ndarray

class ForagerEnv:
    def __init__(self, config: ForagerConfig | None = None, *, config_path: str | None = None, seed: int | None = None):
        if config is None and config_path is None:
            raise InvalidConfigException('Expected either {config} or {config_path} to be specified')

        self.rng = np.random.default_rng(seed)

        # load configuration if necessary
        if config is None:
            assert config_path is not None, InvalidConfigException()

            with open(config_path, 'r') as f:
                d = json.load(f)

            config = ForagerConfig(**d)

        self._c = config

        # parse configuration
        self._size: Tuple[int, int] = to_tuple(config.size)
        self._ap_size: Coords = to_tuple(config.aperture)

        _names = set([
            obj.name for obj in config.objects
        ])
        _names |= {'border'}
        self._names = nb.typed.List(sorted(_names))

        # build object storage
        self._init_object_store()
        self._object_configs: Dict[Coords, ForagerObject] = {}
        self._to_respawn: Dict[int, List[ForagerObject]] = defaultdict(list)

        # build state information
        self._clock = 0
        self._state = (
            int(self._size[0] // 2),
            int(self._size[1] // 2),
        )
        self._populate()

    def step(self, action: int):
        self._respawn()

        if action == 0:
            n = up(self._state, self._size)
        elif action == 1:
            n = right(self._state, self._size)
        elif action == 2:
            n = down(self._state, self._size)
        elif action == 3:
            n = left(self._state, self._size)
        else:
            raise Exception()

        r = 0
        if n in self._object_configs:
            obj = self._object_configs[n]
            r = obj.reward(self.rng)

            if obj.blocking:
                n = self._state

            if obj.collectable:
                del self._object_configs[n]
                n_idx = ravel_idx(n, self._size)
                del self._objects[n_idx]

                delta = obj.regen_delay(self.rng)
                self._to_respawn[self._clock + delta].append(obj)

        obs = get_vision_by_name(n, self._size, self._ap_size, self._objects, self._names)
        self._state = n
        self._clock += 1

        return (obs, r)

    def add_object(self, coords: Coords, obj: ForagerObject):
        # internal = InternalForagerObject(
        #     name=obj.name,
        #     color=to_color(obj.color),
        # )

        idx = ravel_idx(coords, self._size)
        self._objects[idx] = obj.name
        self._object_configs[coords] = obj

    def _respawn(self):
        if self._clock not in self._to_respawn:
            return

        for obj in self._to_respawn[self._clock]:
            if obj.location is not None:
                coords = obj.location
            else:
                coords = get_unpopulated(self._size, self._objects, self.rng)

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
                coords = get_unpopulated(self._size, self._objects, self.rng)
                self.add_object(coords, conf)

    def _init_object_store(self):
        obj = self._fake_object()
        self._objects = nb.typed.Dict.empty(
            key_type=nb.typeof(int(0)),
            value_type=nb.typeof(''),
        )

    def _fake_object(self):
        return InternalForagerObject('test', np.zeros(3))

    def __getstate__(self):
        return {
            '_objects': dict(self._objects),
        }

    def __setstate__(self, state):
        self._init_object_store()

        for obj in state['_objects']:
            self._objects[obj] = state['_objects'][obj]

def to_tuple(i: int | Tuple[int, int]):
    if isinstance(i, int):
        return (i, i)
    return i

@njit
def empty_tuple():
    return (0, 0)

# TODO: make this a stateful color generator
def to_color(c: Tuple[float, float, float] | None):
    if c is None:
        return np.zeros(3)

    return np.asarray(c, dtype=np.float_)

@njit
def get_unpopulated(size: Coords, objs: Dict[int, Any], rng: np.random.Generator):
    for _ in range(20):
        x = rng.integers(0, size[0])
        y = rng.integers(0, size[1])

        c = (x, y)
        idx = ravel_idx(c, size)
        if idx not in objs:
            return c

    raise Exception('Uh-oh! We could not find an open coordinate after 20 tries!')

@njit
def get_vision_by_name(state: Coords, size: Coords, ap_size: Coords, objs: Dict[int, str], names: List[str]):
    dims = len(names)
    out = np.zeros((ap_size[0], ap_size[1], dims), dtype=np.bool_)

    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    mi_x = max(state[0] - ax - 1, 0)
    ma_x = min(state[0] + ax, size[0])

    mi_y = max(state[1] - ay - 1, 0)
    ma_y = min(state[1] + ay, size[1])

    for i, x in enumerate(range(mi_x, ma_x)):
        for j, y in enumerate(range(mi_y, ma_y)):
            c = (x, y)
            print(c)
            idx = ravel_idx(c, size)
            if idx in objs:
                obj = objs[idx]
                d = names.index(obj)
                print(d, obj)
                out[i, j, d] = 1

    return out


def up(c: Coords, s: Coords) -> Coords:
    return (
        c[0],
        min(c[1] + 1, s[1]),
    )

def down(c: Coords, s: Coords) -> Coords:
    return (
        c[0],
        max(c[1] - 1, 0),
    )

def left(c: Coords, s: Coords) -> Coords:
    return (
        max(c[0] - 1, 0),
        c[1],
    )

def right(c: Coords, s: Coords) -> Coords:
    return (
        min(c[0] + 1, s[0]),
        c[1],
    )

@njit(inline='always')
def ravel_idx(c: Coords, s: Coords) -> int:
    return c[0] + c[1] * s[0]

class InvalidConfigException(Exception): ...


class Wall(ForagerObject):
    def __init__(self):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.freq = 0.3

    def reward(self, _: np.random.Generator) -> float:
        return 0

class Flower(ForagerObject):
    def __init__(self):
        super().__init__(name='flower')

        self.blocking = False
        self.collectable = True
        self.freq = 0.1

    def reward(self, _: np.random.Generator) -> float:
        return 1

config = ForagerConfig(
    size=10,
    objects=[
        Flower(),
        Wall(),
    ],
)
env = ForagerEnv(config=config, seed=0)
print(env._state)
print(env._object_configs)
# print(env._objects)

x, r = env.step(2)
print(env._state)
print(x[:, :, 1])
# x, r = env.step(0)
