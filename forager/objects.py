import numpy as np
import forager.grid as grid

from abc import abstractmethod
from typing import Dict
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Coords, RawRGB, Size

class ForagerObject:
    def __init__(self, name: str):
        self.name = name

        self.collectable: bool = True
        self.blocking: bool = False
        self.location: Coords | None = None
        self.color: RawRGB | None = None

    def get_location(self, state: Coords, size: Size, objects: Dict[Coords, str], rng: np.random.Generator) -> Coords:
        if self.location is not None:
            return self.location

        return grid.sample_unpopulated(rng, size, objects, exclusions={state})

    def regen_delay(self, rng: np.random.Generator, clock: int) -> int | None:
        return rng.integers(10, 100)

    @abstractmethod
    def reward(self, rng: np.random.Generator, clock: int) -> float:
        raise NotImplementedError()


class Wall(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0

class Flower(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='flower')

        self.blocking = False
        self.collectable = True
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 1

class Thorns(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='thorns')

        self.blocking = False
        self.collectable = True
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return -1
