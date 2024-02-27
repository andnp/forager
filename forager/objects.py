import numpy as np

from abc import abstractmethod
from forager.interface import Coords, RawRGB

class ForagerObject:
    def __init__(self, name: str):
        self.name = name

        self.collectable: bool = True
        self.blocking: bool = False
        self.target_location: Coords | None = None
        self.current_location: Coords | None = None
        self.color: RawRGB | None = None

        self.last_collision = 0

    def regen_delay(self, rng: np.random.Generator, clock: int) -> int | None:
        return rng.integers(10, 100)

    def collision(self, rng: np.random.Generator, clock: int) -> float:
        r = self.reward(rng, clock)
        self.last_collision = clock
        return r

    @abstractmethod
    def reward(self, rng: np.random.Generator, clock: int) -> float:
        raise NotImplementedError()


class Wall(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.target_location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0

class Flower(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='flower')

        self.blocking = False
        self.collectable = True
        self.target_location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 1

class Thorns(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='thorns')

        self.blocking = False
        self.collectable = True
        self.target_location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return -1
