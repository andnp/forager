import numpy as np
from forager.config import ForagerObject
from forager.exceptions import ForagerInvalidConfigException
from forager.interface import Coords

class Wall(ForagerObject):
    def __init__(self, freq: float | None = 0.2, *, loc: Coords | None = None):
        super().__init__(name='wall')

        if loc is not None:
            freq = None

        if freq is None and loc is None:
            raise ForagerInvalidConfigException()

        self.blocking = True
        self.collectable = False
        self.freq = freq
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0

class Flower(ForagerObject):
    def __init__(self, freq: float | None = 0.1, *, loc: Coords | None = None):
        super().__init__(name='flower')

        if loc is not None:
            freq = None

        if freq is not None and loc is not None:
            raise ForagerInvalidConfigException()

        self.blocking = False
        self.collectable = True
        self.freq = freq
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 1

class Thorns(ForagerObject):
    def __init__(self, freq: float | None = 0.1, *, loc: Coords | None = None):
        super().__init__(name='thorns')

        if loc is not None:
            freq = None

        if freq is not None and loc is not None:
            raise ForagerInvalidConfigException()

        self.blocking = False
        self.collectable = True
        self.freq = freq
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return -1
