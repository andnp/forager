import numpy as np
from forager.config import ForagerObject
from forager.interface import Coords

class WallSpawn(ForagerObject):
    def __init__(self):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.freq = 0.2

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0

class Wall(ForagerObject):
    def __init__(self, loc: Coords):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.location = loc

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0
