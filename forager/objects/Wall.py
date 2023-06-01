import numpy as np
from forager.config import ForagerObject

class Wall(ForagerObject):
    def __init__(self):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.freq = 0.3

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 0
