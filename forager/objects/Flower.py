import numpy as np
from forager.config import ForagerObject

class Flower(ForagerObject):
    def __init__(self):
        super().__init__(name='flower')

        self.blocking = False
        self.collectable = True
        self.freq = 0.1

    def reward(self, rng: np.random.Generator, clock: int) -> float:
        return 1
