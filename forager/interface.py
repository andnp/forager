import numpy as np
from abc import abstractmethod
from typing import Tuple

Coords = Tuple[int, int]
Size = Tuple[int, int]
RawRGB = Tuple[float, float, float]

Action = int

class ForagerObject:
    def __init__(self, name: str):
        self.name = name

        self.freq: float | None = None
        self.collectable: bool = True
        self.blocking: bool = False
        self.location: Coords | None = None
        self.color: RawRGB | None = None

    def regen_delay(self, rng: np.random.Generator, clock: int) -> int | None:
        return rng.integers(10, 100)

    @abstractmethod
    def reward(self, rng: np.random.Generator, clock: int) -> float:
        raise NotImplementedError()
