from __future__ import annotations
from abc import abstractmethod
from forager.interface import Coords, RawRGB

# we have some unfortunate circular imports here
# this ensures that the cycle only exists at type-checking
# time instead of at runtime.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from forager.state import ForagerState

class ForagerObject:
    def __init__(self, name: str):
        self.name = name

        self.collectable: bool = True
        self.blocking: bool = False
        self.location: Coords | None = None
        self.color: RawRGB | None = None

        self.last_collision = 0

    def regen_delay(self, state: ForagerState) -> int | None:
        return state.rng.integers(10, 100)

    def collision(self, state: ForagerState) -> float:
        r = self.reward(state)
        self.last_collision = state.clock
        return r

    @abstractmethod
    def reward(self, state: ForagerState) -> float:
        raise NotImplementedError()


class Wall(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='wall')

        self.blocking = True
        self.collectable = False
        self.location = loc

    def reward(self, state: ForagerState) -> float:
        return 0

class Flower(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='flower')

        self.blocking = False
        self.collectable = True
        self.location = loc

    def reward(self, state: ForagerState) -> float:
        return 1

class Thorns(ForagerObject):
    def __init__(self, loc: Coords | None = None):
        super().__init__(name='thorns')

        self.blocking = False
        self.collectable = True
        self.location = loc

    def reward(self, state: ForagerState) -> float:
        return -1
