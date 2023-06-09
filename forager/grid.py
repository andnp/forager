import numpy as np
import forager._utils.numba as nbu

from typing import Any, Dict
from forager.exceptions import ForagerInvalidAction
from forager.interface import Action, Coords, Size

@nbu.njit
def step(state: Coords, size: Coords, action: Action) -> Coords:
    if action == 0:
        n = up(state, size)
    elif action == 1:
        n = right(state, size)
    elif action == 2:
        n = down(state, size)
    elif action == 3:
        n = left(state, size)
    else:
        raise ForagerInvalidAction()

    return n

@nbu.njit(inline='always')
def up(c: Coords, s: Coords) -> Coords:
    return (
        c[0],
        min(c[1] + 1, s[1]),
    )

@nbu.njit(inline='always')
def down(c: Coords, s: Coords) -> Coords:
    return (
        c[0],
        max(c[1] - 1, 0),
    )

@nbu.njit(inline='always')
def left(c: Coords, s: Coords) -> Coords:
    return (
        max(c[0] - 1, 0),
        c[1],
    )

@nbu.njit(inline='always')
def right(c: Coords, s: Coords) -> Coords:
    return (
        min(c[0] + 1, s[0]),
        c[1],
    )

@nbu.njit
def sample_unpopulated(rng: np.random.Generator, size: Size, objs: Dict[int, Any]):
    c = (0, 0)
    total = size[0] * size[1]
    for _ in range(10):
        idx = rng.integers(0, total)
        c = nbu.unravel(idx, size)
        if idx not in objs:
            return c

    return c
