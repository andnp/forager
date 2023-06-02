import numpy as np
import forager._utils.numba as nbu

from typing import Any, Dict, Set
from forager.exceptions import ForagerInvalidAction
from forager.interface import Action, Coords, Size
from forager.logger import logger

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

_has_logged = False
def sample_unpopulated(rng: np.random.Generator, size: Size, objs: Dict[Coords, Any], exclusions: Set[Coords] = set()):
    global _has_logged

    i = 0
    c = None
    for i in range(10):
        x = rng.integers(0, size[0])
        y = rng.integers(0, size[1])

        c = (x, y)
        if c not in objs and c not in exclusions:
            return c

    if i > 5 and not _has_logged:
        _has_logged = True
        logger.warn('Running into many collisions finding empty places for objects!')

    assert c is not None, "Impossible code path"
    return c
