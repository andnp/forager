import numpy as np
from typing import Tuple

def to_tuple(i: int | Tuple[int, int]):
    if isinstance(i, int):
        return (i, i)
    return i

# TODO: make this a stateful color generator
def to_color(c: Tuple[float, float, float] | None):
    if c is None:
        return np.zeros(3)

    return np.asarray(c, dtype=np.float_)

def nearest_odd(i: int) -> int:
    if i % 2 == 0:
        return i + 1

    return i
