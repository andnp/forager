import numba as nb
import functools as ft
import typing as typ

njit = ft.partial(nb.njit, cache=True, fastmath=True, nogil=True)

# TODO: pylance and mypy aren't aware of these types
# hiding them here to make tooling behave better.
# Could consider my own type-stubs later
Dict: typ.Any = nb.typed.Dict
List: typ.Any = nb.typed.List


@njit(inline='always')
def ravel(coords: typ.Tuple[int, int], size: typ.Tuple[int, int]) -> int:
    return coords[0] + coords[1] * size[0]

@njit(inline='always')
def unravel(idx: int, size: typ.Tuple[int, int]) -> typ.Tuple[int, int]:
    x = int(idx % size[0])
    y = int(idx // size[0])

    return (x, y)
