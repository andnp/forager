import numba as nb
import functools as ft
import typing as typ

njit = ft.partial(nb.njit, cache=True, fastmath=True, nogil=True)

# TODO: pylance and mypy aren't aware of these types
# hiding them here to make tooling behave better.
# Could consider my own type-stubs later
Dict: typ.Any = nb.typed.Dict
List: typ.Any = nb.typed.List
