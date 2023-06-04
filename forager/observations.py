import numpy as np
import forager._utils.numba as nbu

from typing import Any, Dict, List
from forager.interface import Coords, Size

@nbu.njit
def get_vision_by_name(state: Coords, size: Size, ap_size: Size, objs: Dict[Coords, str], names: List[str]):
    dims = len(names)
    out = np.zeros((ap_size[0], ap_size[1], dims), dtype=np.bool_)

    xs, ys = _bounds(state, size, ap_size)
    b_dim = names.index('border')
    out[:, :, b_dim] = 1

    for i, x in enumerate(range(*xs)):
        for j, y in enumerate(range(*ys)):
            # handle border vision
            jr = ap_size[1] - j - 1
            out[jr, i, b_dim] = 0

            c = (x, y)
            if c in objs:
                obj = objs[c]
                d = names.index(obj)
                out[jr, i, d] = 1

    return out

@nbu.njit
def get_vision_points(state: Coords, size: Size, ap_size: Size, objs: Dict[Coords, Any]) -> Dict[Coords, Coords]:
    # maps ap_coords to env_coords
    out = {}

    xs, ys = _bounds(state, size, ap_size)

    for i, x in enumerate(range(*xs)):
        for j, y in enumerate(range(*ys)):
            # handle border vision
            jr = ap_size[1] - j - 1

            ap_c = (jr, i)
            c = (x, y)

            if c in objs:
                out[ap_c] = c

    return out

@nbu.njit
def get_borders(state: Coords, size: Size, ap_size: Size):
    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    mi_x = state[0] - ax
    ma_x = min(state[0] + ax + 1, size[0])

    mi_y = max(state[1] - ay, 0)
    ma_y = min(state[1] + ay + 1, size[1])

@nbu.njit(inline='always')
def _bounds(state: Coords, size: Size, ap_size: Size):
    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    mi_x = max(state[0] - ax, 0)
    ma_x = min(state[0] + ax + 1, size[0])

    mi_y = max(state[1] - ay, 0)
    ma_y = min(state[1] + ay + 1, size[1])

    return (
        (mi_x, ma_x),
        (mi_y, ma_y),
    )
