import numpy as np
import forager._utils.numba as nbu

from typing import Dict
from forager.interface import Coords, Size


@nbu.njit
def get_color_vision(state: Coords, size: Size, ap_size: Size, idx_to_name: Dict[int, str], name_to_color: Dict[str, np.ndarray]):
    out = np.zeros((ap_size[0], ap_size[1], 3), dtype=np.uint8)

    xs, ys = _bounds(state, size, ap_size)
    out[:, :] = 255

    for i, x in enumerate(range(*xs)):
        for j, y in enumerate(range(*ys)):
            # handle border vision
            jr = ap_size[1] - j - 1
            out[jr, i] = 0

            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in idx_to_name:
                name = idx_to_name[idx]
                color = name_to_color[name]
                out[jr, i] = color

    return out

@nbu.njit
def get_object_vision(state: Coords, size: Size, ap_size: Size, objs: Dict[Coords, str], names: Dict[str, int]):
    dims = len(names)
    out = np.zeros((ap_size[0], ap_size[1], dims), dtype=np.bool_)

    xs, ys = _bounds(state, size, ap_size)
    b_dim = names['border']
    out[:, :, b_dim] = 1

    for i, x in enumerate(range(*xs)):
        for j, y in enumerate(range(*ys)):
            # handle border vision
            jr = ap_size[1] - j - 1
            out[jr, i, b_dim] = 0

            c = (x, y)
            idx = nbu.ravel(c, size)
            if idx in objs:
                obj = objs[idx]
                d = names[obj]
                out[jr, i, d] = 1

    return out


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
