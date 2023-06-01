import numpy as np
import forager._utils.numba as nbu

from typing import Dict, List
from forager.interface import Coords, Size

@nbu.njit
def get_vision_by_name(state: Coords, size: Size, ap_size: Size, objs: Dict[Coords, str], names: List[str]):
    dims = len(names)
    out = np.zeros((ap_size[0], ap_size[1], dims), dtype=np.bool_)

    ax = int(ap_size[0] // 2)
    ay = int(ap_size[1] // 2)

    mi_x = max(state[0] - ax - 1, 0)
    ma_x = min(state[0] + ax, size[0])

    mi_y = max(state[1] - ay - 1, 0)
    ma_y = min(state[1] + ay, size[1])

    for i, x in enumerate(range(mi_x, ma_x)):
        for j, y in enumerate(range(mi_y, ma_y)):
            c = (x, y)
            print(c)
            # idx = ravel_idx(c, size)
            if c in objs:
                obj = objs[c]
                d = names.index(obj)
                print(d, obj)
                out[i, j, d] = 1

    return out
