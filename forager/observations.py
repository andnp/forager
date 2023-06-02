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

    mi_x = max(state[0] - ax, 0)
    ma_x = min(state[0] + ax + 1, size[0])

    mi_y = max(state[1] - ay, 0)
    ma_y = min(state[1] + ay + 1, size[1])

    b_dim = names.index('border')
    out[:, :, b_dim] = 1

    for i, x in enumerate(range(mi_x, ma_x)):
        for j, y in enumerate(range(mi_y, ma_y)):
            # handle border vision
            jr = ap_size[1] - j - 1
            out[jr, i, b_dim] = 0

            c = (x, y)
            if c in objs:
                obj = objs[c]
                d = names.index(obj)
                out[jr, i, d] = 1

    return out
