import numpy as np
import numba as nb
import forager._utils.numba as nbu

from forager.observations import get_color_vision

def test_color_vision():
    state = (5, 5)
    size = (10, 10)
    ap_size = (3, 3)

    idx_to_name = nbu.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.typeof(''),
    )

    name_to_color = nbu.Dict.empty(
        key_type=nb.typeof(''),
        value_type=nb.types.uint8[:],
    )

    idx_to_name[nbu.ravel((4, 4), size)] = 'a'
    idx_to_name[nbu.ravel((2, 4), size)] = 'a'
    idx_to_name[nbu.ravel((5, 6), size)] = 'b'

    a = np.array([1, 0, 0], dtype=np.uint8)
    b = np.array([0, 1, 0], dtype=np.uint8)
    e = np.array([0, 0, 0], dtype=np.uint8)

    name_to_color['a'] = a
    name_to_color['b'] = b

    got = get_color_vision(state, size, ap_size, idx_to_name, name_to_color)
    expected = np.array([
        [e, b, e],
        [e, e, e],
        [a, e, e],
    ], dtype=np.uint8)

    assert np.allclose(got, expected)
