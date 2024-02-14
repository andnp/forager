import numpy as np
import numba as nb
import forager._utils.numba as nbu

from forager.observations import get_color_vision, get_object_vision, get_world_vision


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


def test_object_vision():
    size = (10, 10)
    ap_size = (3, 3)

    idx_to_name = nbu.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.typeof(''),
    )
    name_to_dim = nbu.Dict.empty(
        key_type=nb.typeof(''),
        value_type=nb.typeof(int(1)),
    )

    name_to_dim['a'] = 0
    name_to_dim['b'] = 1

    # check middle
    idx_to_name[nbu.ravel((4, 4), size)] = 'a'
    idx_to_name[nbu.ravel((3, 4), size)] = 'a'
    idx_to_name[nbu.ravel((5, 5), size)] = 'a'
    idx_to_name[nbu.ravel((6, 5), size)] = 'b'
    got = get_object_vision((5, 5), size, ap_size, idx_to_name, name_to_dim)
    assert got.shape[2] == 2

    expected_a = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])

    expected_b = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    assert np.allclose(got[:, :, 0], expected_a)
    assert np.allclose(got[:, :, 1], expected_b)


def test_object_vision_wrap():
    size = (5, 5)
    ap_size = (3, 3)

    idx_to_name = nbu.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.typeof(''),
    )

    name_to_dim = nbu.Dict.empty(
        key_type=nb.typeof(''),
        value_type=nb.typeof(int(1)),
    )

    name_to_dim['a'] = 0
    name_to_dim['b'] = 1

    idx_to_name[nbu.ravel((3, 0), size)] = 'a'
    idx_to_name[nbu.ravel((0, 3), size)] = 'b'
    got = get_object_vision((3, 4), size, ap_size, idx_to_name, name_to_dim)
    assert got.shape[2] == 2

    expected_a = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    assert np.allclose(got[:, :, 0], expected_a)

    got = get_object_vision((4, 3), size, ap_size, idx_to_name, name_to_dim)
    assert got.shape[2] == 2

    expected_b = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])

    assert np.allclose(got[:, :, 1], expected_b)

def test_world_vision():
    size = (4, 4)

    idx_to_name = nbu.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.typeof(''),
    )
    name_to_dim = nbu.Dict.empty(
        key_type=nb.typeof(''),
        value_type=nb.typeof(int(1)),
    )

    name_to_dim['agent'] = 0
    name_to_dim['a'] = 1
    name_to_dim['b'] = 2

    # check middle
    idx_to_name[nbu.ravel((2, 2), size)] = 'a'
    idx_to_name[nbu.ravel((3, 1), size)] = 'a'
    idx_to_name[nbu.ravel((3, 0), size)] = 'b'
    got = get_world_vision((3, 1), size, idx_to_name, name_to_dim)

    excepted_agent = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert np.allclose(got[:, :, 0], excepted_agent)

    expected_a = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ])

    expected_b = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])

    assert np.allclose(got[:, :, 1], expected_a)
    assert np.allclose(got[:, :, 2], expected_b)
