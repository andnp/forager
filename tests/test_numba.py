import forager._utils.numba as nbu

def test_ravel_unravel():
    c = (10, 3)
    size = (20, 18)

    idx = nbu.ravel(c, size)
    got = nbu.unravel(idx, size)

    assert c == got
