import pickle
import unittest
import numpy as np
import numba as nb
import forager._utils.numba as nbu
from forager.observations import get_vision_points

class TestObservations(unittest.TestCase):
    def test_get_vision_points(self):
        objs = nbu.Dict.empty(
            key_type=nb.types.UniTuple(nb.types.int64, 2),
            value_type=nb.typeof(''),
        )

        objs[(3, 3)] = ''
        objs[(5, 4)] = ''

        points = get_vision_points((4, 4), (8, 8), (3, 3), objs)
        points = dict(points)

        expected = {
            (2, 0): (3, 3),
            (1, 2): (5, 4),
        }

        self.assertEqual(points, expected)
