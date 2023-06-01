import unittest
import pickle
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects.Flower import Flower

class TestForagerEnv(unittest.TestCase):
    def test_pickleable(self):
        env = ForagerEnv(config=ForagerConfig(size=1, objects=[]))
        data = pickle.dumps(env)
        env2 = pickle.loads(data)
        # TODO: add equality check once api settles
        # for now, this test still ensures that things _can_ be pickled, which with numba is important

    def test_init(self):
        # can specify sizes with integers
        config = ForagerConfig(
            size=8,
            objects=[],
            aperture=3,
        )
        env = ForagerEnv(config)

        self.assertEqual(env._state, (4, 4))
        self.assertEqual(env._size, (8, 8))
        self.assertEqual(env._ap_size, (3, 3))

        # can specify sizes as uneven tuples
        config = ForagerConfig(
            size=(10, 5),
            objects=[],
            aperture=(5, 1),
        )
        env = ForagerEnv(config)

        self.assertEqual(env._state, (5, 2))
        self.assertEqual(env._size, (10, 5))
        self.assertEqual(env._ap_size, (5, 1))

        # can add objects
        flower = Flower()
        flower.freq = 0.1
        config = ForagerConfig(
            size=10,
            objects=[
                flower
            ],
        )
        env = ForagerEnv(config)

        self.assertEqual(len(env._object_configs), int(100 * 0.1))
