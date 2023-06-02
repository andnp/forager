import pickle
import unittest
import numpy as np
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Flower, Wall

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

    def test_basic_movement(self):
        config = ForagerConfig(
            size=7,
            objects=[
                Wall(loc=(3, 4)),
                Wall(loc=(3, 5)),
            ],
        )
        env = ForagerEnv(config)

        self.assertEqual(env._state, (3, 3))

        # stays still when bumping into a wall
        _ = env.step(0)
        self.assertEqual(env._state, (3, 3))

        _ = env.step(1)
        self.assertEqual(env._state, (4, 3))

        _ = env.step(3)
        self.assertEqual(env._state, (3, 3))

        _ = env.step(2)
        self.assertEqual(env._state, (3, 2))

    def test_vision(self):
        config = ForagerConfig(
            size=7,
            objects=[
                Wall(loc=(3, 4)),
                Wall(loc=(3, 5)),
                Wall(loc=(0, 2)),
            ],
            aperture=3,
        )
        env = ForagerEnv(config)
        self.assertEqual(env._state, (3, 3))

        x, _ = env.step(0)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.bool_)

        self.assertEqual(env._state, (3, 3))
        self.assertTrue(np.allclose(x[:, :, 1], expected))

        env.step(1)
        x, _ = env.step(0)
        expected = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.bool_)

        self.assertEqual(env._state, (4, 4))
        self.assertTrue(np.allclose(x[:, :, 1], expected))

        env.step(1) # (5, 4)
        x, _ = env.step(1) # (6, 4)
        expected_0 = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.bool_)

        self.assertEqual(env._state, (6, 4))
        self.assertTrue(np.allclose(x[:, :, 0], expected_0))

    def test_respawn(self):
        flower = Flower()
        flower.freq = None
        flower.location = (3, 4)

        config = ForagerConfig(
            size=7,
            objects=[flower],
            aperture=3,
        )
        env = ForagerEnv(config, seed=1)
        self.assertEqual(len(env._to_respawn), 0)

        _, r = env.step(0)
        self.assertEqual(r, 1)
        self.assertEqual(len(env._object_configs), 0)
        self.assertEqual(len(env._to_respawn), 1)
        self.assertEqual(env._clock, 1)

        for _ in range(50):
            env.step(0)

        self.assertEqual(len(env._to_respawn), 1)
        self.assertEqual(env._clock, 51)

        env.step(0)
        self.assertEqual(env._clock, 52)
        self.assertEqual(len(env._object_configs), 1)
        self.assertEqual(len(env._to_respawn), 0)
