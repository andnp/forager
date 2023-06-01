import unittest
import pickle
from forager.Env import ForagerEnv, ForagerConfig

class TestForagerEnv(unittest.TestCase):
    def test_pickleable(self):
        env = ForagerEnv(config=ForagerConfig(size=1, objects=[]))
        data = pickle.dumps(env)
        env2 = pickle.loads(data)
        # TODO: add equality check once api settles
        # for now, this test still ensures that things _can_ be pickled, which with numba is important
