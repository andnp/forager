import pickle
import numpy as np
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Flower, Wall

def test_pickleable():
    env = ForagerEnv(config=ForagerConfig(size=1, object_types={}))
    data = pickle.dumps(env)
    env2 = pickle.loads(data) # noqa
    # TODO: add equality check once api settles
    # for now, this test still ensures that things _can_ be pickled, which with numba is important

def test_init():
    # can specify sizes with integers
    config = ForagerConfig(
        size=8,
        object_types={},
        aperture=3,
    )
    env = ForagerEnv(config)

    assert env._state == (4, 4)
    assert env._size == (8, 8)
    assert env._ap_size == (3, 3)

    # can specify sizes as uneven tuples
    config = ForagerConfig(
        size=(10, 5),
        object_types={},
        aperture=(5, 1),
    )
    env = ForagerEnv(config)

    assert env._state == (5, 2)
    assert env._size == (10, 5)
    assert env._ap_size == (5, 1)

    # can add objects
    config = ForagerConfig(
        size=10,
        object_types={
            'flower': Flower,
        },
    )
    env = ForagerEnv(config)
    env.generate_objects(0.1, 'flower')

    assert len(env._obj_store) == int(100 * 0.1)

    # world observation mode, apeture should be None
    config = ForagerConfig(
        size=10,
        object_types={},
        observation_mode='world'
    )
    env = ForagerEnv(config)

    assert env._ap_size is None

def test_basic_movement():
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        }
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))

    assert env._state == (3, 3)

    # stays still when bumping into a wall
    _ = env.step(0)
    assert env._state == (3, 3)

    _ = env.step(1)
    assert env._state == (4, 3)

    _ = env.step(3)
    assert env._state == (3, 3)

    _ = env.step(2)
    assert env._state == (3, 2)

def test_vision():
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        },
        aperture=3,
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))
    env.add_object(Wall((0, 2)))

    assert env._state == (3, 3)

    x, _ = env.step(0)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (3, 3)
    assert np.allclose(x[:, :, 0], expected)

    env.step(1)
    x, _ = env.step(0)
    expected = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (4, 4)
    assert np.allclose(x[:, :, 0], expected)

def test_respawn():
    config = ForagerConfig(
        size=7,
        object_types={
            'flower': Flower,
        },
        aperture=3,
        seed=1,
    )
    env = ForagerEnv(config)
    env.add_object(Flower((3, 4)))
    assert len(env._to_respawn) == 0

    _, r = env.step(0)
    assert r == 1
    assert len(env._obj_store) == 0
    assert len(env._to_respawn) == 1
    assert env._clock == 1

    for _ in range(50):
        env.step(0)

    assert len(env._to_respawn) == 1
    assert env._clock == 51

    env.step(0)
    assert env._clock == 52
    assert len(env._obj_store) == 1
    assert len(env._to_respawn) == 0

def test_wrapping_dynamics():
    config = ForagerConfig(
        size=5,
        object_types={
        },
        aperture=3,
        seed=1
    )
    env = ForagerEnv(config)

    # go up
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(0)
    assert env._state == (2, 3)
    _ = env.step(0)
    assert env._state == (2, 4)
    _ = env.step(0)
    assert env._state == (2, 0)
    _ = env.step(0)
    assert env._state == (2, 1)
    _ = env.step(0)
    assert env._state == (2, 2)

    # go down
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(2)
    assert env._state == (2, 1)
    _ = env.step(2)
    assert env._state == (2, 0)
    _ = env.step(2)
    assert env._state == (2, 4)
    _ = env.step(2)
    assert env._state == (2, 3)
    _ = env.step(2)
    assert env._state == (2, 2)

    # go right
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(1)
    assert env._state == (3, 2)
    _ = env.step(1)
    assert env._state == (4, 2)
    _ = env.step(1)
    assert env._state == (0, 2)
    _ = env.step(1)
    assert env._state == (1, 2)
    _ = env.step(1)
    assert env._state == (2, 2)

    # go left
    _ = env.start()
    assert env._state == (2, 2)
    _ = env.step(3)
    assert env._state == (1, 2)
    _ = env.step(3)
    assert env._state == (0, 2)
    _ = env.step(3)
    assert env._state == (4, 2)
    _ = env.step(3)
    assert env._state == (3, 2)
    _ = env.step(3)
    assert env._state == (2, 2)

def test_wrapping_vision():
    config = ForagerConfig(
        size=5,
        object_types={
            'flower': Flower,
        },
        aperture=3,
        seed=1
    )
    env = ForagerEnv(config)
    env.add_object(Flower((0, 0)))

    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    x = env.start()

    assert env._state == (2, 2)
    assert np.allclose(x[:, :, 0], expected)

    # go left
    _ = env.step(3)
    # go down
    x, _ = env.step(2)

    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (1, 1)
    assert np.allclose(x[:, :, 0], expected)

    # go left , go left
    _ = env.step(3)
    x, _ = env.step(3)
    expected = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=np.bool_)

    assert env._state == (4, 1)
    assert np.allclose(x[:, :, 0], expected)


# ----------------------------
# -- Performance benchmarks --
# ----------------------------
def test_benchmark_vision(benchmark):
    config = ForagerConfig(
        size=7,
        object_types={
            'wall': Wall,
        },
        aperture=3,
    )
    env = ForagerEnv(config)
    env.add_object(Wall((3, 4)))
    env.add_object(Wall((3, 5)))
    env.add_object(Wall((0, 2)))

    assert env._state == (3, 3)

    x, _ = benchmark(env.step, 0)
    expected = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.bool_)

    assert env._state == (3, 3)
    assert np.allclose(x[:, :, 0], expected)

def test_benchmark_creation(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=31,
    )

    def _build(config):
        env = ForagerEnv(config)
        env.generate_objects(0.05, 'wall')
        env.generate_objects(0.05, 'flower')

    benchmark(_build, config)

def test_benchmark_small_env(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=15,
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_big_env(benchmark):
    config = ForagerConfig(
        size=10_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=61,
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_small_env_color(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        aperture=15,
        observation_mode='colors',
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)

def test_benchmark_small_env_world(benchmark):
    config = ForagerConfig(
        size=1_000,
        object_types={
            'wall': Wall,
            'flower': Flower,
        },
        observation_mode='world',
    )

    env = ForagerEnv(config)
    env.generate_objects(0.05, 'wall')
    env.generate_objects(0.05, 'flower')

    def _run(env):
        for _ in range(100):
            env.step(0)

    benchmark(_run, env)
